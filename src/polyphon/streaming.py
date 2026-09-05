"""Streaming Engine and Audio Stream Processing for Polyphon."""

from __future__ import annotations

import io
import threading
import time
import wave
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from polyphon.asr.base import ASRBackend, FasterWhisperASR
from polyphon.audio import load_wav_mono_16k
from polyphon.diarization.base import DiarizationBackend, SortformerDiarizer
from polyphon.reconciliation.base import ReconciliationEngine
from polyphon.types import PolyphonResult, Segment, SpeakerInfo, WordToken
from polyphon.voicedb.base import VoiceDB


@dataclass
class StreamingEvent:
    """Event emitted during real-time streaming."""

    event_type: str  # "chunk", "segment", "speaker_identified", "complete"
    stream_time: float
    active_speaker: str
    active_speaker_id: str
    text: str
    is_final: bool = False
    segments: list[Segment] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "stream_time": round(self.stream_time, 2),
            "active_speaker": self.active_speaker,
            "active_speaker_id": self.active_speaker_id,
            "text": self.text,
            "is_final": self.is_final,
            "segments": [
                {
                    "start": round(s.start, 2),
                    "end": round(s.end, 2),
                    "speaker_id": s.speaker_id,
                    "speaker_name": s.speaker_name,
                    "text": s.text,
                    "words": [
                        {
                            "word": w.word,
                            "start": round(w.start, 2),
                            "end": round(w.end, 2),
                        }
                        for w in s.words
                    ],
                }
                for s in self.segments
            ],
            "metadata": self.metadata,
        }


class AudioStreamBuffer:
    """Thread-safe circular and accumulating buffer for 16kHz float32 mono audio."""

    def __init__(self, sample_rate: int = 16000, max_accumulate_seconds: float = 3600.0):
        self.sample_rate = sample_rate
        self.max_samples = int(max_accumulate_seconds * sample_rate)
        self._buffer = np.zeros(0, dtype=np.float32)
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._processed_samples = 0

    def _consolidate_locked(self) -> None:
        """Consolidate pending chunks into the continuous buffer (must hold _lock)."""
        if not self._chunks:
            return
        if len(self._buffer) == 0:
            self._buffer = np.concatenate(self._chunks)
        else:
            self._buffer = np.concatenate([self._buffer] + self._chunks)
        self._chunks.clear()

        if len(self._buffer) > self.max_samples:
            overflow = len(self._buffer) - self.max_samples
            self._buffer = self._buffer[overflow:]
            self._processed_samples = max(0, self._processed_samples - overflow)

    def write(self, data: np.ndarray | bytes) -> int:
        """Append audio data (float32 array, int16 bytes, or WebM/PCM chunk)."""
        if isinstance(data, bytes):
            # Fast C-level buffer decoding (zero-copy view converted to float32)
            if len(data) % 2 == 0:
                arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                arr = np.frombuffer(data, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
        elif isinstance(data, np.ndarray):
            arr = data.astype(np.float32)
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
        else:
            raise TypeError(f"Unsupported audio data type: {type(data)}")

        if len(arr) == 0:
            return 0

        with self._lock:
            self._chunks.append(arr)
            # Consolidate if pending chunks exceed 1 second
            if sum(len(c) for c in self._chunks) >= self.sample_rate:
                self._consolidate_locked()

        return len(arr)

    def get_total_duration(self) -> float:
        """Total duration of buffered audio in seconds."""
        with self._lock:
            self._consolidate_locked()
            return float(len(self._buffer)) / float(self.sample_rate)

    def get_all_samples(self) -> np.ndarray:
        """Get copy of all accumulated samples."""
        with self._lock:
            self._consolidate_locked()
            return self._buffer.copy()

    def get_recent_window(self, duration: float) -> tuple[np.ndarray, float]:
        """Get the most recent audio window of up to `duration` seconds.
        Returns (samples, start_timestamp).
        """
        with self._lock:
            self._consolidate_locked()
            total_samples = len(self._buffer)
            target_samples = int(duration * self.sample_rate)
            if total_samples <= target_samples:
                return self._buffer.copy(), 0.0
            start_idx = total_samples - target_samples
            start_time = float(start_idx) / float(self.sample_rate)
            return self._buffer[start_idx:].copy(), start_time

    def get_unprocessed_chunk(self, chunk_duration: float = 1.0) -> tuple[np.ndarray, float] | None:
        """Get the next unprocessed chunk. Returns (chunk, start_time) or None if not enough samples."""
        chunk_samples = int(chunk_duration * self.sample_rate)
        with self._lock:
            self._consolidate_locked()
            available = len(self._buffer) - self._processed_samples
            if available < chunk_samples:
                return None
            start_idx = self._processed_samples
            end_idx = start_idx + chunk_samples
            chunk = self._buffer[start_idx:end_idx].copy()
            self._processed_samples = end_idx
            start_time = float(start_idx) / float(self.sample_rate)
            return chunk, start_time

    def save_to_wav(self, output_path: str | Path) -> Path:
        """Save all accumulated audio to a 16kHz 16-bit mono WAV file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        samples = self.get_all_samples()
        int16_samples = np.clip(samples * 32767.0, -32768.0, 32767.0).astype(np.int16)

        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(int16_samples.tobytes())

        return path


class StreamingPolyphonEngine:
    """Unified engine for real-time streaming speech recognition and neural diarization."""

    def __init__(
        self,
        asr: ASRBackend | None = None,
        asr_model: str = "base",
        language: str | None = "en",
        diarizer: DiarizationBackend | None = None,
        voicedb: VoiceDB | None = None,
        reconciler: ReconciliationEngine | None = None,
        chunk_duration: float = 1.5,
        window_duration: float = 6.0,
        commit_lag: float = 2.0,
    ):
        self.language = language
        self.asr = asr or FasterWhisperASR(model_size=asr_model)
        self.diarizer = diarizer or SortformerDiarizer()
        self.voicedb = voicedb or VoiceDB()
        self.reconciler = reconciler or ReconciliationEngine()

        self.chunk_duration = chunk_duration
        self.window_duration = window_duration
        self.commit_lag = commit_lag

        self.buffer = AudioStreamBuffer()
        self.committed_segments: list[Segment] = []
        self.speculative_segment: Segment | None = None
        self.speaker_map: dict[str, str] = {}
        self.speaker_infos: list[SpeakerInfo] = []
        self.speaker_accumulated_audio: dict[str, list[np.ndarray]] = {}

        self._active_stream = False
        self._lock = threading.Lock()

    def start_stream(self) -> None:
        """Reset and start a new streaming session."""
        with self._lock:
            self.buffer = AudioStreamBuffer()
            self.committed_segments.clear()
            self.speculative_segment = None
            self.speaker_map.clear()
            self.speaker_infos.clear()
            self.speaker_accumulated_audio.clear()
            self._active_stream = True

            if hasattr(self.diarizer, "init_stream"):
                self.diarizer.init_stream(0.0)

    def feed_audio(self, data: np.ndarray | bytes) -> None:
        """Feed raw audio samples into the stream buffer."""
        self.buffer.write(data)

    def process_step(self, on_event: Callable[[StreamingEvent], None] | None = None) -> StreamingEvent | None:
        """Process available audio in the buffer and emit a streaming event."""
        with self._lock:
            if not self._active_stream:
                return None

            total_dur = self.buffer.get_total_duration()
            # We process sliding window of recent audio
            window_audio, window_start = self.buffer.get_recent_window(self.window_duration)
            if len(window_audio) < int(self.chunk_duration * 16000):
                return None

        # Run Diarizer on the recent window
        if hasattr(self.diarizer, "stream_step"):
            intervals = self.diarizer.stream_step(window_audio, timestamp_offset=window_start)
        else:
            # Fallback to general diarize
            intervals = []

        # Determine dominant active speaker in the most recent slice
        active_speaker_id = "SPEAKER_00"
        active_speaker_name = self.speaker_map.get(active_speaker_id, "Speaker 00")

        if intervals:
            # Find interval covering the latest point
            latest_t = window_start + (len(window_audio) / 16000.0)
            matching = [iv for iv in intervals if iv.start <= latest_t <= (iv.end + 0.3)]
            if matching:
                active_speaker_id = matching[-1].speaker_id
            else:
                active_speaker_id = intervals[-1].speaker_id

            if active_speaker_id not in self.speaker_map:
                self.speaker_map[active_speaker_id] = active_speaker_id.replace("SPEAKER_", "Speaker ")
                self.speaker_infos.append(
                    SpeakerInfo(id=active_speaker_id, name=self.speaker_map[active_speaker_id], confidence=1.0)
                )

            active_speaker_name = self.speaker_map[active_speaker_id]

        # Transcribe window only if audio contains discernible energy or active speaker intervals
        max_amp = float(np.max(np.abs(window_audio))) if len(window_audio) > 0 else 0.0
        words: list[WordToken] = []

        if max_amp >= 0.015 and (intervals or max_amp >= 0.025):
            # In-memory WAV for ASR chunk
            mem_wav = io.BytesIO()
            int16_samples = np.clip(window_audio * 32767.0, -32768.0, 32767.0).astype(np.int16)
            with wave.open(mem_wav, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(int16_samples.tobytes())
            mem_wav.seek(0)

            try:
                _, words = self.asr.transcribe(mem_wav, progress_callback=None, language=self.language)
            except Exception:
                words = []

            # Filter out single-word hallucinations on borderline low-energy audio chunks
            if len(words) == 1 and max_amp < 0.04:
                clean_w = words[0].word.strip().lower().rstrip(".,!?")
                if clean_w in {"oh", "ah", "uh", "um", "you", "bye", "thanks", "thank you", "subtitles", "transcript"}:
                    words = []

        # Adjust word timestamps to absolute stream time
        shifted_words: list[WordToken] = [
            WordToken(
                word=w.word,
                start=round(window_start + w.start, 2),
                end=round(window_start + w.end, 2),
                score=w.score,
            )
            for w in words
        ]

        # Reconcile with intervals
        step_segments = self.reconciler.reconcile(shifted_words, intervals, self.speaker_map)

        # Separate committed vs speculative segments based on commit horizon
        committed_horizon = max(0.0, total_dur - self.commit_lag)
        new_committed: list[Segment] = []
        speculative: Segment | None = None

        for seg in step_segments:
            if seg.end <= committed_horizon:
                # Check if not already committed
                if not any(
                    abs(c.start - seg.start) < 0.3 and abs(c.end - seg.end) < 0.3 for c in self.committed_segments
                ):
                    self.committed_segments.append(seg)
                    new_committed.append(seg)
            else:
                speculative = seg

        self.speculative_segment = speculative

        # Construct current visible segments
        all_display_segments = list(self.committed_segments)
        if speculative:
            all_display_segments.append(speculative)

        latest_text = speculative.text if speculative else (new_committed[-1].text if new_committed else "")

        event = StreamingEvent(
            event_type="segment" if new_committed else "chunk",
            stream_time=total_dur,
            active_speaker=active_speaker_name,
            active_speaker_id=active_speaker_id,
            text=latest_text,
            is_final=bool(new_committed),
            segments=all_display_segments,
            metadata={
                "committed_count": len(self.committed_segments),
                "is_speculative": speculative is not None,
            },
        )

        if on_event:
            on_event(event)

        return event

    def stop_stream(
        self,
        output_dir: str | Path = "outputs",
        session_name: str | None = None,
        summarize: bool = False,
    ) -> PolyphonResult:
        """Finalize the stream, commit all pending segments, save audio and JSON result."""
        with self._lock:
            self._active_stream = False
            total_dur = self.buffer.get_total_duration()

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        name = session_name or f"stream_{int(time.time())}"

        # Save audio file
        wav_path = out_dir / f"{name}.wav"
        self.buffer.save_to_wav(wav_path)

        # If speculative segment remains, commit it
        if self.speculative_segment:
            self.committed_segments.append(self.speculative_segment)
            self.speculative_segment = None

        # Build final PolyphonResult
        result = PolyphonResult(
            duration=total_dur,
            language=self.language or "en",
            speakers=self.speaker_infos,
            segments=self.committed_segments,
        )

        # Save JSON
        json_path = out_dir / f"{name}.json"
        json_path.write_text(result.to_json(), encoding="utf-8")

        return result


def simulate_file_stream(
    file_path: str | Path,
    chunk_duration: float = 1.0,
    speed: float = 1.0,
) -> Generator[np.ndarray, None, None]:
    """Simulate a live audio stream by yielding real-time chunks from an audio file."""
    audio, sr = load_wav_mono_16k(file_path)
    chunk_samples = int(chunk_duration * sr)
    sleep_time = (chunk_duration / speed) if speed > 0 else 0.0

    total_samples = len(audio)
    for start in range(0, total_samples, chunk_samples):
        chunk = audio[start : start + chunk_samples]
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
        yield chunk
        if sleep_time > 0:
            time.sleep(sleep_time)


def capture_mic_stream(
    sample_rate: int = 16000,
    chunk_duration: float = 1.0,
) -> Generator[np.ndarray, None, None]:
    """Capture audio in real-time from the default microphone."""
    chunk_samples = int(chunk_duration * sample_rate)

    # Try sounddevice first
    try:
        import sounddevice as sd

        def callback(indata, frames, time_info, status):
            pass

        with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32") as stream:
            while True:
                data, overflowed = stream.read(chunk_samples)
                yield data.flatten()
        return
    except (ImportError, Exception):
        pass

    # Try pyaudio
    try:
        import pyaudio

        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=chunk_samples,
        )
        while True:
            raw = stream.read(chunk_samples, exception_on_overflow=False)
            data = np.frombuffer(raw, dtype=np.float32)
            yield data
        return
    except (ImportError, Exception):
        pass

    # Fallback to FFmpeg stdin/mic capture
    import subprocess

    # Detect platform capture device
    import sys

    if sys.platform.startswith("linux"):
        input_fmt = "alsa"
        dev = "default"
    elif sys.platform == "darwin":
        input_fmt = "avfoundation"
        dev = ":0"
    else:
        input_fmt = "dshow"
        dev = "audio=Microphone"

    cmd = [
        "ffmpeg",
        "-f",
        input_fmt,
        "-i",
        dev,
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "-",
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        bytes_per_chunk = chunk_samples * 4
        while True:
            raw = proc.stdout.read(bytes_per_chunk)
            if not raw or len(raw) < bytes_per_chunk:
                break
            data = np.frombuffer(raw, dtype=np.float32)
            yield data
    except Exception as e:
        raise RuntimeError(
            f"No microphone capture backend available (sounddevice, pyaudio, or ffmpeg {input_fmt}): {e}"
        ) from e
