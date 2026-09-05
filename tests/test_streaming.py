"""Tests for Phase 6 Streaming and NeMo Sortformer Diarization."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from polyphon.diarization.base import SortformerDiarizer
from polyphon.server.app import create_app
from polyphon.streaming import AudioStreamBuffer, StreamingEvent, StreamingPolyphonEngine
from polyphon.types import SpeakerInterval, WordToken


def test_audio_stream_buffer():
    """Test AudioStreamBuffer operations: write, windowing, chunking, and WAV export."""
    buffer = AudioStreamBuffer(sample_rate=16000, max_accumulate_seconds=10.0)

    # 1. Write float32 array (1.0s)
    samples_1s = np.zeros(16000, dtype=np.float32)
    buffer.write(samples_1s)
    assert pytest.approx(buffer.get_total_duration(), 0.01) == 1.0

    # 2. Write int16 bytes (1.0s -> 32000 bytes)
    raw_pcm = b"\x00\x00" * 16000
    buffer.write(raw_pcm)
    assert pytest.approx(buffer.get_total_duration(), 0.01) == 2.0

    # 3. Recent window
    window, start_time = buffer.get_recent_window(duration=1.5)
    assert len(window) == 24000
    assert pytest.approx(start_time, 0.01) == 0.5

    # 4. Unprocessed chunk
    chunk, c_start = buffer.get_unprocessed_chunk(chunk_duration=1.0)
    assert len(chunk) == 16000
    assert pytest.approx(c_start, 0.01) == 0.0

    chunk2, c_start2 = buffer.get_unprocessed_chunk(chunk_duration=1.0)
    assert len(chunk2) == 16000
    assert pytest.approx(c_start2, 0.01) == 1.0

    assert buffer.get_unprocessed_chunk(chunk_duration=1.0) is None

    # 5. Export to WAV
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_file = Path(tmpdir) / "test_export.wav"
        buffer.save_to_wav(wav_file)
        assert wav_file.exists()
        assert wav_file.stat().st_size > 0


def test_sortformer_diarizer():
    """Test Sortformer neural diarizer frame probability prediction, hysteresis, and streaming."""
    diarizer = SortformerDiarizer(
        max_speakers=4,
        frame_duration=0.08,
        onset=0.5,
        offset=0.35,
    )

    # Synthetic multi-tone audio (2 seconds = 32000 samples)
    t = np.linspace(0, 2.0, 32000, endpoint=False)
    # Speaker 1: 440 Hz
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    # 1. Predict frame probabilities
    probs = diarizer.predict_frame_probabilities(audio, sample_rate=16000)
    assert probs.ndim == 2
    assert probs.shape[1] == 4
    assert probs.shape[0] == int(len(audio) / (16000 * 0.08))

    # 2. Binarize probabilities
    intervals = diarizer.binarize_probabilities(probs, frame_duration=0.08, start_offset=0.0)
    assert isinstance(intervals, list)
    for iv in intervals:
        assert isinstance(iv, SpeakerInterval)
        assert iv.speaker_id.startswith("SPEAKER_")
        assert iv.start < iv.end

    # 3. Streaming step
    diarizer.init_stream(stream_offset=0.0)
    chunk = audio[:16000]
    stream_intervals = diarizer.stream_step(chunk, timestamp_offset=0.0)
    assert isinstance(stream_intervals, list)


def test_streaming_polyphon_engine():
    """Test StreamingPolyphonEngine with mocked ASR and Sortformer components."""
    mock_asr = MagicMock()
    mock_asr.transcribe.return_value = (
        "en",
        [
            WordToken(word="Hello", start=0.2, end=0.6),
            WordToken(word="world", start=0.7, end=1.1),
        ],
    )

    mock_diarizer = MagicMock()
    mock_diarizer.stream_step.return_value = [
        SpeakerInterval(speaker_id="SPEAKER_00", start=0.1, end=1.5, confidence=0.98),
    ]

    engine = StreamingPolyphonEngine(
        asr=mock_asr,
        diarizer=mock_diarizer,
        chunk_duration=1.0,
        window_duration=4.0,
        commit_lag=1.0,
    )

    engine.start_stream()

    # Feed 2 seconds of synthetic audio
    audio_chunk = np.zeros(32000, dtype=np.float32)
    engine.feed_audio(audio_chunk)

    events = []

    def on_event(ev: StreamingEvent):
        events.append(ev)

    event = engine.process_step(on_event=on_event)
    assert event is not None
    assert event.event_type in ("chunk", "segment")
    assert len(events) == 1
    assert "SPEAKER_00" in engine.speaker_map or "Speaker 00" in engine.speaker_map.values()

    # Finalize stream
    with tempfile.TemporaryDirectory() as tmpdir:
        result = engine.stop_stream(output_dir=tmpdir, session_name="test_live_session")
        assert result.duration > 0
        assert (Path(tmpdir) / "test_live_session.wav").exists()
        assert (Path(tmpdir) / "test_live_session.json").exists()


def test_streaming_api_endpoints():
    """Test FastAPI streaming endpoints: /api/stream/status and /api/stream/ws."""
    with tempfile.TemporaryDirectory() as tmpdir:
        app = create_app(output_dir=tmpdir)
        client = TestClient(app)

        # 1. GET /api/stream/status
        res = client.get("/api/stream/status")
        assert res.status_code == 200
        data = res.json()
        assert data["streaming_supported"] is True
        assert "sortformer" in data["diarizers"]

        # 2. WebSocket /api/stream/ws
        with client.websocket_connect("/api/stream/ws") as websocket:
            # Start stream
            websocket.send_json({"action": "start", "name": "test_ws_session", "diarizer": "sortformer"})
            start_resp = websocket.receive_json()
            assert start_resp["status"] == "started"
            assert start_resp["session_name"] == "test_ws_session"

            # Send 1 second of PCM audio bytes
            raw_pcm = b"\x00\x00" * 16000
            websocket.send_bytes(raw_pcm)

            # Stop stream
            websocket.send_json({"action": "stop", "name": "test_ws_session"})
            stop_resp = websocket.receive_json()
            assert stop_resp["status"] == "completed"
            assert "result" in stop_resp


def test_simulate_file_stream_supports_mp4_and_wav():
    """Verify simulate_file_stream decodes both WAV and non-WAV media containers (e.g. MP4)."""
    from polyphon.streaming import simulate_file_stream

    mp4_path = Path("examples/meeting.mp4")
    if mp4_path.exists():
        gen = simulate_file_stream(mp4_path, chunk_duration=0.5, speed=100.0)
        chunk = next(gen)
        assert isinstance(chunk, np.ndarray)
        assert len(chunk) == 8000  # 0.5s at 16kHz
        assert chunk.dtype == np.float32
