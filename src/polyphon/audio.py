"""Audio utilities for format normalization and preprocessing."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def get_audio_duration(file_path: str | Path) -> float:
    """Get duration of audio file in seconds. Supports standard WAV natively."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                if rate > 0:
                    return float(frames) / float(rate)
        except (wave.Error, OSError):
            pass

    # Fallback to ffprobe for MP4, MP3, M4A, etc.
    try:
        import subprocess

        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
        res = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        if res:
            return float(res)
    except (subprocess.SubprocessError, OSError, ValueError):
        pass

    return 0.0


def load_wav_mono_16k(file_path: str | Path) -> tuple[np.ndarray, int]:
    """Read any audio or video file, resample to 16kHz mono float32 array in [-1.0, 1.0].

    Supports WAV natively, and MP4, WEBM, MKV, MP3, M4A, FLAC, AAC via FFmpeg / torchcodec.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    # Fast path: standard 16kHz mono 16-bit WAV files
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as wf:
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                if framerate == 16000 and sampwidth in (1, 2, 4):
                    n_frames = wf.getnframes()
                    raw_data = wf.readframes(n_frames)
                    if sampwidth == 2:
                        audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
                    elif sampwidth == 4:
                        audio = np.frombuffer(raw_data, dtype=np.int32).astype(np.float32) / 2147483648.0
                    else:
                        audio = (np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
                    if n_channels > 1:
                        audio = audio.reshape(-1, n_channels).mean(axis=1)
                    return audio, 16000
        except Exception:
            pass  # Fall back to ffmpeg/torchcodec for other sample rates, formats, or corrupted headers

    # Universal decoding & 16kHz mono resampling via FFmpeg
    import shutil
    import subprocess

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        cmd = [
            ffmpeg_bin,
            "-nostdin",
            "-threads",
            "0",
            "-i",
            str(path),
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-",
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, check=True)
            audio = np.frombuffer(res.stdout, dtype=np.int16).astype(np.float32) / 32768.0
            return audio, 16000
        except subprocess.SubprocessError:
            pass

    # Fallback to torchcodec if ffmpeg CLI binary is unavailable
    try:
        from torchcodec.decoders import AudioDecoder

        decoder = AudioDecoder(str(path))
        samples = decoder.get_all_samples()
        audio = samples.data.mean(dim=0).cpu().numpy().astype(np.float32)
        sr = int(decoder.metadata.sample_rate)
        if sr != 16000:
            target_len = int(len(audio) * 16000 / sr)
            audio = np.interp(
                np.linspace(0, len(audio), target_len, endpoint=False),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)
        return audio, 16000
    except Exception as err:
        raise RuntimeError(f"Could not load and decode audio from {path}. Ensure FFmpeg is installed.") from err


def has_start_offset(file_path: str | Path, tolerance: float = 0.001) -> bool:
    """True if the audio stream does not start at t=0.

    MP4/AAC containers (WhatsApp voice notes, phone recordings, screen captures)
    carry encoder priming delay, so the stream begins a few milliseconds late.
    pyannote requests fixed-length chunks and allows only a one-sample tolerance,
    so such a file fails with "resulted in N samples instead of the expected M".
    """
    import subprocess

    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=start_time",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(file_path),
            ],
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.SubprocessError, OSError):
        return False

    raw = out.decode().strip()
    if not raw or raw == "N/A":
        return False
    try:
        return abs(float(raw)) > tolerance
    except ValueError:
        return False


def normalize_to_wav(file_path: str | Path, target_dir: str | Path | None = None) -> Path:
    """Re-encode audio to 16 kHz mono PCM WAV, dropping any container start offset.

    Returns the path to the converted file. Raises RuntimeError if FFmpeg fails.
    """
    import subprocess
    import tempfile

    src = Path(file_path)
    out_dir = Path(target_dir) if target_dir else Path(tempfile.mkdtemp(prefix="polyphon_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{src.stem}.normalized.wav"

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        # -af aresample=async=1 resets the timestamp origin so the output starts at 0.
        "-af",
        "aresample=async=1",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(dest),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except (subprocess.CalledProcessError, OSError) as err:
        raise RuntimeError(f"Could not normalize {src.name} with FFmpeg: {err}") from err

    return dest


load_audio_mono_16k = load_wav_mono_16k
