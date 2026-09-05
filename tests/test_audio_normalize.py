import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from polyphon.audio import has_start_offset, normalize_to_wav


def _probe_output(value: str):
    return value.encode()


def test_has_start_offset_detects_container_priming_delay():
    # MP4/AAC files report a non-zero stream start_time (encoder priming delay).
    with patch("subprocess.check_output", return_value=_probe_output("0.044000")):
        assert has_start_offset("x.mp4") is True


def test_has_start_offset_false_for_aligned_and_unknown():
    for probe in ("0.000000", "N/A", ""):
        with patch("subprocess.check_output", return_value=_probe_output(probe)):
            assert has_start_offset("x.wav") is False


def test_has_start_offset_survives_missing_ffprobe():
    # A missing ffprobe must not break diarization; assume no offset.
    with patch("subprocess.check_output", side_effect=OSError("no ffprobe")):
        assert has_start_offset("x.mp4") is False

    with patch("subprocess.check_output", side_effect=subprocess.SubprocessError()):
        assert has_start_offset("x.mp4") is False


def test_has_start_offset_ignores_unparseable_value():
    with patch("subprocess.check_output", return_value=_probe_output("not-a-number")):
        assert has_start_offset("x.mp4") is False


def test_normalize_to_wav_raises_on_ffmpeg_failure(tmp_path):
    with patch(
        "subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "ffmpeg"),
    ):
        with pytest.raises(RuntimeError, match="Could not normalize"):
            normalize_to_wav(tmp_path / "in.mp4", target_dir=tmp_path)


def test_normalize_to_wav_targets_requested_directory(tmp_path):
    with patch("subprocess.run") as run:
        dest = normalize_to_wav("some/input.mp4", target_dir=tmp_path)

    assert dest == tmp_path / "input.normalized.wav"
    cmd = run.call_args[0][0]
    # 16 kHz mono PCM, with the resampler resetting the timestamp origin.
    assert "pcm_s16le" in cmd
    assert "16000" in cmd
    assert "aresample=async=1" in cmd
    assert str(Path("some/input.mp4")) in cmd


def test_diarizer_normalizes_only_when_offset_present():
    from unittest.mock import MagicMock

    from polyphon.diarization.base import PyannoteDiarizer

    diarizer = PyannoteDiarizer()
    pipeline = MagicMock(return_value=MagicMock(itertracks=lambda **kw: []))

    with patch.object(diarizer, "_get_pipeline", return_value=pipeline):
        # An offset-free file must be passed straight through: no conversion cost.
        with (
            patch("polyphon.diarization.base.has_start_offset", return_value=False),
            patch("polyphon.diarization.base.normalize_to_wav") as convert,
        ):
            diarizer.diarize("clean.wav", show_progress=False)
        assert not convert.called
        assert pipeline.call_args[0][0] == "clean.wav"

        # A file with priming delay must be converted first.
        with (
            patch("polyphon.diarization.base.has_start_offset", return_value=True),
            patch("polyphon.diarization.base.normalize_to_wav", return_value="converted.wav") as convert,
        ):
            diarizer.diarize("voice-note.mp4", show_progress=False)
        assert convert.called
        assert pipeline.call_args[0][0] == "converted.wav"


def test_diarizer_falls_back_when_normalization_fails():
    """A failed repair must warn and continue, not abort diarization."""
    from unittest.mock import MagicMock

    from polyphon.diarization.base import PyannoteDiarizer

    diarizer = PyannoteDiarizer()
    pipeline = MagicMock(return_value=MagicMock(itertracks=lambda **kw: []))

    with (
        patch.object(diarizer, "_get_pipeline", return_value=pipeline),
        patch("polyphon.diarization.base.has_start_offset", return_value=True),
        patch("polyphon.diarization.base.normalize_to_wav", side_effect=RuntimeError("ffmpeg exploded")),
        pytest.warns(RuntimeWarning, match="Could not normalize"),
    ):
        diarizer.diarize("voice-note.mp4", show_progress=False)

    assert pipeline.call_args[0][0] == "voice-note.mp4"
