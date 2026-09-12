from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from polyphon.cli import app

# `doctor` probes the real audio stack, and these tests patch torchcodec
# directly. torchcodec arrives with the [diarization] extra, so skip rather
# than fail in lean environments (such as the CI test matrix) without it.
pytest.importorskip("torchcodec")

runner = CliRunner()


def test_doctor_reports_healthy_environment():
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "FFmpeg shared libraries loadable" in result.stdout


def test_doctor_fails_when_ffmpeg_libraries_missing():
    # An import alone would succeed here: torchcodec swallows the FFmpeg load
    # error to keep its image decoders usable. doctor must decode, so simulate
    # the failure at the decode step.
    error = RuntimeError("Could not load libtorchcodec. Likely causes: ...")

    with patch("torchcodec.encoders.AudioEncoder", side_effect=error):
        result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 1
    assert "could not be loaded" in result.stdout
    # The guidance must name the actual remedy, not just the symptom.
    assert "shared" in result.stdout.lower()


def test_doctor_surfaces_unrelated_failures_distinctly():
    with patch("torchcodec.encoders.AudioEncoder", side_effect=RuntimeError("disk on fire")):
        result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 1
    assert "Audio decoding check failed" in result.stdout
    assert "disk on fire" in result.stdout
