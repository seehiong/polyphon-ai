"""Unit tests for batch speaker assignment and VoiceDB registration."""

import json
from pathlib import Path
from unittest.mock import MagicMock

from typer.testing import CliRunner

from polyphon.assign import assign_speakers_to_manifest, parse_speaker_map
from polyphon.cli import app
from polyphon.engine import PolyphonEngine
from polyphon.types import SpeakerInterval, WordToken


def test_parse_speaker_map():
    """Verify parse_speaker_map handles dicts, comma strings, and list formats."""
    # 1. Dictionary with numeric shorthand
    d_input = {"0": "Marcus Vance", "SPEAKER_01": "David (UI Designer)"}
    parsed = parse_speaker_map(d_input)
    assert parsed == {
        "SPEAKER_00": "Marcus Vance",
        "SPEAKER_01": "David (UI Designer)",
    }

    # 2. Comma-separated string with spaces and quotes
    s_input = " 0 = 'Marcus Vance', 1=\"David Chen\" , SPEAKER_02=Sarah "
    parsed = parse_speaker_map(s_input)
    assert parsed == {
        "SPEAKER_00": "Marcus Vance",
        "SPEAKER_01": "David Chen",
        "SPEAKER_02": "Sarah",
    }

    # 3. List of strings
    l_input = ["SPEAKER_00=Marcus Vance", "SPEAKER_01=David (UI Designer)"]
    parsed = parse_speaker_map(l_input)
    assert parsed == {
        "SPEAKER_00": "Marcus Vance",
        "SPEAKER_01": "David (UI Designer)",
    }

    # 4. Empty/None
    assert parse_speaker_map(None) == {}
    assert parse_speaker_map("") == {}


def test_assign_speakers_to_manifest(tmp_path: Path):
    """Verify assign_speakers_to_manifest updates JSON manifest, stats, and reports."""
    mock_manifest = {
        "duration": 60.0,
        "language": "en",
        "speakers": [
            {"id": "SPEAKER_00", "name": "Speaker 00", "confidence": 1.0},
            {"id": "SPEAKER_01", "name": "Speaker 01", "confidence": 1.0},
        ],
        "segments": [
            {
                "start": 0.0,
                "end": 5.0,
                "speaker_id": "SPEAKER_00",
                "speaker_name": "Speaker 00",
                "text": "Hello, welcome everyone.",
                "words": [
                    {"word": "Hello,", "start": 0.0, "end": 1.0, "score": 0.9},
                    {"word": "welcome", "start": 1.0, "end": 2.0, "score": 0.9},
                    {"word": "everyone.", "start": 2.0, "end": 3.0, "score": 0.9},
                ],
            },
            {
                "start": 5.5,
                "end": 10.0,
                "speaker_id": "SPEAKER_01",
                "speaker_name": "Speaker 01",
                "text": "Thanks, glad to be here.",
                "words": [
                    {"word": "Thanks,", "start": 5.5, "end": 6.5, "score": 0.9},
                    {"word": "glad", "start": 6.5, "end": 7.5, "score": 0.9},
                    {"word": "to", "start": 7.5, "end": 8.0, "score": 0.9},
                    {"word": "be", "start": 8.0, "end": 8.5, "score": 0.9},
                    {"word": "here.", "start": 8.5, "end": 9.5, "score": 0.9},
                ],
            },
        ],
        "insights": {
            "summary": "Team kickoff meeting.",
            "speaker_stats": {
                "Speaker 00": {"talk_time_seconds": 5.0, "percentage": 50.0, "segment_count": 1},
                "Speaker 01": {"talk_time_seconds": 4.5, "percentage": 50.0, "segment_count": 1},
            },
        },
    }

    meeting_file = tmp_path / "kickoff.json"
    meeting_file.write_text(json.dumps(mock_manifest, indent=2), encoding="utf-8")

    updated_data, enrolled = assign_speakers_to_manifest(
        meeting_path_or_id=meeting_file,
        speaker_map={"0": "Marcus Vance", "SPEAKER_01": "David (UI Designer)"},
        enroll=False,
        save=True,
    )

    assert enrolled == []
    speakers_dict = {s["id"]: s["name"] for s in updated_data["speakers"]}
    assert speakers_dict["SPEAKER_00"] == "Marcus Vance"
    assert speakers_dict["SPEAKER_01"] == "David (UI Designer)"

    assert updated_data["segments"][0]["speaker_name"] == "Marcus Vance"
    assert updated_data["segments"][1]["speaker_name"] == "David (UI Designer)"

    stats = updated_data["insights"]["speaker_stats"]
    assert "Marcus Vance" in stats
    assert "David (UI Designer)" in stats
    assert "Speaker 00" not in stats

    # Verify generated report files
    assert (tmp_path / "kickoff.html").exists()
    assert (tmp_path / "kickoff.md").exists()
    assert (tmp_path / "kickoff.srt").exists()


def test_polyphon_engine_with_speaker_map():
    """Verify PolyphonEngine.process respects explicit speaker_map."""
    mock_asr = MagicMock()
    mock_asr.transcribe.return_value = (
        "en",
        [
            WordToken(word="Hello", start=0.0, end=1.0, score=0.95),
            WordToken(word="world", start=1.0, end=2.0, score=0.95),
            WordToken(word="Response", start=3.0, end=4.0, score=0.95),
        ],
    )

    mock_diarizer = MagicMock()
    mock_diarizer.diarize.return_value = [
        SpeakerInterval(speaker_id="SPEAKER_00", start=0.0, end=2.5),
        SpeakerInterval(speaker_id="SPEAKER_01", start=2.5, end=4.5),
    ]

    from unittest.mock import patch

    engine = PolyphonEngine(asr=mock_asr, diarizer=mock_diarizer)
    with patch("polyphon.engine.get_audio_duration", return_value=5.0):
        result = engine.process(
            audio_path="dummy.wav",
            speaker_map={"0": "Marcus Vance", "SPEAKER_01": "David Chen"},
            infer_names=False,
            summarize=False,
            verbose=False,
        )

    spk_map = {s.id: s.name for s in result.speakers}
    assert spk_map["SPEAKER_00"] == "Marcus Vance"
    assert spk_map["SPEAKER_01"] == "David Chen"

    assert result.segments[0].speaker_name == "Marcus Vance"
    assert result.segments[1].speaker_name == "David Chen"


def test_cli_assign_command(tmp_path: Path):
    """Verify polyphon assign CLI command."""
    mock_manifest = {
        "duration": 30.0,
        "language": "en",
        "speakers": [{"id": "SPEAKER_00", "name": "Speaker 00", "confidence": 1.0}],
        "segments": [
            {
                "start": 0.0,
                "end": 2.0,
                "speaker_id": "SPEAKER_00",
                "speaker_name": "Speaker 00",
                "text": "Checking CLI assignment.",
                "words": [{"word": "Checking", "start": 0.0, "end": 1.0, "score": 0.9}],
            }
        ],
    }
    meeting_file = tmp_path / "test_cli_meeting.json"
    meeting_file.write_text(json.dumps(mock_manifest), encoding="utf-8")

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "assign",
            str(meeting_file),
            "--map",
            "SPEAKER_00=Alice Smith",
            "--no-enroll",
        ],
    )
    assert res.exit_code == 0
    assert "Alice Smith" in res.stdout

    saved = json.loads(meeting_file.read_text(encoding="utf-8"))
    assert saved["speakers"][0]["name"] == "Alice Smith"
    assert saved["segments"][0]["speaker_name"] == "Alice Smith"


def test_api_assign_speakers_endpoint(tmp_path: Path):
    """Verify /api/meetings/assign_speakers server route."""
    from fastapi.testclient import TestClient

    from polyphon.server.app import create_app

    mock_manifest = {
        "duration": 20.0,
        "language": "en",
        "speakers": [{"id": "SPEAKER_00", "name": "Speaker 00", "confidence": 1.0}],
        "segments": [
            {
                "start": 0.0,
                "end": 2.0,
                "speaker_id": "SPEAKER_00",
                "speaker_name": "Speaker 00",
                "text": "Testing API endpoint.",
                "words": [],
            }
        ],
    }
    meeting_file = tmp_path / "test_api.json"
    meeting_file.write_text(json.dumps(mock_manifest), encoding="utf-8")

    server_app = create_app(output_dir=str(tmp_path))
    client = TestClient(server_app)

    resp = client.post(
        "/api/meetings/assign_speakers",
        json={
            "meeting_id": "test_api",
            "speaker_map": {"SPEAKER_00": "Bob Jones"},
            "enroll": False,
            "summarize": False,
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["meeting"]["speakers"][0]["name"] == "Bob Jones"
    assert data["meeting"]["segments"][0]["speaker_name"] == "Bob Jones"


def test_cli_assign_missing_manifest_explains_json_requirement(tmp_path):
    """A missing manifest is usually a forgotten --format json, so say so."""
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["assign", str(tmp_path / "absent.json"), "--map", "SPEAKER_00=Alice"],
    )

    assert result.exit_code == 1
    assert "Meeting manifest not found" in result.stdout
    assert "--format json" in result.stdout


def test_assign_summarize_writes_insights(tmp_path):
    """--summarize must persist insights: Insights is a dataclass, not pydantic."""
    from unittest.mock import MagicMock

    from polyphon.assign import assign_speakers_to_manifest
    from polyphon.types import Insights

    manifest = tmp_path / "m.json"
    manifest.write_text(
        json.dumps(
            {
                "duration": 10.0,
                "language": "en",
                "speakers": [{"id": "SPEAKER_00", "name": "Speaker 00"}],
                "segments": [
                    {
                        "start": 0.0,
                        "end": 5.0,
                        "speaker_id": "SPEAKER_00",
                        "speaker_name": "Speaker 00",
                        "text": "Hello there.",
                        "words": [],
                    }
                ],
                "insights": {},
            }
        ),
        encoding="utf-8",
    )

    fake_llm = MagicMock()
    fake_llm.extract_insights.return_value = Insights(summary="A summary.", topics=["T1"])

    data, _ = assign_speakers_to_manifest(
        meeting_path_or_id=manifest,
        speaker_map={"SPEAKER_00": "Alice"},
        enroll=False,
        summarize=True,
        llm=fake_llm,
        save=False,
    )

    # extract_insights takes a rendered markdown transcript, not raw segments.
    (called_with,) = fake_llm.extract_insights.call_args[0]
    assert isinstance(called_with, str)
    assert "Hello there." in called_with

    assert data["insights"]["summary"] == "A summary."
    assert data["insights"]["topics"] == ["T1"]


def test_cli_rename_command(tmp_path):
    """Test polyphon rename CLI command updates JSON manifest and HTML/MD reports."""
    from polyphon.cli import app

    runner = CliRunner()
    manifest = tmp_path / "meeting_sample.json"
    html_file = tmp_path / "meeting_sample.html"
    md_file = tmp_path / "meeting_sample.md"
    html_file.write_text("<html><title>Old</title></html>", encoding="utf-8")
    md_file.write_text("# Old", encoding="utf-8")

    manifest.write_text(
        json.dumps(
            {
                "duration": 20.0,
                "language": "en",
                "speakers": [{"id": "SPEAKER_00", "name": "Alice"}],
                "segments": [
                    {
                        "start": 0.0,
                        "end": 2.0,
                        "speaker_id": "SPEAKER_00",
                        "speaker_name": "Alice",
                        "text": "Meeting conversation.",
                        "words": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    res = runner.invoke(app, ["rename", str(manifest), "Project Alpha Review", "--output-dir", str(tmp_path)])
    assert res.exit_code == 0
    assert "Successfully renamed meeting" in res.stdout
    assert "Project Alpha Review" in res.stdout

    updated = json.loads(manifest.read_text(encoding="utf-8"))
    assert updated["title"] == "Project Alpha Review"
    assert "Project Alpha Review" in md_file.read_text(encoding="utf-8")
    assert "Project Alpha Review" in html_file.read_text(encoding="utf-8")


def test_assign_speakers_merges_clusters_into_single_speaker(tmp_path):
    """When multiple clusters (e.g. SPEAKER_00 and SPEAKER_01) map to the same name, stats combine to 1 speaker."""
    from polyphon.assign import assign_speakers_to_manifest

    manifest = tmp_path / "two_clusters.json"
    manifest.write_text(
        json.dumps(
            {
                "duration": 10.0,
                "language": "en",
                "speakers": [
                    {"id": "SPEAKER_00", "name": "SPEAKER_00"},
                    {"id": "SPEAKER_01", "name": "SPEAKER_01"},
                ],
                "segments": [
                    {"start": 0.0, "end": 6.0, "speaker_id": "SPEAKER_00", "speaker_name": "SPEAKER_00", "text": "Hi"},
                    {
                        "start": 6.0,
                        "end": 10.0,
                        "speaker_id": "SPEAKER_01",
                        "speaker_name": "SPEAKER_01",
                        "text": "Hello",
                    },
                ],
                "insights": {
                    "speaker_stats": {
                        "SPEAKER_00": {"talk_time_seconds": 6.0, "percentage": 60.0, "segment_count": 1},
                        "SPEAKER_01": {"talk_time_seconds": 4.0, "percentage": 40.0, "segment_count": 1},
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    data, _ = assign_speakers_to_manifest(
        str(manifest),
        speaker_map={"SPEAKER_00": "Solo Speaker", "SPEAKER_01": "Solo Speaker"},
        enroll=False,
        summarize=False,
    )

    stats = data["insights"]["speaker_stats"]
    assert len(stats) == 1
    assert "Solo Speaker" in stats
    assert stats["Solo Speaker"]["talk_time_seconds"] == 10.0
    assert stats["Solo Speaker"]["percentage"] == 100.0
    assert stats["Solo Speaker"]["segment_count"] == 1
    assert len(data["segments"]) == 1
    assert data["segments"][0]["text"] == "Hi Hello"
    assert len(data["speakers"]) == 1
    assert data["speakers"][0]["name"] == "Solo Speaker"
