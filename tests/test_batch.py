import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from polyphon.batch import BatchProcessor
from polyphon.types import Insights, PolyphonResult, Segment, SpeakerInfo


def test_batch_processor_discovery_and_execution():
    with (
        tempfile.TemporaryDirectory() as tmp_in_dir,
        tempfile.TemporaryDirectory() as tmp_out_dir,
    ):
        # resolve() to match discover_files, which normalises its input. On
        # Windows, TEMP may be an 8.3 short path (C:\PROGRA~3\Temp) that expands
        # to a different string (C:\ProgramData\Temp) for the same directory,
        # so unresolved paths compare unequal despite pointing at one place.
        in_path = Path(tmp_in_dir).resolve()
        out_path = Path(tmp_out_dir).resolve()

        # Create dummy media files and dummy non-media files
        file1 = in_path / "meeting_a.mp4"
        file2 = in_path / "meeting_b.wav"
        file_txt = in_path / "notes.txt"

        file1.write_bytes(b"dummy mp4")
        file2.write_bytes(b"dummy wav")
        file_txt.write_text("not media")

        mock_engine = MagicMock()
        mock_res = PolyphonResult(
            duration=45.0,
            language="en",
            speakers=[SpeakerInfo(id="SPEAKER_00", name="Speaker 00")],
            segments=[
                Segment(
                    start=0.0,
                    end=10.0,
                    speaker_id="SPEAKER_00",
                    speaker_name="Speaker 00",
                    text="Hello batch.",
                )
            ],
            insights=Insights(
                summary="Batch meeting summary.",
                topics=["Batching"],
                decisions=["Proceed with batch"],
                action_items=[],
            ),
        )
        mock_engine.process.return_value = mock_res

        processor = BatchProcessor(engine=mock_engine)

        # 1. Test discovery
        discovered = processor.discover_files(in_path)
        assert len(discovered) == 2
        assert file1 in discovered
        assert file2 in discovered
        assert file_txt not in discovered

        # 2. Test execution
        batch_res = processor.process_directory(
            input_dir=in_path,
            output_dir=out_path,
            formats=["all"],
        )

        assert batch_res.total_files == 2
        assert batch_res.successful_files == 2
        assert batch_res.failed_files == 0
        assert mock_engine.process.call_count == 2

        # Check generated output files
        assert (out_path / "meeting_a.md").exists()
        assert (out_path / "meeting_a.json").exists()
        assert (out_path / "meeting_a.srt").exists()
        assert (out_path / "meeting_a.html").exists()

        assert (out_path / "meeting_b.md").exists()
        assert (out_path / "meeting_b.html").exists()

        # Check batch index manifest
        index_json = out_path / "batch_index.json"
        index_md = out_path / "batch_index.md"
        assert index_json.exists()
        assert index_md.exists()

        data = json.loads(index_json.read_text(encoding="utf-8"))
        assert data["total_files"] == 2
        assert data["successful_files"] == 2
        assert len(data["items"]) == 2

        md_content = index_md.read_text(encoding="utf-8")
        assert "# Polyphon Batch Intelligence Index" in md_content
        assert "meeting_a.mp4" in md_content
        assert "meeting_b.wav" in md_content
