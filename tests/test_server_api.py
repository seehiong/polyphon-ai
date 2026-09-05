import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from polyphon.server.app import create_app


def test_server_routes():
    with tempfile.TemporaryDirectory() as tmpdir:
        app = create_app(output_dir=tmpdir)
        client = TestClient(app)

        # 1. Dashboard root HTML
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Polyphon Studio" in resp.text

        # 2. System status endpoint
        resp = client.get("/api/system")
        assert resp.status_code == 200
        data = resp.json()
        assert "version" in data
        assert "llm" in data
        assert "voicedb" in data

        # 3. Meetings archive list
        resp = client.get("/api/meetings")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

        # 4. Speakers list
        resp = client.get("/api/speakers")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

        # 5. File upload
        dummy_content = b"RIFF....WAVEfmt ...."
        with patch("polyphon.server.app.get_audio_duration", return_value=12.5):
            upload_resp = client.post(
                "/api/upload",
                files={"file": ("test_meeting.wav", io.BytesIO(dummy_content), "audio/wav")},
            )
            assert upload_resp.status_code == 200
            up_data = upload_resp.json()
            assert "saved_filename" in up_data
            assert up_data["duration"] == 12.5
            saved_name = up_data["saved_filename"]

            # 6. Process initiation
            proc_resp = client.post(
                "/api/process",
                data={
                    "saved_filename": saved_name,
                    "diarize": "true",
                    "identify": "false",
                    "infer_names": "true",
                    "auto_enroll": "false",
                    "summarize": "false",
                },
            )
            assert proc_resp.status_code == 200
            proc_data = proc_resp.json()
            assert "job_id" in proc_data
            assert proc_data["status"] == "pending"

            # 7. Media streaming endpoint
            media_resp = client.get(f"/media/{saved_name}")
            assert media_resp.status_code == 200
            assert media_resp.content == dummy_content

            # 8. Static assets endpoint
            logo_resp = client.get("/static/logo.jpg")
            assert logo_resp.status_code == 200
            assert "image" in logo_resp.headers.get("content-type", "")

            # 9. Archived meeting details and deletion
            import json
            from pathlib import Path

            meeting_stem = "test_meeting_archive"
            dummy_json_path = Path(tmpdir) / f"{meeting_stem}.json"
            dummy_json_path.write_text(
                json.dumps(
                    {
                        "speakers": [{"id": "spk_0", "name": "Alice"}],
                        "segments": [],
                        "duration": 10.0,
                    }
                )
            )
            dummy_html_path = Path(tmpdir) / f"{meeting_stem}.html"
            dummy_html_path.write_text("<html>Report</html>")

            get_meeting_resp = client.get(f"/api/meetings/{meeting_stem}")
            assert get_meeting_resp.status_code == 200
            m_data = get_meeting_resp.json()
            assert m_data["id"] == meeting_stem

            del_meeting_resp = client.delete(f"/api/meetings/{meeting_stem}")
            assert del_meeting_resp.status_code == 200
            assert del_meeting_resp.json()["status"] == "deleted"
            assert not dummy_json_path.exists()
            assert not dummy_html_path.exists()

            # 10. Dynamic on-demand report generation for MD and SRT
            meeting_dyn = "dyn_meeting"
            dyn_json = Path(tmpdir) / f"{meeting_dyn}.json"
            dyn_json.write_text(
                json.dumps(
                    {
                        "duration": 45.0,
                        "speakers": [{"id": "spk_1", "name": "Bob"}],
                        "segments": [
                            {
                                "start": 0.0,
                                "end": 2.5,
                                "speaker_id": "spk_1",
                                "speaker_name": "Bob",
                                "text": "Hello world.",
                                "words": [],
                            }
                        ],
                    }
                )
            )

            # Test on-demand MD generation
            md_resp = client.get(f"/reports/{meeting_dyn}.md")
            assert md_resp.status_code == 200
            assert "Hello world." in md_resp.text
            assert "Bob" in md_resp.text

            # Test on-demand SRT generation
            srt_resp = client.get(f"/reports/{meeting_dyn}.srt")
            assert srt_resp.status_code == 200
            assert "00:00:00,000 --> 00:00:02,500" in srt_resp.text
            assert "[Bob] Hello world." in srt_resp.text

            # 11. Media streaming range headers
            dummy_media = Path(tmpdir) / "test_stream.mp4"
            dummy_media.write_bytes(b"\x00" * 2048)
            media_resp = client.get("/media/test_stream.mp4", headers={"Accept-Encoding": "gzip"})
            assert media_resp.status_code == 200
            assert media_resp.headers.get("accept-ranges") == "bytes"

            # 12. GZip compression on responses > 1000 bytes
            big_report_resp = client.get(f"/reports/{meeting_dyn}.json", headers={"Accept-Encoding": "gzip"})
            assert big_report_resp.status_code == 200

            # 13. Live Session Archival, Comparison & Reprocessing
            live_stem = "live_stream_sample"
            live_wav = Path(tmpdir) / f"{live_stem}.wav"
            live_wav.write_bytes(dummy_content)
            live_json = Path(tmpdir) / f"{live_stem}.json"
            live_json.write_text(
                json.dumps(
                    {
                        "duration": 20.0,
                        "speakers": [{"id": "spk_live", "name": "Live Speaker"}],
                        "segments": [
                            {
                                "start": 0.0,
                                "end": 3.0,
                                "speaker_id": "spk_live",
                                "speaker_name": "Live Speaker",
                                "text": "Realtime speech chunk",
                                "words": [],
                            }
                        ],
                    }
                )
            )

            # Check /api/meetings detects is_live_only = True and populates created_at
            meetings_resp = client.get("/api/meetings")
            assert meetings_resp.status_code == 200
            m_list = {m["id"]: m for m in meetings_resp.json()}
            assert live_stem in m_list
            assert m_list[live_stem]["is_live_only"] is True
            assert m_list[live_stem]["has_comparison"] is False
            assert "created_at" in m_list[live_stem]
            assert isinstance(m_list[live_stem]["created_at"], (int, float))
            assert m_list[live_stem]["created_at"] > 0

            # Reprocess the live stream session using its stem or .wav
            reproc_resp = client.post(
                "/api/process",
                data={
                    "saved_filename": f"{live_stem}.wav",
                    "diarize": "false",
                    "identify": "false",
                    "infer_names": "false",
                    "auto_enroll": "false",
                    "summarize": "false",
                },
            )
            assert reproc_resp.status_code == 200
            assert reproc_resp.json()["status"] == "pending"

            # Create side-by-side comparison data
            live_archive_json = Path(tmpdir) / f"{live_stem}_live.json"
            live_archive_json.write_text(
                json.dumps(
                    {
                        "segments": [
                            {
                                "start": 0.0,
                                "end": 3.0,
                                "speaker_id": "spk_live",
                                "speaker_name": "Live Speaker",
                                "text": "Realtime speech chunk",
                            }
                        ]
                    }
                )
            )

            # Details endpoint should now indicate has_comparison = True and attach live_segments
            details_resp = client.get(f"/api/meetings/{live_stem}")
            assert details_resp.status_code == 200
            d_data = details_resp.json()
            assert d_data["has_comparison"] is True
            assert len(d_data["live_segments"]) == 1
            assert d_data["live_segments"][0]["text"] == "Realtime speech chunk"

            # Meetings list should exclude _live.json and show has_comparison = True
            meetings_resp2 = client.get("/api/meetings")
            m_list2 = {m["id"]: m for m in meetings_resp2.json()}
            assert f"{live_stem}_live" not in m_list2
            assert m_list2[live_stem]["has_comparison"] is True

            # Deletion should clean up live_archive_json and wav
            del_live_resp = client.delete(f"/api/meetings/{live_stem}")
            assert del_live_resp.status_code == 200
            assert not live_json.exists()
            assert not live_archive_json.exists()


def test_enroll_speaker_from_meeting():
    with tempfile.TemporaryDirectory() as tmpdir:
        app = create_app(output_dir=tmpdir)
        client = TestClient(app)

        # Create dummy meeting wav and json
        m_stem = "meeting_test_enroll"
        wav_file = Path(tmpdir) / f"{m_stem}.wav"
        wav_file.write_bytes(b"RIFF....WAVEfmt ....")

        json_file = Path(tmpdir) / f"{m_stem}.json"
        json_file.write_text(
            json.dumps(
                {
                    "duration": 5.0,
                    "language": "en",
                    "speakers": [{"id": "SPEAKER_00", "name": "SPEAKER_00"}],
                    "segments": [
                        {
                            "start": 0.5,
                            "end": 3.5,
                            "speaker_id": "SPEAKER_00",
                            "speaker_name": "SPEAKER_00",
                            "text": "Hello world from test speaker.",
                            "words": [],
                        }
                    ],
                    "insights": {
                        "speaker_stats": {
                            "SPEAKER_00": {
                                "talk_time_seconds": 3.0,
                                "percentage": 100.0,
                            }
                        }
                    },
                }
            )
        )

        fake_emb = np.ones(128, dtype=np.float32)
        fake_norm = fake_emb / np.linalg.norm(fake_emb)

        with (
            patch("polyphon.diarization.base.PyannoteDiarizer.extract_single_embedding", return_value=fake_norm),
            patch("subprocess.run"),
        ):
            resp = client.post(
                "/api/speakers/enroll_from_meeting",
                json={
                    "meeting_id": m_stem,
                    "speaker_id": "SPEAKER_00",
                    "name": "Alice Smith",
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "enrolled"
            assert data["name"] == "Alice Smith"
            assert data["id"] == "alice_smith"

            # Check manifest updated
            updated_manifest = json.loads(json_file.read_text(encoding="utf-8"))
            assert updated_manifest["speakers"][0]["name"] == "Alice Smith"
            assert updated_manifest["segments"][0]["speaker_name"] == "Alice Smith"
            assert "Alice Smith" in updated_manifest["insights"]["speaker_stats"]

            # Check VoiceDB lists the enrolled speaker
            vdb_resp = client.get("/api/speakers")
            assert vdb_resp.status_code == 200
            names = [s["name"] for s in vdb_resp.json()]
            assert "Alice Smith" in names

            # Test enrolling by job UUID
            job_uuid = "7c176d64b9c3438caed9b7877c5b7b94"
            from polyphon.server.app import JobState, jobs

            j_state = JobState(
                job_id=job_uuid,
                filename=f"{m_stem}.wav",
                filepath=wav_file,
            )
            j_state.status = "completed"
            jobs[job_uuid] = j_state
            resp_job = client.post(
                "/api/speakers/enroll_from_meeting",
                json={
                    "meeting_id": job_uuid,
                    "speaker_id": "SPEAKER_00",
                    "name": "Bob Jones",
                },
            )
            assert resp_job.status_code == 200
            assert resp_job.json()["status"] == "enrolled"
            assert resp_job.json()["name"] == "Bob Jones"
            jobs.pop(job_uuid, None)


def test_rename_meeting_endpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        app = create_app(output_dir=tmp_path)
        client = TestClient(app)

        m_stem = "live_1234567890"
        json_file = tmp_path / f"{m_stem}.json"
        html_file = tmp_path / f"{m_stem}.html"
        html_file.write_text("<html><title>Old Title</title></html>", encoding="utf-8")
        json_file.write_text(
            json.dumps(
                {
                    "duration": 45.0,
                    "language": "en",
                    "speakers": [{"id": "SPEAKER_00", "name": "Alice"}],
                    "segments": [
                        {
                            "start": 0.0,
                            "end": 3.0,
                            "speaker_id": "SPEAKER_00",
                            "speaker_name": "Alice",
                            "text": "Hello world.",
                            "words": [],
                        }
                    ],
                }
            )
        )

        # 1. Check default title in /api/meetings is unformatted stem
        meetings = client.get("/api/meetings").json()
        assert len(meetings) == 1
        assert meetings[0]["title"] == "live 1234567890"

        # 2. Rename via POST /api/meetings/{id}/rename
        rename_resp = client.post(
            f"/api/meetings/{m_stem}/rename",
            json={"title": "Q3 Engineering Sync"},
        )
        assert rename_resp.status_code == 200
        data = rename_resp.json()
        assert data["status"] == "ok"
        assert data["title"] == "Q3 Engineering Sync"

        # 3. Check /api/meetings reflects the new title
        updated_meetings = client.get("/api/meetings").json()
        assert updated_meetings[0]["title"] == "Q3 Engineering Sync"

        # 4. Check /api/meetings/{id} reflects the new title
        details = client.get(f"/api/meetings/{m_stem}").json()
        assert details["title"] == "Q3 Engineering Sync"

        # 5. Check manifest file on disk
        disk_data = json.loads(json_file.read_text(encoding="utf-8"))
        assert disk_data["title"] == "Q3 Engineering Sync"

        # 6. Check PATCH endpoint also works
        patch_resp = client.patch(
            f"/api/meetings/{m_stem}",
            json={"title": "Sprint Planning 2026"},
        )
        assert patch_resp.status_code == 200
        assert patch_resp.json()["title"] == "Sprint Planning 2026"
        assert client.get("/api/meetings").json()[0]["title"] == "Sprint Planning 2026"

        # 7. Validation: empty title returns 400
        bad_resp = client.post(f"/api/meetings/{m_stem}/rename", json={"title": "   "})
        assert bad_resp.status_code == 400

        # 8. Nonexistent meeting returns 404
        nf_resp = client.post("/api/meetings/nonexistent_123/rename", json={"title": "Test"})
        assert nf_resp.status_code == 404
