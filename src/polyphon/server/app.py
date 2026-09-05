"""FastAPI application for Polyphon Studio."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import threading
import time
import urllib.request
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.gzip import GZipMiddleware

from polyphon.audio import get_audio_duration
from polyphon.diarization.base import PyannoteDiarizer, SortformerDiarizer
from polyphon.engine import PolyphonEngine
from polyphon.llm.base import OpenAICompatibleLLM
from polyphon.streaming import StreamingPolyphonEngine
from polyphon.types import PolyphonResult
from polyphon.voicedb.base import VoiceDB

logger = logging.getLogger("polyphon.server")

STATIC_DIR = Path(__file__).parent / "static"


class EnrollFromMeetingRequest(BaseModel):
    meeting_id: str
    speaker_id: str
    name: str


class AssignSpeakersRequest(BaseModel):
    meeting_id: str
    speaker_map: dict[str, str]
    enroll: bool = True
    summarize: bool = False


class JobState:
    def __init__(
        self,
        job_id: str,
        filename: str,
        filepath: Path,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        self.job_id = job_id
        self.filename = filename
        self.filepath = filepath
        self.status = "pending"  # pending, processing, completed, failed
        self.stage = 0
        self.stage_name = "Queued"
        self.percent = 0.0
        self.details = "Initializing pipeline..."
        self.result: dict[str, Any] | None = None
        self.error: str | None = None
        self.created_at = time.time()
        self.updated_at = time.time()
        self.event = asyncio.Event()
        self.loop = loop

    def update(
        self,
        status: str | None = None,
        stage: int | None = None,
        stage_name: str | None = None,
        percent: float | None = None,
        details: str | None = None,
    ):
        if status:
            self.status = status
        if stage is not None:
            self.stage = stage
        if stage_name:
            self.stage_name = stage_name
        if percent is not None:
            self.percent = percent
        if details is not None:
            self.details = details
        self.updated_at = time.time()

        if self.loop and not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self.event.set)
        else:
            self.event.set()

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "filename": self.filename,
            "status": self.status,
            "stage": self.stage,
            "stage_name": self.stage_name,
            "percent": self.percent,
            "details": self.details,
            "elapsed_seconds": round(time.time() - self.created_at, 1),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "has_result": self.result is not None,
            "error": self.error,
        }


jobs: dict[str, JobState] = {}


def create_app(output_dir: str | Path | None = None) -> FastAPI:
    """Create and configure the Polyphon Studio FastAPI server."""
    app = FastAPI(title="Polyphon Studio", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    resolved_out_dir = Path(output_dir or os.environ.get("POLYPHON_OUTPUT_DIR") or "outputs").expanduser().resolve()
    resolved_out_dir.mkdir(parents=True, exist_ok=True)

    uploads_dir = resolved_out_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    # Mount static assets
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.middleware("http")
    async def add_no_cache_for_static(request: Request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path.startswith("/static") or path == "/" or path.startswith("/api/meetings"):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    @app.get("/reports/{filename:path}")
    async def get_report_file(filename: str):
        """Serve or dynamically generate requested report format (HTML, MD, JSON, SRT)."""
        target = resolved_out_dir / filename
        if target.exists() and target.is_file():
            return FileResponse(target)

        # Check in project root directory or examples directory (e.g. meeting_challenging.md)
        for cand in [Path(filename), Path("examples") / filename]:
            if cand.exists() and cand.is_file():
                return FileResponse(cand)

        stem = Path(filename).stem
        ext = Path(filename).suffix.lower()

        # Check if corresponding .json exists in outputs or project directory
        json_candidates = [
            resolved_out_dir / f"{stem}.json",
            Path("examples") / f"{stem}.json",
            Path(f"{stem}.json"),
        ]
        if "_" in stem:
            base = stem.rsplit("_", 1)[0]
            json_candidates.append(resolved_out_dir / f"{base}.json")
            json_candidates.append(Path("examples") / f"{base}.json")
            json_candidates.append(Path(f"{base}.json"))

        json_file = next((f for f in json_candidates if f.exists() and f.is_file()), None)
        if json_file:
            try:
                with open(json_file, encoding="utf-8") as jf:
                    data = json.load(jf)
                result = PolyphonResult.from_dict(data)

                if ext == ".html":
                    html_content = result.to_html(title=f"Polyphon Report: {stem}")
                    out_path = resolved_out_dir / f"{stem}.html"
                    out_path.write_text(html_content, encoding="utf-8")
                    return FileResponse(out_path)
                elif ext == ".md":
                    md_content = result.to_markdown()
                    out_path = resolved_out_dir / f"{stem}.md"
                    out_path.write_text(md_content, encoding="utf-8")
                    return FileResponse(out_path)
                elif ext == ".srt":
                    srt_content = result.to_srt()
                    out_path = resolved_out_dir / f"{stem}.srt"
                    out_path.write_text(srt_content, encoding="utf-8")
                    return FileResponse(out_path)
                elif ext == ".json":
                    return FileResponse(json_file)
            except Exception as e:
                logger.warning(f"Failed to dynamically generate {filename}: {e}")

        raise HTTPException(status_code=404, detail=f"Report file {filename} not found")

    @app.api_route("/media/{filename:path}", methods=["GET", "HEAD"])
    async def get_media_stream(filename: str):
        """Stream media file with range request support."""
        media_headers = {"Accept-Ranges": "bytes"}
        # 1. Search in uploads_dir
        p = uploads_dir / filename
        if p.exists() and p.is_file():
            return FileResponse(p, headers=media_headers)
        # 2. Search in outputs_dir
        p = resolved_out_dir / filename
        if p.exists() and p.is_file():
            return FileResponse(p, headers=media_headers)
        # 3. Search in examples directory
        p = Path("examples") / filename
        if p.exists() and p.is_file():
            return FileResponse(p, headers=media_headers)
        # 4. Search in current working directory
        p = Path(filename)
        if p.exists() and p.is_file():
            return FileResponse(p, headers=media_headers)
        # 5. Search by stem match in uploads_dir
        stem = Path(filename).stem
        for match in uploads_dir.glob(f"{stem}.*"):
            if match.is_file():
                return FileResponse(match, headers=media_headers)
        # 6. Search by stem match in examples directory
        for match in Path("examples").glob(f"{stem}.*"):
            if match.is_file():
                return FileResponse(match, headers=media_headers)
        # 7. Search by stem match in project root
        for match in Path(".").glob(f"{stem}.*"):
            if match.is_file():
                return FileResponse(match, headers=media_headers)
        # 8. Fallback match if filename has a random hash suffix (e.g. meeting_challenging_f0d4d0a6)
        if "_" in stem:
            base = stem.rsplit("_", 1)[0]
            for folder in [Path("examples"), Path("."), uploads_dir]:
                for match in folder.glob(f"{base}.*"):
                    if match.is_file() and match.suffix.lower() in [
                        ".mp4",
                        ".wav",
                        ".mp3",
                        ".m4a",
                        ".webm",
                    ]:
                        return FileResponse(match, headers=media_headers)
                for match in folder.glob(f"{base}*"):
                    if match.is_file() and match.suffix.lower() in [
                        ".mp4",
                        ".wav",
                        ".mp3",
                        ".m4a",
                        ".webm",
                    ]:
                        return FileResponse(match, headers=media_headers)
        raise HTTPException(status_code=404, detail=f"Media file not found: {filename}")

    @app.get("/", response_class=HTMLResponse)
    async def get_dashboard():
        index_path = STATIC_DIR / "index.html"
        if not index_path.exists():
            return HTMLResponse("<h1>Polyphon Studio: static/index.html not found</h1>", status_code=404)
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))

    @app.get("/api/system")
    async def get_system_status():
        """Check system readiness, local LLM server connection, and VoiceDB stats."""
        vdb = VoiceDB()
        speakers = vdb.list_speakers()

        llm_endpoint = os.environ.get("POLYPHON_LLM_ENDPOINT", "http://127.0.0.1:8080/v1")
        llm_connected = False
        llm_model = os.environ.get("POLYPHON_LLM_MODEL", "qwen3.8-27b")

        try:
            req = urllib.request.Request(f"{llm_endpoint.rstrip('/')}/models")
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = data.get("data", []) or data.get("models", [])
                if models:
                    llm_connected = True
        except Exception:
            llm_connected = False

        return {
            "version": "0.1.0",
            "llm": {
                "endpoint": llm_endpoint,
                "model": llm_model,
                "connected": llm_connected,
            },
            "voicedb": {
                "enrolled_count": len(speakers),
                "speakers": speakers,
            },
            "output_dir": str(resolved_out_dir),
        }

    @app.post("/api/upload")
    async def upload_media(file: UploadFile = File(...)):
        """Upload audio or video file for processing."""
        file_ext = Path(file.filename or "audio.wav").suffix
        unique_name = f"{Path(file.filename or 'audio').stem}_{uuid.uuid4().hex[:8]}{file_ext}"
        target_path = uploads_dir / unique_name

        with target_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        duration = get_audio_duration(target_path)
        return {
            "filename": file.filename,
            "saved_filename": unique_name,
            "media_url": f"/media/{unique_name}",
            "duration": duration,
        }

    @app.post("/api/process")
    async def process_media(
        saved_filename: str = Form(...),
        diarize: bool = Form(True),
        identify: bool = Form(True),
        infer_names: bool = Form(True),
        auto_enroll: bool = Form(False),
        summarize: bool = Form(True),
        language: str = Form("en"),
        clean_fillers: bool = Form(False),
    ):
        """Initiate async background processing of an uploaded file or existing recording."""
        media_path = None
        for folder in [uploads_dir, resolved_out_dir, Path("examples"), Path(".")]:
            p = folder / saved_filename
            if p.exists() and p.is_file():
                media_path = p
                break

        if not media_path:
            stem = Path(saved_filename).stem
            for ext in [".wav", ".mp4", ".webm", ".mp3", ".m4a"]:
                for folder in [resolved_out_dir, uploads_dir, Path("examples"), Path(".")]:
                    p = folder / f"{stem}{ext}"
                    if p.exists() and p.is_file():
                        media_path = p
                        break
                if media_path:
                    break

        if not media_path or not media_path.exists():
            raise HTTPException(status_code=404, detail=f"Media file '{saved_filename}' not found")

        job_id = uuid.uuid4().hex
        loop = asyncio.get_running_loop()
        job = JobState(job_id=job_id, filename=media_path.name, filepath=media_path, loop=loop)
        jobs[job_id] = job

        # Run process in background task
        asyncio.create_task(
            _run_pipeline_job(
                job=job,
                output_dir=resolved_out_dir,
                diarize=diarize,
                identify=identify,
                infer_names=infer_names,
                auto_enroll=auto_enroll,
                summarize=summarize,
                language=language,
                clean_fillers=clean_fillers,
            )
        )

        return {"job_id": job_id, "status": "pending"}

    @app.get("/api/jobs/current")
    async def get_current_job():
        """Retrieve currently processing job or the most recently completed meeting."""
        # 1. Active processing job
        for j in reversed(list(jobs.values())):
            if j.status == "processing":
                d = j.to_dict()
                d["media_url"] = f"/media/{j.filename}"
                return d

        # 2. In-memory completed job
        for j in reversed(list(jobs.values())):
            if j.status == "completed":
                d = j.to_dict()
                d["result"] = j.result
                d["media_url"] = f"/media/{j.filename}"
                stem = Path(j.filename).stem
                d["html_report_url"] = f"/reports/{stem}.html"
                return d

        # 3. Latest archived meeting from outputs directory
        for file in sorted(resolved_out_dir.glob("*.json"), key=os.path.getmtime, reverse=True):
            if file.name.startswith("batch_index") or file.stem.endswith("_live"):
                continue
            try:
                data = json.loads(file.read_text(encoding="utf-8"))
                stem = file.stem
                live_file = resolved_out_dir / f"{stem}_live.json"
                if "live_segments" not in data and live_file.exists():
                    try:
                        live_data = json.loads(live_file.read_text(encoding="utf-8"))
                        data["live_segments"] = live_data.get("segments", [])
                    except Exception:
                        pass
                media_file = uploads_dir / f"{stem}.mp4"
                if not media_file.exists():
                    matches = list(uploads_dir.glob(f"{stem}.*"))
                    if matches:
                        media_file = matches[0]
                    else:
                        out_matches = list(resolved_out_dir.glob(f"{stem}.*"))
                        for om in out_matches:
                            if om.suffix in [".wav", ".mp4", ".webm", ".mp3", ".m4a", ".flac", ".mkv"]:
                                media_file = om
                                break
                return {
                    "has_job": True,
                    "job_id": stem,
                    "filename": (media_file.name if media_file.exists() else f"{stem}.mp4"),
                    "status": "completed",
                    "stage": 4,
                    "stage_name": "Complete",
                    "percent": 100.0,
                    "details": "Archived meeting ready",
                    "elapsed_seconds": 0,
                    "created_at": os.path.getmtime(file),
                    "media_url": (f"/media/{media_file.name}" if media_file.exists() else ""),
                    "html_report_url": (
                        f"/reports/{stem}.html" if (resolved_out_dir / f"{stem}.html").exists() else ""
                    ),
                    "result": data,
                }
            except Exception:
                continue

        return {"has_job": False, "status": "idle"}

    @app.get("/api/jobs/{job_id}")
    async def get_job_status(job_id: str):
        """Get instant JSON status snapshot of a job without streaming."""
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        j = jobs[job_id]
        d = j.to_dict()
        if j.status == "completed" and j.result:
            d["result"] = j.result
            d["media_url"] = f"/media/{j.filename}"
            stem = Path(j.filename).stem
            d["html_report_url"] = f"/reports/{stem}.html"
        return d

    @app.get("/api/jobs/{job_id}/events")
    async def stream_job_events(job_id: str):
        """Server-Sent Events (SSE) stream for real-time progress updates."""
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")

        job = jobs[job_id]

        async def event_generator():
            last_percent = -1.0
            last_stage = -1

            while True:
                # Yield update if state changed
                if job.percent != last_percent or job.stage != last_stage or job.status in ("completed", "failed"):
                    last_percent = job.percent
                    last_stage = job.stage

                    payload = {
                        "job_id": job.job_id,
                        "status": job.status,
                        "stage": job.stage,
                        "stage_name": job.stage_name,
                        "percent": job.percent,
                        "details": job.details,
                    }

                    if job.status == "completed" and job.result:
                        payload["result"] = job.result
                        payload["media_url"] = f"/media/{job.filename}"
                        stem = Path(job.filename).stem
                        payload["html_report_url"] = f"/reports/{stem}.html"

                    if job.status == "failed":
                        payload["error"] = job.error

                    yield f"data: {json.dumps(payload)}\n\n"

                    if job.status in ("completed", "failed"):
                        break

                # Wait for next state change or heartbeat
                try:
                    await asyncio.wait_for(job.event.wait(), timeout=0.5)
                    job.event.clear()
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/meetings")
    async def list_meetings():
        """List past processed meetings from output directory."""
        items = []
        for file in sorted(resolved_out_dir.glob("*.json"), key=os.path.getmtime, reverse=True):
            if file.name.startswith("batch_index") or file.stem.endswith("_live"):
                continue
            try:
                data = json.loads(file.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    continue

                duration = float(data.get("duration", 0.0))
                segments = data.get("segments", [])
                if duration <= 0.0 and not segments:
                    continue

                stem = file.stem
                insights = data.get("insights")
                is_live_only = not bool(insights)
                has_comparison = bool(data.get("live_segments")) or (resolved_out_dir / f"{stem}_live.json").exists()
                summary = insights.get("summary") if isinstance(insights, dict) else None
                if summary:
                    summary_snippet = summary[:180] + "..." if len(summary) > 180 else summary
                else:
                    if segments:
                        preview = " ".join(s.get("text", "") for s in segments[:3]).strip()
                        summary_snippet = (
                            (preview[:180] + "...") if len(preview) > 180 else (preview or "Live stream recording")
                        )
                    else:
                        summary_snippet = "Live stream recording"

                meeting_title = data.get("title") or stem.replace("_", " ")

                items.append(
                    {
                        "id": stem,
                        "title": meeting_title,
                        "duration": duration,
                        "language": data.get("language", "en"),
                        "speakers_count": len(data.get("speakers", [])),
                        "turns_count": len(segments),
                        "summary_snippet": summary_snippet,
                        "is_live_only": is_live_only,
                        "has_comparison": has_comparison,
                        "created_at": os.path.getmtime(file),
                        "json_url": f"/reports/{file.name}",
                        "html_url": (f"/reports/{stem}.html" if (resolved_out_dir / f"{stem}.html").exists() else None),
                        "md_url": (f"/reports/{stem}.md" if (resolved_out_dir / f"{stem}.md").exists() else None),
                        "srt_url": (f"/reports/{stem}.srt" if (resolved_out_dir / f"{stem}.srt").exists() else None),
                    }
                )
            except Exception as err:
                logger.warning("Error reading meeting manifest %s: %s", file, err)
                continue

        return items

    @app.get("/api/meetings/{meeting_id}")
    async def get_meeting_details(meeting_id: str):
        """Retrieve full details and transcript segments for a meeting."""
        json_file = resolved_out_dir / f"{meeting_id}.json"
        if not json_file.exists():
            matches = list(resolved_out_dir.glob(f"{meeting_id}*.json"))
            if matches:
                json_file = matches[0]
            elif "_" in meeting_id:
                base = meeting_id.rsplit("_", 1)[0]
                base_matches = list(resolved_out_dir.glob(f"{base}*.json"))
                if base_matches:
                    json_file = base_matches[0]
        if not json_file or not json_file.exists():
            raise HTTPException(status_code=404, detail="Meeting not found")
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("segments"):
                res = PolyphonResult.from_dict(data)
                data["segments"] = [asdict(s) for s in res.segments]
            stem = json_file.stem

            live_file = resolved_out_dir / f"{stem}_live.json"
            if "live_segments" not in data and live_file.exists():
                try:
                    live_data = json.loads(live_file.read_text(encoding="utf-8"))
                    data["live_segments"] = live_data.get("segments", [])
                except Exception:
                    pass

            data["has_comparison"] = bool(data.get("live_segments")) and bool(data.get("segments"))
            data["is_live_only"] = not bool(data.get("insights"))

            media_url = None
            # Search for exact stem match across supported media extensions
            for ext in [".mp4", ".wav", ".webm", ".mp3", ".m4a", ".flac", ".mkv"]:
                for folder in [resolved_out_dir, uploads_dir, Path("examples"), Path(".")]:
                    candidate = folder / f"{stem}{ext}"
                    if candidate.exists() and candidate.is_file():
                        media_url = f"/media/{stem}{ext}"
                        break
                if media_url:
                    break

            # Fallback for hash-suffixed files (e.g. meeting_challenging_3min_abc)
            if not media_url and "_" in stem:
                base = stem.rsplit("_", 1)[0]
                for ext in [".mp4", ".wav", ".webm", ".mp3", ".m4a"]:
                    for folder in [Path("examples"), uploads_dir, Path("."), resolved_out_dir]:
                        candidate = folder / f"{base}{ext}"
                        if candidate.exists() and candidate.is_file():
                            media_url = f"/media/{base}{ext}"
                            break
                    if media_url:
                        break

            data["media_url"] = media_url or (
                f"/media/{stem}.wav" if (resolved_out_dir / f"{stem}.wav").exists() else f"/media/{stem}.mp4"
            )
            data["id"] = stem
            data["title"] = data.get("title") or stem.replace("_", " ")
            data["html_report_url"] = f"/reports/{stem}.html" if (resolved_out_dir / f"{stem}.html").exists() else None
            return data
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/api/meetings/{meeting_id:path}/rename")
    @app.patch("/api/meetings/{meeting_id:path}")
    async def rename_meeting(meeting_id: str, payload: dict[str, Any]):
        """Rename the display title of an existing meeting."""
        from urllib.parse import unquote

        clean_id = unquote(meeting_id).strip()
        target_stems = [clean_id, Path(clean_id).stem, clean_id.removesuffix(".json")]

        # Resolve job UUID if applicable
        for jid, j in list(jobs.items()):
            j_stem = Path(j.filename).stem if getattr(j, "filename", None) else None
            j_file_stem = j.filepath.stem if getattr(j, "filepath", None) else None
            if clean_id in (jid, j.job_id, j.filename, j_stem, j_file_stem):
                if j_file_stem:
                    target_stems.insert(0, j_file_stem)
                break

        new_title = (payload.get("title") or "").strip()
        if not new_title:
            raise HTTPException(status_code=400, detail="Title cannot be empty")

        json_file = None
        matched_stem = None
        for stem in target_stems:
            cand = resolved_out_dir / f"{stem}.json"
            if cand.exists() and not cand.stem.endswith("_live"):
                json_file = cand
                matched_stem = stem
                break

        if not json_file:
            # Try case-insensitive matching
            for cand in resolved_out_dir.glob("*.json"):
                if cand.stem.endswith("_live") or cand.name.startswith("batch_index"):
                    continue
                if clean_id.lower() == cand.stem.lower():
                    json_file = cand
                    matched_stem = cand.stem
                    break

        if not json_file or not json_file.exists():
            raise HTTPException(status_code=404, detail="Meeting not found")

        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            data["title"] = new_title
            json_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

            # Update live json companion if present
            live_json_file = resolved_out_dir / f"{matched_stem}_live.json"
            if live_json_file.exists():
                try:
                    live_data = json.loads(live_json_file.read_text(encoding="utf-8"))
                    live_data["title"] = new_title
                    live_json_file.write_text(json.dumps(live_data, indent=2, ensure_ascii=False), encoding="utf-8")
                except Exception:
                    pass

            # Regenerate HTML and Markdown reports with the new title
            try:
                poly_res = PolyphonResult.from_dict(data)
                poly_res.title = new_title
                html_file = resolved_out_dir / f"{matched_stem}.html"
                if html_file.exists():
                    html_file.write_text(poly_res.to_html(title=new_title), encoding="utf-8")
                md_file = resolved_out_dir / f"{matched_stem}.md"
                if md_file.exists():
                    md_file.write_text(poly_res.to_markdown(title=new_title), encoding="utf-8")
            except Exception as e:
                logger.warning("Could not re-render HTML/MD reports on rename: %s", e)

            # Update in-memory job if active
            for _jid, j in list(jobs.items()):
                j_file_stem = j.filepath.stem if getattr(j, "filepath", None) else None
                if matched_stem in (j_file_stem, getattr(j, "filename", None)) and j.result:
                    j.result["title"] = new_title

            return {"status": "ok", "id": matched_stem, "title": new_title}
        except Exception as e:
            logger.error("Error renaming meeting %s: %s", matched_stem, e)
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.delete("/api/meetings/{meeting_id:path}")
    async def delete_meeting(meeting_id: str):
        """Delete an archived meeting, all associated reports, and any uploaded media."""
        from urllib.parse import unquote

        clean_id = unquote(meeting_id).strip()
        deleted_files = []

        target_stems = {clean_id, Path(clean_id).stem, clean_id.removesuffix(".json")}

        # If clean_id is a job UUID or filename, resolve to its actual file stem
        for jid, j in list(jobs.items()):
            j_stem = Path(j.filename).stem if getattr(j, "filename", None) else None
            j_file_stem = j.filepath.stem if getattr(j, "filepath", None) else None
            if jid == clean_id or j_stem in target_stems or j_file_stem in target_stems:
                if j_stem:
                    target_stems.add(j_stem)
                if j_file_stem:
                    target_stems.add(j_file_stem)
                jobs.pop(jid, None)

        for s in target_stems:
            if not s:
                continue
            # 1. Delete reports and metadata files in resolved_out_dir
            for ext in [".json", ".html", ".md", ".srt", ".wav", ".mp4", "_live.json"]:
                target = resolved_out_dir / f"{s}{ext}"
                if target.exists() and target.is_file() and target.name not in deleted_files:
                    try:
                        target.unlink()
                        deleted_files.append(target.name)
                    except Exception as e:
                        logger.warning(f"Could not delete {target}: {e}")

            # 2. Iterate directory to delete any remaining file matching the stem (safe against glob syntax)
            for dir_path in [resolved_out_dir, uploads_dir]:
                if not dir_path.exists():
                    continue
                for f in dir_path.iterdir():
                    if f.is_file() and (f.stem == s or f.stem.startswith(f"{s}_live")) and f.name not in deleted_files:
                        try:
                            f.unlink()
                            deleted_files.append(f.name)
                        except Exception as e:
                            logger.warning(f"Could not delete {f}: {e}")

        logger.info(f"Deleted meeting '{clean_id}', removed {len(deleted_files)} files: {deleted_files}")
        return {"status": "deleted", "id": clean_id, "deleted_files": deleted_files}

    @app.get("/api/speakers")
    async def list_speakers():
        """List all enrolled speakers in VoiceDB."""
        vdb = VoiceDB()
        return vdb.list_speakers()

    @app.post("/api/speakers/enroll")
    async def enroll_speaker(name: str = Form(...), file: UploadFile = File(...)):
        """Enroll a new voice profile from audio sample."""
        sample_path = uploads_dir / f"sample_{uuid.uuid4().hex[:6]}_{file.filename}"
        with sample_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        from polyphon.diarization.base import PyannoteDiarizer

        diarizer = PyannoteDiarizer()
        embedding = diarizer.extract_single_embedding(str(sample_path))

        vdb = VoiceDB()
        slug = vdb.enroll(name=name, embedding=embedding, audio_path=str(sample_path))
        return {"status": "enrolled", "id": slug, "name": name}

    @app.delete("/api/speakers/{speaker_id}")
    async def delete_speaker(speaker_id: str):
        """Delete an enrolled voice profile."""
        vdb = VoiceDB()
        success = vdb.delete_speaker(speaker_id)
        if not success:
            raise HTTPException(status_code=404, detail="Speaker profile not found")
        return {"status": "deleted", "id": speaker_id}

    @app.post("/api/speakers/enroll_from_meeting")
    async def enroll_speaker_from_meeting(req: EnrollFromMeetingRequest):
        """Enroll a speaker into VoiceDB by extracting embeddings from their segments in a meeting."""
        from urllib.parse import unquote

        import numpy as np

        meeting_id = req.meeting_id.strip()
        speaker_id = req.speaker_id.strip()
        new_name = req.name.strip()

        if not meeting_id or not speaker_id or not new_name:
            raise HTTPException(status_code=400, detail="meeting_id, speaker_id, and name must be provided")

        clean_id = unquote(meeting_id).strip()
        target_stems: list[str] = [clean_id, Path(clean_id).stem, clean_id.removesuffix(".json")]

        # 1. Resolve job UUID if applicable
        job_obj = jobs.get(clean_id)
        if not job_obj:
            for jid, j in list(jobs.items()):
                if clean_id in (jid, getattr(j, "job_id", None)):
                    job_obj = j
                    break

        if job_obj:
            j_stem = Path(job_obj.filename).stem if getattr(job_obj, "filename", None) else None
            j_file_stem = job_obj.filepath.stem if getattr(job_obj, "filepath", None) else None
            if j_file_stem:
                target_stems.insert(0, j_file_stem)
            if j_stem:
                target_stems.insert(0, j_stem)

        # 2. Locate meeting JSON manifest
        manifest_path = None
        matched_stem = None
        for stem_cand in target_stems:
            for cand in [
                resolved_out_dir / f"{stem_cand}.json",
                resolved_out_dir / f"{stem_cand}_live.json",
                resolved_out_dir / stem_cand,
            ]:
                if cand.exists() and cand.is_file():
                    manifest_path = cand
                    matched_stem = cand.stem.removesuffix("_live")
                    break
            if manifest_path:
                break

        if not manifest_path:
            # Case-insensitive fallback
            for cand in resolved_out_dir.glob("*.json"):
                if cand.stem.lower() in [s.lower() for s in target_stems]:
                    manifest_path = cand
                    matched_stem = cand.stem.removesuffix("_live")
                    break

        if not manifest_path:
            raise HTTPException(status_code=404, detail=f"Meeting manifest not found for {meeting_id}")

        try:
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read meeting manifest: {e}") from e

        # 3. Locate audio/media file
        audio_file = None
        if job_obj and getattr(job_obj, "filepath", None) and job_obj.filepath.exists():
            audio_file = job_obj.filepath

        search_dirs = [
            resolved_out_dir,
            uploads_dir,
            Path(__file__).parent.parent.parent.parent / "examples",
        ]
        search_exts = [".wav", ".mp4", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".mov"]

        if not audio_file:
            for key in ["audio_path", "media_path", "filename"]:
                val = manifest_data.get(key)
                if val:
                    p = Path(val)
                    if p.exists() and p.is_file():
                        audio_file = p
                        break
                    for s_dir in search_dirs:
                        p_cand = s_dir / p.name
                        if p_cand.exists() and p_cand.is_file():
                            audio_file = p_cand
                            break
                if audio_file:
                    break

        if not audio_file:
            for stem_cand in [matched_stem] + target_stems:
                if not stem_cand:
                    continue
                for s_dir in search_dirs:
                    for ext in search_exts:
                        p_cand = s_dir / f"{stem_cand}{ext}"
                        if p_cand.exists() and p_cand.is_file():
                            audio_file = p_cand
                            break
                    if audio_file:
                        break
                if audio_file:
                    break

        if not audio_file:
            raise HTTPException(
                status_code=404,
                detail=f"Could not locate media/audio file for meeting '{meeting_id}'",
            )

        # 3. Find segments spoken by this speaker
        segments = manifest_data.get("segments") or []
        live_segments = manifest_data.get("live_segments") or []
        target_segments = [
            s
            for s in (segments or live_segments)
            if s.get("speaker_id") == speaker_id or s.get("speaker_name") == speaker_id
        ]

        if not target_segments:
            raise HTTPException(
                status_code=400,
                detail=f"No dialogue segments found for speaker '{speaker_id}' in this meeting",
            )

        # Sort segments by duration descending
        target_segments.sort(key=lambda s: (s.get("end", 0.0) - s.get("start", 0.0)), reverse=True)

        # Take top longest segments (at least 0.5s duration)
        viable_segments = [s for s in target_segments if (s.get("end", 0.0) - s.get("start", 0.0)) >= 0.5]
        if not viable_segments:
            viable_segments = target_segments[:3]

        # 4. Extract embeddings
        from polyphon.diarization.base import PyannoteDiarizer

        diarizer = PyannoteDiarizer()

        spk_embs = []
        for seg in viable_segments[:5]:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
            if end > start:
                try:
                    emb = diarizer.extract_single_embedding(str(audio_file), start=start, end=end)
                    spk_embs.append(emb)
                except Exception as exc:
                    logger.warning(f"Could not extract embedding for segment [{start:.2f}-{end:.2f}]: {exc}")

        if not spk_embs:
            try:
                emb = diarizer.extract_single_embedding(str(audio_file))
                spk_embs.append(emb)
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Failed to extract speaker embedding: {exc}") from exc

        avg_emb = np.mean(spk_embs, axis=0)
        norm = np.linalg.norm(avg_emb)
        norm_emb = (avg_emb / norm) if norm > 0 else avg_emb

        # 5. Extract a short audio sample snippet for VoiceDB preview
        vdb_samples_dir = resolved_out_dir / "samples"
        vdb_samples_dir.mkdir(exist_ok=True)
        slug = new_name.lower().replace(" ", "_")
        sample_audio_path = vdb_samples_dir / f"sample_{slug}.wav"

        best_seg = viable_segments[0]
        s_start = max(0.0, float(best_seg.get("start", 0.0)))
        s_dur = min(5.0, max(1.0, float(best_seg.get("end", 0.0)) - s_start))
        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                f"{s_start:.2f}",
                "-t",
                f"{s_dur:.2f}",
                "-i",
                str(audio_file),
                "-ar",
                "16000",
                "-ac",
                "1",
                str(sample_audio_path),
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except Exception as exc:
            logger.warning(f"Could not slice sample audio via ffmpeg: {exc}")
            sample_audio_path = None

        # 6. Enroll into VoiceDB
        vdb = VoiceDB()
        enrolled_id = vdb.enroll(
            name=new_name,
            embedding=norm_emb,
            speaker_id=slug,
            audio_path=str(sample_audio_path) if sample_audio_path else None,
        )

        # 7. Propagate updated name across meeting manifest
        spk_found = False
        for spk in manifest_data.get("speakers", []):
            if spk.get("id") == speaker_id or spk.get("name") == speaker_id:
                spk["name"] = new_name
                spk["id"] = speaker_id
                spk["confidence"] = 1.0
                spk_found = True
        if not spk_found:
            manifest_data.setdefault("speakers", []).append({"id": speaker_id, "name": new_name, "confidence": 1.0})

        for seg in manifest_data.get("segments", []):
            if seg.get("speaker_id") == speaker_id or seg.get("speaker_name") == speaker_id:
                seg["speaker_name"] = new_name

        for seg in manifest_data.get("live_segments", []):
            if seg.get("speaker_id") == speaker_id or seg.get("speaker_name") == speaker_id:
                seg["speaker_name"] = new_name

        # Coalesce adjacent segments by the same speaker
        def _coalesce_segs(raw_segs: list[dict[str, Any]], max_gap: float = 2.5) -> list[dict[str, Any]]:
            if not raw_segs:
                return []
            res: list[dict[str, Any]] = []
            curr = dict(raw_segs[0])
            for nxt in raw_segs[1:]:
                c_name = curr.get("speaker_name") or curr.get("speaker_id")
                n_name = nxt.get("speaker_name") or nxt.get("speaker_id")
                c_end = float(curr.get("end", 0.0))
                n_start = float(nxt.get("start", 0.0))
                if c_name == n_name and (n_start - c_end) <= max_gap:
                    curr["end"] = max(c_end, float(nxt.get("end", 0.0)))
                    curr["text"] = f"{curr.get('text', '').strip()} {nxt.get('text', '').strip()}".strip()
                    c_w = curr.get("words")
                    n_w = nxt.get("words")
                    if c_w is not None and n_w is not None:
                        curr["words"] = list(c_w) + list(n_w)
                else:
                    res.append(curr)
                    curr = dict(nxt)
            res.append(curr)
            return res

        if "segments" in manifest_data and manifest_data["segments"]:
            manifest_data["segments"] = _coalesce_segs(manifest_data["segments"])
        if "live_segments" in manifest_data and manifest_data["live_segments"]:
            manifest_data["live_segments"] = _coalesce_segs(manifest_data["live_segments"])

        # Deduplicate speakers by name
        seen_spk_names = set()
        deduped_speakers = []
        for spk in manifest_data.get("speakers", []):
            sname = spk.get("name") or spk.get("id")
            if sname not in seen_spk_names:
                seen_spk_names.add(sname)
                deduped_speakers.append(spk)
        manifest_data["speakers"] = deduped_speakers

        # Recalculate speaker stats from updated segments so merged speakers combine accurately
        from dataclasses import asdict

        from polyphon.reconciliation.base import ReconciliationEngine
        from polyphon.types import Segment

        all_segs = [
            Segment(
                start=float(s.get("start", 0.0)),
                end=float(s.get("end", 0.0)),
                speaker_id=str(s.get("speaker_id", "")),
                speaker_name=str(s.get("speaker_name") or s.get("speaker_id", "")),
                text=str(s.get("text", "")),
            )
            for s in (manifest_data.get("segments") or manifest_data.get("live_segments") or [])
        ]
        if all_segs:
            dur = float(manifest_data.get("duration", 0.0))
            if dur <= 0:
                dur = max(s.end for s in all_segs)
            recomputed = ReconciliationEngine.compute_speaker_stats(all_segs, dur)
            manifest_data.setdefault("insights", {})["speaker_stats"] = {k: asdict(v) for k, v in recomputed.items()}

        manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

        # Update live companion file if present
        if matched_stem:
            live_file = resolved_out_dir / f"{matched_stem}_live.json"
            if live_file.exists():
                try:
                    live_file.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
                except Exception:
                    pass

        # Update in-memory job if active
        for _jid, j in list(jobs.items()):
            j_file_stem = j.filepath.stem if getattr(j, "filepath", None) else None
            if matched_stem in (j_file_stem, getattr(j, "filename", None), clean_id) and j.result:
                j.result = manifest_data

        try:
            poly_result = PolyphonResult.from_dict(manifest_data)
            out_stem = matched_stem or stem_cand or Path(clean_id).stem
            out_title = (
                manifest_data.get("title")
                or (poly_result.title if hasattr(poly_result, "title") else None)
                or f"Polyphon Report: {out_stem}"
            )
            html_file = resolved_out_dir / f"{out_stem}.html"
            md_file = resolved_out_dir / f"{out_stem}.md"
            srt_file = resolved_out_dir / f"{out_stem}.srt"

            html_file.write_text(poly_result.to_html(title=out_title), encoding="utf-8")
            md_file.write_text(poly_result.to_markdown(title=out_title), encoding="utf-8")
            srt_file.write_text(poly_result.to_srt(), encoding="utf-8")
        except Exception as exc:
            logger.warning(f"Could not regenerate HTML/MD reports after renaming: {exc}")

        return {
            "status": "enrolled",
            "id": enrolled_id,
            "name": new_name,
            "speaker_id": speaker_id,
            "samples_count": len(spk_embs),
            "meeting_id": matched_stem or meeting_id,
            "meeting": manifest_data,
        }

    @app.post("/api/meetings/assign_speakers")
    async def assign_speakers_endpoint(req: AssignSpeakersRequest):
        """Assign speaker names to an existing meeting and optionally enroll in VoiceDB."""
        from urllib.parse import unquote

        from polyphon.assign import assign_speakers_to_manifest

        meeting_id = req.meeting_id.strip()
        if not meeting_id or not req.speaker_map:
            raise HTTPException(status_code=400, detail="meeting_id and speaker_map must be provided")

        clean_id = unquote(meeting_id).strip()
        for jid, j in list(jobs.items()):
            if clean_id in (jid, getattr(j, "job_id", None)):
                if getattr(j, "filepath", None):
                    clean_id = j.filepath.stem
                elif getattr(j, "filename", None):
                    clean_id = Path(j.filename).stem
                break

        try:
            manifest_data, enrolled = assign_speakers_to_manifest(
                meeting_path_or_id=clean_id,
                speaker_map=req.speaker_map,
                enroll=req.enroll,
                summarize=req.summarize,
                output_dir=resolved_out_dir,
                save=True,
            )
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to assign speakers: {e}") from e

        return {
            "status": "success",
            "meeting_id": meeting_id,
            "enrolled_speakers": enrolled,
            "meeting": manifest_data,
        }

    @app.get("/api/stream/status")
    async def get_stream_status():
        """Get streaming capabilities, models, and available neural diarizers."""
        return {
            "streaming_supported": True,
            "diarizers": ["sortformer", "pyannote"],
            "default_diarizer": "sortformer",
            "models": ["base", "small", "tiny", "large-v3"],
            "default_model": "base",
            "default_language": "en",
            "sample_rate": 16000,
        }

    @app.websocket("/api/stream/ws")
    async def stream_audio_ws(websocket: WebSocket):
        """WebSocket endpoint for real-time bidirectional audio streaming & live transcription."""
        await websocket.accept()
        session_name = f"live_{int(time.time())}"
        diarizer = SortformerDiarizer()
        engine = StreamingPolyphonEngine(asr_model="base", language="en", diarizer=diarizer)
        engine.start_stream()

        is_running = True
        proc_task: asyncio.Task[None] | None = None

        async def step_worker():
            while is_running:
                await asyncio.sleep(1.0)
                if not is_running:
                    break
                if engine.buffer.get_total_duration() >= 1.0:
                    try:
                        event = await asyncio.to_thread(engine.process_step)
                        if event and is_running:
                            await websocket.send_json(event.to_dict())
                    except Exception as err:
                        logger.warning("Streaming step worker error: %s", err)

        proc_task = asyncio.create_task(step_worker())

        try:
            while True:
                message = await websocket.receive()
                if "bytes" in message and message["bytes"]:
                    engine.feed_audio(message["bytes"])
                elif "text" in message and message["text"]:
                    try:
                        payload = json.loads(message["text"])
                    except json.JSONDecodeError:
                        continue

                    action = payload.get("action", "")
                    if action == "start":
                        req_diarizer = payload.get("diarizer", "sortformer")
                        req_identify = payload.get("identify", False)
                        req_lang = payload.get("language", "en")
                        req_model = payload.get("model", "base")
                        if req_lang in ("auto", "", None):
                            req_lang = None
                        session_name = payload.get("name") or f"live_{int(time.time())}"
                        session_title = payload.get("title")
                        d_backend = SortformerDiarizer() if req_diarizer == "sortformer" else PyannoteDiarizer()
                        vdb = VoiceDB() if req_identify else None

                        # Cancel previous worker
                        if proc_task and not proc_task.done():
                            is_running = False
                            proc_task.cancel()

                        engine = StreamingPolyphonEngine(
                            asr_model=req_model,
                            language=req_lang,
                            diarizer=d_backend,
                            voicedb=vdb,
                            chunk_duration=1.2,
                            window_duration=5.0,
                        )
                        engine.start_stream()
                        is_running = True
                        proc_task = asyncio.create_task(step_worker())
                        await websocket.send_json({"status": "started", "session_name": session_name})

                    elif action == "stop":
                        is_running = False
                        if proc_task and not proc_task.done():
                            proc_task.cancel()

                        result = await asyncio.to_thread(
                            engine.stop_stream, output_dir=resolved_out_dir, session_name=session_name
                        )
                        if session_title:
                            result.title = session_title
                            json_path = resolved_out_dir / f"{session_name}.json"
                            if json_path.exists():
                                try:
                                    jdata = json.loads(json_path.read_text(encoding="utf-8"))
                                    jdata["title"] = session_title
                                    json_path.write_text(
                                        json.dumps(jdata, indent=2, ensure_ascii=False), encoding="utf-8"
                                    )
                                except Exception:
                                    pass
                        # Save HTML, MD, SRT, JSON
                        try:
                            html_file = resolved_out_dir / f"{session_name}.html"
                            md_file = resolved_out_dir / f"{session_name}.md"
                            srt_file = resolved_out_dir / f"{session_name}.srt"
                            html_file.write_text(
                                result.to_html(title=session_title or f"Polyphon Live Session: {session_name}"),
                                encoding="utf-8",
                            )
                            md_file.write_text(result.to_markdown(title=session_title), encoding="utf-8")
                            srt_file.write_text(result.to_srt(), encoding="utf-8")
                        except Exception as err:
                            logger.warning("Could not write all live reports: %s", err)

                        await websocket.send_json(
                            {
                                "status": "completed",
                                "session_name": session_name,
                                "result": result.to_dict(),
                            }
                        )
                        break
        except (WebSocketDisconnect, RuntimeError):
            pass
        except Exception as e:
            logger.error("Error in streaming websocket: %s", e)
        finally:
            is_running = False
            if proc_task and not proc_task.done():
                proc_task.cancel()
            if getattr(engine, "_active_stream", False):
                result = engine.stop_stream(output_dir=resolved_out_dir, session_name=session_name)
                if session_title and result:
                    result.title = session_title
                    json_path = resolved_out_dir / f"{session_name}.json"
                    if json_path.exists():
                        try:
                            jdata = json.loads(json_path.read_text(encoding="utf-8"))
                            jdata["title"] = session_title
                            json_path.write_text(json.dumps(jdata, indent=2, ensure_ascii=False), encoding="utf-8")
                        except Exception:
                            pass
                try:
                    html_file = resolved_out_dir / f"{session_name}.html"
                    md_file = resolved_out_dir / f"{session_name}.md"
                    srt_file = resolved_out_dir / f"{session_name}.srt"
                    if not html_file.exists() and result:
                        html_file.write_text(
                            result.to_html(title=session_title or f"Polyphon Live Session: {session_name}"),
                            encoding="utf-8",
                        )
                    if not md_file.exists() and result:
                        md_file.write_text(result.to_markdown(title=session_title), encoding="utf-8")
                    if not srt_file.exists() and result:
                        srt_file.write_text(result.to_srt(), encoding="utf-8")
                except Exception as err:
                    logger.warning("Could not write all live reports on disconnect: %s", err)

    return app


async def _run_pipeline_job(
    job: JobState,
    output_dir: Path,
    diarize: bool,
    identify: bool,
    infer_names: bool,
    auto_enroll: bool,
    summarize: bool,
    language: str = "en",
    clean_fillers: bool = False,
):
    """Execute end-to-end processing pipeline in a background thread."""
    loop = asyncio.get_running_loop()

    def sync_worker():
        t_start = time.time()
        llm = OpenAICompatibleLLM() if summarize else None
        engine = PolyphonEngine(llm=llm)

        job.update(
            status="processing",
            stage=1,
            stage_name="Transcribing audio (Whisper large-v3)",
            percent=0.0,
            details="Loading audio & transcribing speech...",
        )

        def asr_cb(curr: float, tot: float):
            pct = round(min(curr / tot * 40.0, 39.5), 1) if tot > 0 else 0.0
            elapsed = int(time.time() - t_start)
            job.update(
                stage=1,
                percent=pct,
                details=f"Transcribed {int(curr)}s / {int(tot)}s audio ({elapsed}s elapsed)",
            )

        # Hook progress before diarize
        if diarize:
            orig_diarize = engine.diarizer.diarize

            def hooked_diarize(path, show_progress=False):
                job.update(
                    stage=2,
                    stage_name="Diarizing speaker turns (Pyannote 3.1)",
                    percent=42.0,
                    details="Segmenting speech & extracting speaker embeddings...",
                )
                stop_diarize_ticker = threading.Event()

                def diarize_ticker():
                    t0 = time.time()
                    while not stop_diarize_ticker.is_set():
                        time.sleep(1.5)
                        elapsed = int(time.time() - t0)
                        curr_pct = min(42.0 + (elapsed / 6.0), 76.0)
                        job.update(
                            stage=2,
                            percent=round(curr_pct, 1),
                            details=f"Diarizing audio chunks ({elapsed}s elapsed)...",
                        )

                ticker_t = threading.Thread(target=diarize_ticker, daemon=True)
                ticker_t.start()
                try:
                    res = orig_diarize(path, show_progress=False)
                finally:
                    stop_diarize_ticker.set()

                job.update(
                    stage=3,
                    stage_name="Reconciling turns & speaker identities",
                    percent=78.0,
                    details="Matching VoiceDB voiceprints & verbal self-introductions...",
                )
                return res

            engine.diarizer.diarize = hooked_diarize

        # Hook progress before LLM
        if summarize and engine.llm:
            orig_extract = engine.llm.extract_insights

            def hooked_extract(transcript_md):
                job.update(
                    stage=4,
                    stage_name=f"Extracting executive intelligence ({engine.llm.model})",
                    percent=86.0,
                    details=f"Prompting local {engine.llm.model} model...",
                )
                stop_llm_ticker = threading.Event()

                def llm_ticker():
                    t0 = time.time()
                    while not stop_llm_ticker.is_set():
                        time.sleep(1.5)
                        elapsed = int(time.time() - t0)
                        curr_pct = min(86.0 + (elapsed / 4.0), 98.0)
                        job.update(
                            stage=4,
                            percent=round(curr_pct, 1),
                            details=f"Analyzing dialogue & extracting decisions ({elapsed}s elapsed)...",
                        )

                ticker_t = threading.Thread(target=llm_ticker, daemon=True)
                ticker_t.start()
                try:
                    res = orig_extract(transcript_md)
                finally:
                    stop_llm_ticker.set()
                return res

            engine.llm.extract_insights = hooked_extract

        result: PolyphonResult = engine.process(
            audio_path=job.filepath,
            diarize=diarize,
            identify_speakers=identify,
            summarize=summarize,
            infer_names=infer_names,
            auto_enroll=auto_enroll,
            language=language,
            asr_progress_callback=asr_cb,
            verbose=False,
            clean_fillers=clean_fillers,
        )

        job.update(
            stage=4,
            stage_name="Saving reports & manifests",
            percent=99.0,
            details="Writing HTML, Markdown, JSON, and SRT...",
        )

        # Save result formats to output directory
        stem = job.filepath.stem
        json_file = output_dir / f"{stem}.json"
        live_json_file = output_dir / f"{stem}_live.json"
        html_file = output_dir / f"{stem}.html"
        md_file = output_dir / f"{stem}.md"
        srt_file = output_dir / f"{stem}.srt"

        # Check if an existing live session exists or live_segments already saved
        existing_live_segments = None
        existing_title = None
        if live_json_file.exists():
            try:
                live_data = json.loads(live_json_file.read_text(encoding="utf-8"))
                existing_live_segments = live_data.get("segments", [])
                existing_title = live_data.get("title")
            except Exception as e:
                logger.warning("Failed to load existing live segments from %s: %s", live_json_file, e)
        if json_file.exists():
            try:
                old_data = json.loads(json_file.read_text(encoding="utf-8"))
                if not existing_title and old_data.get("title"):
                    existing_title = old_data["title"]
                if not existing_live_segments:
                    if old_data.get("live_segments"):
                        existing_live_segments = old_data["live_segments"]
                    elif not old_data.get("insights") and old_data.get("segments"):
                        # This was a live stream session! Preserve its live segments
                        existing_live_segments = old_data.get("segments", [])
                        live_json_file.write_text(json.dumps(old_data, indent=2), encoding="utf-8")
            except Exception as e:
                logger.warning("Failed to check old json for live segments: %s", e)

        # Write result files
        if existing_title:
            result.title = existing_title
        json_dict = json.loads(result.to_json())
        if existing_title:
            json_dict["title"] = existing_title
        if existing_live_segments:
            json_dict["live_segments"] = existing_live_segments
            if not live_json_file.exists():
                live_json_file.write_text(
                    json.dumps({"segments": existing_live_segments, "title": existing_title}, indent=2),
                    encoding="utf-8",
                )

        json_file.write_text(json.dumps(json_dict, indent=2), encoding="utf-8")
        html_file.write_text(
            result.to_html(title=existing_title or f"Polyphon Report: {job.filename}", clean_fillers=clean_fillers),
            encoding="utf-8",
        )
        md_file.write_text(result.to_markdown(title=existing_title, clean_fillers=clean_fillers), encoding="utf-8")
        srt_file.write_text(result.to_srt(clean_fillers=clean_fillers), encoding="utf-8")

        # Convert result to serializable dict
        res_dict = {
            "duration": result.duration,
            "language": result.language,
            "speakers": [asdict(s) for s in result.speakers],
            "segments": [asdict(s) for s in result.segments],
            "insights": asdict(result.insights) if result.insights else None,
        }
        if existing_title:
            res_dict["title"] = existing_title
        if existing_live_segments:
            res_dict["live_segments"] = existing_live_segments

        total_elapsed = int(time.time() - t_start)
        job.result = res_dict
        job.update(
            status="completed",
            stage=4,
            stage_name="Complete",
            percent=100.0,
            details=f"Processed in {total_elapsed}s • All files generated",
        )

    try:
        await loop.run_in_executor(None, sync_worker)
    except Exception as e:
        job.error = str(e)
        job.update(status="failed", stage_name="Failed", details=str(e))
