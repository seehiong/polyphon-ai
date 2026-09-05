"""Batch speaker assignment and VoiceDB enrollment."""

from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from polyphon.types import PolyphonResult
from polyphon.voicedb.base import VoiceDB

logger = logging.getLogger(__name__)


def parse_speaker_map(speaker_map_input: Any) -> dict[str, str]:
    """Parse speaker mapping inputs into a normalized dictionary.

    Supports:
      - Dictionary: {"SPEAKER_00": "Alice", "1": "Bob"}
      - Comma-separated string: "SPEAKER_00=Alice,SPEAKER_01=Bob" or "0=Alice,1=Bob"
      - List of strings: ["SPEAKER_00=Alice", "SPEAKER_01=Bob"]
    """
    if not speaker_map_input:
        return {}

    if isinstance(speaker_map_input, dict):
        raw_items = [f"{k}={v}" for k, v in speaker_map_input.items()]
    elif isinstance(speaker_map_input, str):
        raw_items = [item.strip() for item in speaker_map_input.split(",") if item.strip()]
    elif isinstance(speaker_map_input, (list, tuple)):
        raw_items = []
        for entry in speaker_map_input:
            for item in str(entry).split(","):
                if item.strip():
                    raw_items.append(item.strip())
    else:
        return {}

    result = {}
    for item in raw_items:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        clean_k = k.strip()
        clean_v = v.strip().strip("'\"")

        if not clean_k.startswith("SPEAKER_"):
            if clean_k.isdigit():
                clean_k = f"SPEAKER_{clean_k.zfill(2)}"
            else:
                clean_k = f"SPEAKER_{clean_k}"

        result[clean_k] = clean_v

    return result


def find_meeting_manifest(meeting_path_or_id: str | Path, output_dir: Path | None = None) -> Path | None:
    """Locate meeting manifest JSON by path or identifier."""
    cand = Path(meeting_path_or_id)
    if cand.exists() and cand.is_file():
        return cand

    out_dir = output_dir or Path("outputs")
    stem = cand.stem

    for candidate in [
        out_dir / f"{stem}.json",
        out_dir / f"{stem}_live.json",
        out_dir / stem,
        cand.with_suffix(".json"),
    ]:
        if candidate.exists() and candidate.is_file():
            return candidate

    return None


def find_audio_file(manifest_data: dict[str, Any], manifest_path: Path) -> Path | None:
    """Locate the media or audio file associated with a meeting manifest."""
    stem = manifest_path.stem.replace("_live", "")
    search_dirs = [
        manifest_path.parent,
        Path("examples"),
        Path("outputs"),
        Path("uploads"),
    ]
    search_exts = [".wav", ".mp4", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".mov"]

    for key in ["audio_path", "media_path", "filename"]:
        val = manifest_data.get(key)
        if val:
            p = Path(val)
            if p.exists() and p.is_file():
                return p
            for s_dir in search_dirs:
                p_cand = s_dir / p.name
                if p_cand.exists() and p_cand.is_file():
                    return p_cand

    for s_dir in search_dirs:
        for ext in search_exts:
            p_cand = s_dir / f"{stem}{ext}"
            if p_cand.exists() and p_cand.is_file():
                return p_cand

    return None


def assign_speakers_to_manifest(
    meeting_path_or_id: str | Path,
    speaker_map: Any,
    enroll: bool = True,
    summarize: bool = False,
    llm: Any = None,
    save: bool = True,
    output_dir: Path | None = None,
    verbose: bool = True,
) -> tuple[dict[str, Any], list[str]]:
    """Assign real speaker identities to an existing meeting manifest.

    Optionally enrolls mapped speakers into VoiceDB and regenerates HTML/MD/SRT reports.
    """
    clean_map = parse_speaker_map(speaker_map)
    if not clean_map:
        raise ValueError("No valid speaker mappings provided (e.g. SPEAKER_00=Alice)")

    manifest_path = find_meeting_manifest(meeting_path_or_id, output_dir=output_dir)
    if not manifest_path:
        raise FileNotFoundError(f"Meeting manifest not found for '{meeting_path_or_id}'")

    try:
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to read meeting manifest at {manifest_path}: {exc}") from exc

    # 1. Update speakers array
    seen_ids = set()
    for spk in manifest_data.get("speakers", []):
        spk_id = spk.get("id")
        seen_ids.add(spk_id)
        if spk_id in clean_map:
            spk["name"] = clean_map[spk_id]
            spk["confidence"] = 1.0

    for spk_id, new_name in clean_map.items():
        if spk_id not in seen_ids:
            manifest_data.setdefault("speakers", []).append(
                {
                    "id": spk_id,
                    "name": new_name,
                    "confidence": 1.0,
                }
            )

    # 2. Update segments and live_segments
    for seg in manifest_data.get("segments", []):
        spk_id = seg.get("speaker_id")
        if spk_id in clean_map:
            seg["speaker_name"] = clean_map[spk_id]

    for seg in manifest_data.get("live_segments", []):
        spk_id = seg.get("speaker_id")
        if spk_id in clean_map:
            seg["speaker_name"] = clean_map[spk_id]

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

    # 3. Update speaker_stats in insights
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

    # 4. Optional VoiceDB Enrollment
    enrolled_speakers: list[str] = []
    if enroll:
        audio_file = find_audio_file(manifest_data, manifest_path)
        if audio_file:
            from polyphon.diarization.base import PyannoteDiarizer

            diarizer = PyannoteDiarizer()
            vdb = VoiceDB()
            samples_dir = manifest_path.parent / "samples"
            samples_dir.mkdir(parents=True, exist_ok=True)

            all_segments = manifest_data.get("segments") or manifest_data.get("live_segments") or []

            for spk_id, new_name in clean_map.items():
                spk_segs = [s for s in all_segments if s.get("speaker_id") == spk_id]
                viable_segs = [s for s in spk_segs if (s.get("end", 0.0) - s.get("start", 0.0)) >= 0.5]
                viable_segs.sort(key=lambda s: (s.get("end", 0.0) - s.get("start", 0.0)), reverse=True)

                if not viable_segs:
                    continue

                embs = []
                for seg in viable_segs[:5]:
                    st = float(seg.get("start", 0.0))
                    en = float(seg.get("end", 0.0))
                    if en > st:
                        try:
                            emb = diarizer.extract_single_embedding(str(audio_file), start=st, end=en)
                            embs.append(emb)
                        except Exception as e:
                            logger.debug(f"Could not extract embedding for segment {st}-{en}: {e}")

                if embs:
                    avg_emb = np.mean(embs, axis=0)
                    norm = np.linalg.norm(avg_emb)
                    norm_emb = (avg_emb / norm) if norm > 0 else avg_emb

                    # Slice audio sample snippet
                    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", new_name.lower()).strip("_")
                    sample_path = samples_dir / f"sample_{slug}.wav"
                    best_seg = viable_segs[0]
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
                            str(sample_path),
                        ]
                        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                    except Exception as e:
                        logger.warning(f"Could not slice sample audio for {new_name}: {e}")
                        sample_path = None

                    vdb.enroll(
                        name=new_name,
                        embedding=norm_emb,
                        speaker_id=slug,
                        audio_path=str(sample_path) if sample_path and sample_path.exists() else None,
                    )
                    enrolled_speakers.append(new_name)

    # 5. Optional Re-summarization
    if summarize:
        active_llm = llm
        if active_llm is None:
            try:
                from polyphon.llm.base import OpenAICompatibleLLM

                active_llm = OpenAICompatibleLLM()
            except Exception:
                active_llm = None

        if active_llm:
            try:
                # extract_insights expects a rendered markdown transcript, and
                # Insights is a dataclass -- not a pydantic model.
                transcript_md = PolyphonResult.from_dict(manifest_data).to_markdown()
                new_insights = active_llm.extract_insights(transcript_md)
                manifest_data["insights"] = asdict(new_insights)
            except Exception as e:
                logger.warning(f"Failed to re-summarize meeting insights: {e}")

    # 6. Save Manifest & Regenerate Reports
    if save:
        manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
        stem = manifest_path.stem.replace("_live", "")
        out_dir = manifest_path.parent

        try:
            poly_result = PolyphonResult.from_dict(manifest_data)
            (out_dir / f"{stem}.html").write_text(
                poly_result.to_html(title=f"Polyphon Report: {stem}"), encoding="utf-8"
            )
            (out_dir / f"{stem}.md").write_text(poly_result.to_markdown(), encoding="utf-8")
            (out_dir / f"{stem}.srt").write_text(poly_result.to_srt(), encoding="utf-8")
        except Exception as exc:
            logger.warning(f"Could not regenerate HTML/MD reports after assign: {exc}")

    return manifest_data, enrolled_speakers
