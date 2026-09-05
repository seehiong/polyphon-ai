"""In-Context Verbal Speaker Identification & Name Inference."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from polyphon.types import Segment

if TYPE_CHECKING:
    from polyphon.llm.base import LLMBackend


class SpeakerNameInferer:
    """Discovers speaker names and titles directly from dialogue patterns and self-introductions."""

    # Common job titles / roles for clean normalization
    KNOWN_ROLES = {
        "project manager": "Project Manager",
        "product manager": "Product Manager",
        "industrial designer": "Industrial Designer",
        "user interface designer": "UI Designer",
        "marketing expert": "Marketing",
        "software engineer": "Software Engineer",
        "lead developer": "Lead Developer",
        "designer": "Designer",
        "host": "Host",
    }

    @classmethod
    def _clean_role(cls, role_str: str) -> str | None:
        if not role_str:
            return None
        role_lower = role_str.strip().lower()
        for key, val in cls.KNOWN_ROLES.items():
            if key in role_lower:
                return val
        # Clean title case if short (<= 3 words)
        words = role_lower.split()
        if len(words) <= 3:
            return " ".join(w.capitalize() for w in words)
        return None

    @classmethod
    def infer_from_text(cls, text: str) -> str | None:
        """Extract speaker name and optional role from self-introduction phrases."""
        # 1. "I'm <Name>, <Role> ..." or "I am <Name>, the <Role> ..."
        match_role = re.search(
            r"\b(?:i am|i'm|im)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)(?:,\s*(?:the\s+)?([^,.]+?))?(?:\band\b|\bthis\b|\bokay\b|\bso\b|\.|\,|$)",
            text,
            re.IGNORECASE,
        )
        if match_role:
            raw_name = match_role.group(1).strip()
            first_w = raw_name.split()[0].lower()
            false_positives = {
                "sure",
                "not",
                "sorry",
                "glad",
                "happy",
                "ready",
                "going",
                "just",
                "very",
                "here",
                "also",
                "fine",
                "ok",
                "okay",
                "trying",
                "looking",
                "working",
            }
            if first_w not in false_positives:
                name = " ".join(w.capitalize() for w in raw_name.split())
                raw_role = match_role.group(2)
                clean_role = cls._clean_role(raw_role) if raw_role else None
                if clean_role:
                    return f"{name} ({clean_role})"
                return name

        # 2. "My name is <Name> ..."
        match_name = re.search(
            r"\b(?:my name is|my name's)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            text,
            re.IGNORECASE,
        )
        if match_name:
            raw_name = match_name.group(1).strip()
            return " ".join(w.capitalize() for w in raw_name.split())

        # 3. "<Name> here from ..." or "<Name> here, ..."
        match_here = re.search(
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+here(?:\s+(?:from|with|,|\.))",
            text,
            re.IGNORECASE,
        )
        if match_here:
            raw_name = match_here.group(1).strip()
            first_w = raw_name.split()[0].lower()
            if first_w not in {
                "someone",
                "anyone",
                "everyone",
                "people",
                "over",
                "right",
                "down",
                "up",
                "out",
            }:
                return " ".join(w.capitalize() for w in raw_name.split())

        # 4. "This is <Name> ..."
        match_this = re.search(
            r"\bthis is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            text,
            re.IGNORECASE,
        )
        if match_this:
            raw_name = match_this.group(1).strip()
            first_w = raw_name.split()[0].lower()
            false_positives = {
                "a",
                "an",
                "the",
                "our",
                "my",
                "your",
                "his",
                "her",
                "their",
                "what",
                "how",
                "why",
                "where",
                "when",
                "who",
                "which",
                "it",
                "not",
                "all",
                "just",
                "so",
                "going",
                "one",
                "two",
                "part",
                "good",
                "great",
                "something",
                "really",
            }
            if first_w not in false_positives:
                return " ".join(w.capitalize() for w in raw_name.split())

        return None

    @classmethod
    def infer_names(
        cls,
        segments: list[Segment],
        llm: LLMBackend | None = None,
        max_turns_per_speaker: int = 4,
    ) -> dict[str, str]:
        """Infer speaker names from dialogue turns using rule-based patterns and optional LLM resolution.

        Returns:
            Dictionary mapping speaker_id (e.g. 'SPEAKER_00') to inferred name (e.g. 'Sarah (Project Manager)').
        """
        inferred_map: dict[str, str] = {}

        # 1. Fast Pattern-Based Inference on each speaker's opening turns
        speaker_turns: dict[str, list[str]] = {}
        for seg in segments:
            turns = speaker_turns.setdefault(seg.speaker_id, [])
            if len(turns) < max_turns_per_speaker:
                turns.append(seg.text)

        for spk_id, turns in speaker_turns.items():
            for turn_text in turns:
                name = cls.infer_from_text(turn_text)
                if name:
                    inferred_map[spk_id] = name
                    break

        # 2. LLM-Based Inference for unmapped speakers if LLM backend is provided
        unmapped_speakers = [s for s in speaker_turns if s not in inferred_map]
        if unmapped_speakers and llm is not None:
            try:
                snippet_lines = [f"[{seg.speaker_id}]: {seg.text.strip()}" for seg in segments[:10]]
                dialogue_snippet = "\n".join(snippet_lines)
                llm_parsed = llm.infer_speakers(dialogue_snippet)
                if isinstance(llm_parsed, dict):
                    for spk_id, inferred_name in llm_parsed.items():
                        if spk_id in unmapped_speakers and isinstance(inferred_name, str) and inferred_name.strip():
                            inferred_map[spk_id] = inferred_name.strip()
            except Exception:
                # LLM inference is best-effort; pattern matching has already completed
                pass

        return inferred_map
