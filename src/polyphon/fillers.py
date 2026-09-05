"""Filler word detection and clean verbatim text processing for Polyphon.

Implements Tier 1 involuntary vocal disfluency detection (um, uh, er, ah, etc.)
with zero impact on word-level timestamps, audio synchronization, or karaoke playback.
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polyphon.types import WordToken

# Standard Tier 1 acoustic vocal disfluencies (involuntary hesitation sounds).
# These carry zero semantic meaning and are universally safe to remove in clean verbatim.
DEFAULT_FILLER_WORDS: frozenset[str] = frozenset(
    {
        # English hesitation sounds
        "um",
        "uh",
        "er",
        "ah",
        "erm",
        "umm",
        "uhh",
        "err",
        "ahh",
        "hmm",
        "mhm",
        "mm",
        "mm-hmm",
        "mmhmm",
        "uh-huh",
        "uhhuh",
        "uh-oh",
        "uhoh",
        "eh",
        "huh",
        # Common phonetic variants
        "um-hum",
        "hum",
        # Common non-English disfluencies emitted by Whisper
        "euh",
        "ehm",
        "äh",
        "ähm",
        "öhm",
        "este",
    }
)

# Regex matching elongated hesitation sounds like "uuum", "uuuh", "hhmm", etc.
_ELONGATED_FILLER_RE = re.compile(
    r"^(u+m+|u+h+|e+r+m*|a+h+|h+m+|m+h+m+|e+u+h*|ä+h+m*)$",
    re.IGNORECASE,
)

# Punctuation to strip from token boundaries before matching
_PUNCTUATION_STRIP = ".,!?:;\"'()[]{}<>-~—/\\"


def get_custom_filler_words() -> set[str]:
    """Load additional filler words from environment if configured."""
    raw = os.environ.get("POLYPHON_CUSTOM_FILLERS", "").strip()
    if not raw:
        return set()
    return {w.strip().lower() for w in raw.split(",") if w.strip()}


def is_filler_word(word: str, custom_fillers: set[str] | None = None) -> bool:
    """Check whether a single token or word is a vocal disfluency/filler.

    Punctuation like commas, ellipses, and quotes are stripped before evaluation.
    """
    if not word:
        return False

    cleaned = word.strip().strip(_PUNCTUATION_STRIP).lower()
    if not cleaned:
        return False

    if cleaned in DEFAULT_FILLER_WORDS:
        return True

    if custom_fillers and cleaned in custom_fillers:
        return True

    env_fillers = get_custom_filler_words()
    if env_fillers and cleaned in env_fillers:
        return True

    # Check for elongated disfluencies like "uuum", "uhhh", "aaah"
    if len(cleaned) >= 2 and _ELONGATED_FILLER_RE.match(cleaned):
        return True

    return False


def clean_text_from_words(words: list[WordToken], custom_fillers: set[str] | None = None) -> str:
    """Produce clean verbatim text by joining only non-filler word tokens.

    Preserves original capitalization and trailing punctuation on non-filler words.
    """
    clean_parts: list[str] = []
    for w in words:
        if getattr(w, "is_filler", False) or is_filler_word(w.word, custom_fillers):
            continue
        clean_parts.append(w.word)
    return " ".join(clean_parts).strip()


def tag_filler_tokens(words: list[WordToken], custom_fillers: set[str] | None = None) -> list[WordToken]:
    """Tag is_filler attribute on WordTokens in-place without altering timestamps."""
    for w in words:
        w.is_filler = is_filler_word(w.word, custom_fillers)
    return words
