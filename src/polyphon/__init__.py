"""Polyphon: Local, identity-aware speech transcription and diarization engine."""

__version__ = "0.1.0"

# Load .env before any submodule reads os.environ for defaults.
from polyphon.config import load_env

load_env()

from polyphon.assign import assign_speakers_to_manifest, parse_speaker_map
from polyphon.diarization.base import PyannoteDiarizer, SortformerDiarizer
from polyphon.engine import PolyphonEngine
from polyphon.reconciliation.base import ReconciliationEngine
from polyphon.streaming import (
    AudioStreamBuffer,
    StreamingEvent,
    StreamingPolyphonEngine,
)
from polyphon.types import (
    ActionItem,
    Insights,
    PolyphonResult,
    Segment,
    SpeakerInfo,
    SpeakerInterval,
    SpeakerStat,
    WordToken,
)
from polyphon.voicedb.base import VoiceDB

__all__ = [
    "ActionItem",
    "AudioStreamBuffer",
    "Insights",
    "PolyphonEngine",
    "PolyphonResult",
    "PyannoteDiarizer",
    "ReconciliationEngine",
    "Segment",
    "SortformerDiarizer",
    "SpeakerInfo",
    "SpeakerInterval",
    "SpeakerStat",
    "StreamingEvent",
    "StreamingPolyphonEngine",
    "VoiceDB",
    "WordToken",
    "assign_speakers_to_manifest",
    "parse_speaker_map",
]
