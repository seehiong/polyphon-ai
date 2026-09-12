"""Polyphon: Local, identity-aware speech transcription and diarization engine."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("polyphon-ai")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0.dev0"

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
