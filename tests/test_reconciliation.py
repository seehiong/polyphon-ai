from polyphon.reconciliation.base import ReconciliationEngine
from polyphon.types import SpeakerInterval, WordToken


def test_reconciliation_basic_two_speakers():
    reconciler = ReconciliationEngine()

    words = [
        WordToken(word="Hello", start=1.0, end=1.5),
        WordToken(word="Alice", start=1.6, end=2.0),
        WordToken(word="Hi", start=2.2, end=2.5),
        WordToken(word="Bob", start=2.6, end=3.0),
    ]

    intervals = [
        SpeakerInterval(speaker_id="SPEAKER_00", start=0.9, end=2.1),
        SpeakerInterval(speaker_id="SPEAKER_01", start=2.15, end=3.1),
    ]

    speaker_map = {
        "SPEAKER_00": "Alice",
        "SPEAKER_01": "Bob",
    }

    segments = reconciler.reconcile(words, intervals, speaker_map)

    assert len(segments) == 2
    assert segments[0].speaker_name == "Alice"
    assert segments[0].text == "Hello Alice"
    assert segments[1].speaker_name == "Bob"
    assert segments[1].text == "Hi Bob"


def test_reconciliation_gap_tolerance():
    reconciler = ReconciliationEngine(gap_tolerance_sec=1.0)

    # Word slightly after interval ends
    words = [
        WordToken(word="Continuing", start=3.2, end=3.6),
    ]

    intervals = [
        SpeakerInterval(speaker_id="SPEAKER_00", start=1.0, end=3.0),
    ]

    segments = reconciler.reconcile(words, intervals)
    assert len(segments) == 1
    assert segments[0].speaker_id == "SPEAKER_00"
    assert segments[0].text == "Continuing"
