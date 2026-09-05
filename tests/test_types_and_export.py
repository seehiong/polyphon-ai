import json

from polyphon.reconciliation.base import ReconciliationEngine
from polyphon.types import PolyphonResult, Segment, SpeakerInfo, WordToken


def test_polyphon_result_export():
    words = [
        WordToken(word="Good", start=0.0, end=0.3),
        WordToken(word="morning.", start=0.4, end=0.9),
    ]
    seg = Segment(
        start=0.0,
        end=0.9,
        speaker_id="spk_01",
        speaker_name="Alice",
        text="Good morning.",
        words=words,
    )
    result = PolyphonResult(
        duration=1.0,
        language="en",
        speakers=[SpeakerInfo(id="spk_01", name="Alice")],
        segments=[seg],
    )

    # Test Markdown export
    md = result.to_markdown()
    assert "[00:00] Alice:" in md
    assert "Good morning." in md

    # Test JSON export
    json_str = result.to_json()
    data = json.loads(json_str)
    assert data["duration"] == 1.0
    assert data["speakers"][0]["name"] == "Alice"
    assert data["segments"][0]["text"] == "Good morning."

    # Test SRT export
    srt = result.to_srt()
    assert "00:00:00,000 --> 00:00:00,900" in srt
    assert "[Alice] Good morning." in srt

    # Test Speaker Stats
    stats = ReconciliationEngine.compute_speaker_stats([seg], total_duration=1.0)
    assert "Alice" in stats
    assert (
        stats["Alice"].percentage == 90.0
        or stats["Alice"].percentage == 100.0
        or stats["Alice"].talk_time_seconds == 0.9
    )
