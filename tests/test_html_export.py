from polyphon.types import (
    ActionItem,
    Insights,
    PolyphonResult,
    Segment,
    SpeakerInfo,
    SpeakerStat,
)


def test_html_export_structure():
    result = PolyphonResult(
        duration=125.0,
        language="en",
        speakers=[
            SpeakerInfo(id="SPEAKER_00", name="Sarah"),
            SpeakerInfo(id="SPEAKER_01", name="Mark"),
        ],
        segments=[
            Segment(
                start=0.0,
                end=10.0,
                speaker_id="SPEAKER_00",
                speaker_name="Sarah",
                text="Welcome to the kickoff meeting.",
            ),
            Segment(
                start=10.5,
                end=25.0,
                speaker_id="SPEAKER_01",
                speaker_name="Mark",
                text="Excited to work on the remote control.",
            ),
        ],
        insights=Insights(
            summary="Kickoff meeting discussing remote control.",
            topics=["Kickoff", "Design"],
            decisions=["Keep selling price at 25 euros"],
            action_items=[ActionItem(task="Order components", assignee="Sarah")],
            speaker_stats={
                "Sarah": SpeakerStat(talk_time_seconds=10.0, percentage=40.0, segment_count=1),
                "Mark": SpeakerStat(talk_time_seconds=14.5, percentage=60.0, segment_count=1),
            },
        ),
    )

    html_text = result.to_html(title="Kickoff Report")

    assert "<!DOCTYPE html>" in html_text
    assert "<title>Kickoff Report</title>" in html_text
    assert "Sarah" in html_text
    assert "Mark" in html_text
    assert "Welcome to the kickoff meeting." in html_text
    assert "Kickoff meeting discussing remote control." in html_text
    assert "Keep selling price at 25 euros" in html_text
    assert "Order components" in html_text
    assert 'class="talktime-bar"' in html_text
    assert 'id="searchBox"' in html_text
    assert "filterSpeaker" in html_text
