from unittest.mock import MagicMock

from polyphon.nlp.speaker_inference import SpeakerNameInferer
from polyphon.types import Segment


def test_speaker_name_infer_from_text_patterns():
    # 1. Name + Role pattern
    assert (
        SpeakerNameInferer.infer_from_text("Hello everybody, I'm Sarah, project manager and this is our first meeting")
        == "Sarah (Project Manager)"
    )
    assert (
        SpeakerNameInferer.infer_from_text("Hi team, I am Alex, the software engineer on this project.")
        == "Alex (Software Engineer)"
    )

    # 2. My name is pattern
    assert SpeakerNameInferer.infer_from_text("Good morning, my name is Mark Zuckerberg.") == "Mark Zuckerberg"
    assert SpeakerNameInferer.infer_from_text("Hello, my name's Sarah.") == "Sarah"

    # 3. Name here pattern
    assert SpeakerNameInferer.infer_from_text("Hey everyone, David here from the industrial design team.") == "David"

    # 4. This is pattern
    assert SpeakerNameInferer.infer_from_text("Welcome back, this is Lex Fridman.") == "Lex Fridman"

    # 5. False positives should return None
    assert SpeakerNameInferer.infer_from_text("I'm sure we will have a great time.") is None
    assert SpeakerNameInferer.infer_from_text("I am not going to turn off the lights.") is None
    assert SpeakerNameInferer.infer_from_text("I'm sorry for being late.") is None
    assert SpeakerNameInferer.infer_from_text("This is what we need to focus on.") is None


def test_speaker_name_infer_names_from_segments():
    segments = [
        Segment(
            start=0.0,
            end=5.0,
            speaker_id="SPEAKER_00",
            speaker_name="Speaker 00",
            text="Hello everybody, I'm Sarah, project manager and this is our kickoff.",
        ),
        Segment(
            start=6.0,
            end=12.0,
            speaker_id="SPEAKER_01",
            speaker_name="Speaker 01",
            text="Hi Sarah, my name is David.",
        ),
        Segment(
            start=13.0,
            end=18.0,
            speaker_id="SPEAKER_02",
            speaker_name="Speaker 02",
            text="Yes, totally agree on the schedule.",
        ),
    ]

    inferred = SpeakerNameInferer.infer_names(segments)
    assert inferred["SPEAKER_00"] == "Sarah (Project Manager)"
    assert inferred["SPEAKER_01"] == "David"
    assert "SPEAKER_02" not in inferred


def test_speaker_name_infer_names_with_llm_fallback():
    segments = [
        Segment(
            start=0.0,
            end=5.0,
            speaker_id="SPEAKER_03",
            speaker_name="Speaker 03",
            text="Let's pass it over to marketing.",
        )
    ]

    mock_llm = MagicMock()
    mock_llm.infer_speakers.return_value = {"SPEAKER_03": "Elena (Marketing)"}

    inferred = SpeakerNameInferer.infer_names(segments, llm=mock_llm)
    assert inferred.get("SPEAKER_03") == "Elena (Marketing)"
