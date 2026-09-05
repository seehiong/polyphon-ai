"""Unit tests for filler word detection and clean verbatim formatting in Polyphon."""

import os

from polyphon.fillers import (
    clean_text_from_words,
    is_filler_word,
    tag_filler_tokens,
)
from polyphon.types import PolyphonResult, Segment, SpeakerInfo, WordToken


def test_standard_filler_words_detection():
    """Verify that all standard Tier 1 vocal disfluencies are detected."""
    disfluencies = [
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
        "uh-huh",
        "uh-oh",
        "eh",
        "huh",
    ]
    for word in disfluencies:
        assert is_filler_word(word), f"Expected '{word}' to be identified as a filler"
        assert is_filler_word(word.upper()), f"Expected uppercase '{word.upper()}' to be identified as a filler"
        assert is_filler_word(
            word.capitalize()
        ), f"Expected capitalized '{word.capitalize()}' to be identified as a filler"


def test_filler_with_punctuation():
    """Verify punctuation surrounding filler words is stripped during detection."""
    assert is_filler_word("Um,")
    assert is_filler_word("...uh")
    assert is_filler_word("(er)")
    assert is_filler_word("ah!")
    assert is_filler_word('"hmm"')
    assert is_filler_word("—erm—")


def test_elongated_filler_words():
    """Verify phonetic regex detects elongated disfluencies."""
    assert is_filler_word("uuum")
    assert is_filler_word("uuuuummmmm")
    assert is_filler_word("uhhh")
    assert is_filler_word("aaah")
    assert is_filler_word("hhmm")


def test_legitimate_words_not_flagged():
    """Verify regular vocabulary containing filler substrings is not flagged."""
    normal_words = [
        "umbrella",
        "understand",
        "urgent",
        "error",
        "aha",
        "home",
        "hammer",
        "like",
        "you",
        "know",
        "actually",
        "basically",
    ]
    for word in normal_words:
        assert not is_filler_word(word), f"Word '{word}' should NOT be flagged as filler"


def test_custom_fillers():
    """Verify custom filler words via argument and environment variable."""
    # Argument
    assert not is_filler_word("yo")
    assert is_filler_word("yo", custom_fillers={"yo"})

    # Environment variable
    old_env = os.environ.get("POLYPHON_CUSTOM_FILLERS")
    try:
        os.environ["POLYPHON_CUSTOM_FILLERS"] = "customfiller, indeed_filler"
        assert is_filler_word("customfiller")
        assert is_filler_word("indeed_filler")
        assert not is_filler_word("otherword")
    finally:
        if old_env is None:
            os.environ.pop("POLYPHON_CUSTOM_FILLERS", None)
        else:
            os.environ["POLYPHON_CUSTOM_FILLERS"] = old_env


def test_clean_text_from_words_preserves_casing_and_punctuation():
    """Verify clean_text_from_words produces clean dialogue without touching non-fillers."""
    words = [
        WordToken(word="Well,", start=0.0, end=0.3),
        WordToken(word="um,", start=0.35, end=0.6, is_filler=True),
        WordToken(word="I", start=0.65, end=0.8),
        WordToken(word="think", start=0.85, end=1.1),
        WordToken(word="uh", start=1.15, end=1.3, is_filler=True),
        WordToken(word="we", start=1.35, end=1.5),
        WordToken(word="should", start=1.55, end=1.8),
        WordToken(word="proceed.", start=1.85, end=2.2),
    ]

    cleaned = clean_text_from_words(words)
    assert cleaned == "Well, I think we should proceed."


def test_tag_filler_tokens_preserves_timestamps():
    """Verify tag_filler_tokens tags tokens in-place with zero timing impact."""
    words = [
        WordToken(word="Um", start=10.123, end=10.456, score=0.98),
        WordToken(word="hello", start=10.500, end=11.000, score=0.99),
        WordToken(word="world", start=11.050, end=11.600, score=0.95),
    ]

    tag_filler_tokens(words)

    assert words[0].is_filler is True
    assert words[0].start == 10.123
    assert words[0].end == 10.456
    assert words[0].score == 0.98

    assert words[1].is_filler is False
    assert words[1].start == 10.500
    assert words[1].end == 11.000

    assert words[2].is_filler is False
    assert words[2].start == 11.050
    assert words[2].end == 11.600


def test_segment_get_clean_text():
    """Verify Segment.get_clean_text() works for both tokenized and untokenized segments."""
    # With WordTokens
    seg = Segment(
        start=0.0,
        end=2.0,
        speaker_id="SPEAKER_00",
        speaker_name="Alice",
        text="Um, hello there uh world.",
        words=[
            WordToken(word="Um,", start=0.0, end=0.3, is_filler=True),
            WordToken(word="hello", start=0.35, end=0.7),
            WordToken(word="there", start=0.75, end=1.1),
            WordToken(word="uh", start=1.15, end=1.4, is_filler=True),
            WordToken(word="world.", start=1.45, end=2.0),
        ],
    )
    assert seg.get_clean_text() == "hello there world."

    # Without WordTokens (falls back to text filtering)
    seg_untokenized = Segment(
        start=0.0,
        end=2.0,
        speaker_id="SPEAKER_00",
        speaker_name="Alice",
        text="Um hello uh world",
        words=[],
    )
    assert seg_untokenized.get_clean_text() == "hello world"


def test_markdown_and_srt_clean_verbatim_exports():
    """Verify to_markdown, to_srt, and to_html export clean verbatim when requested."""
    seg1 = Segment(
        start=0.0,
        end=2.0,
        speaker_id="spk1",
        speaker_name="Alice",
        text="Um hello world",
        words=[
            WordToken(word="Um", start=0.0, end=0.4, is_filler=True),
            WordToken(word="hello", start=0.5, end=1.0),
            WordToken(word="world", start=1.1, end=1.6),
        ],
    )
    # seg2 is entirely filler
    seg2 = Segment(
        start=2.5,
        end=3.0,
        speaker_id="spk2",
        speaker_name="Bob",
        text="Uh, um.",
        words=[
            WordToken(word="Uh,", start=2.5, end=2.7, is_filler=True),
            WordToken(word="um.", start=2.75, end=3.0, is_filler=True),
        ],
    )
    seg3 = Segment(
        start=3.5,
        end=5.0,
        speaker_id="spk2",
        speaker_name="Bob",
        text="Sounds, er, great!",
        words=[
            WordToken(word="Sounds,", start=3.5, end=4.0),
            WordToken(word="er,", start=4.05, end=4.3, is_filler=True),
            WordToken(word="great!", start=4.35, end=5.0),
        ],
    )

    result = PolyphonResult(
        duration=5.0,
        speakers=[SpeakerInfo(id="spk1", name="Alice"), SpeakerInfo(id="spk2", name="Bob")],
        segments=[seg1, seg2, seg3],
    )

    # Full verbatim markdown
    md_verbatim = result.to_markdown(clean_fillers=False)
    assert "Um hello world" in md_verbatim
    assert "Uh, um." in md_verbatim
    assert "Sounds, er, great!" in md_verbatim

    # Clean verbatim markdown: filler words removed, entirely-filler turns skipped
    md_clean = result.to_markdown(clean_fillers=True)
    assert "hello world" in md_clean
    assert "Um" not in md_clean
    assert "Uh, um." not in md_clean
    assert "Sounds, great!" in md_clean

    # Clean verbatim SRT: subtitle indexes contiguous (1, 2), timestamp of Bob's turn unchanged
    srt_clean = result.to_srt(clean_fillers=True)
    assert "1\n00:00:00,000 --> 00:00:02,000\n[Alice] hello world" in srt_clean
    # Bob's turn is subtitle #2 (not #3) and starts at 00:00:03,500
    assert "2\n00:00:03,500 --> 00:00:05,000\n[Bob] Sounds, great!" in srt_clean
    assert "3\n" not in srt_clean


def test_json_serialization_preserves_and_rebuilds_fillers():
    """Verify JSON serialization stores is_filler and from_dict recovers it."""
    words = [
        WordToken(word="um", start=1.0, end=1.5, is_filler=True),
        WordToken(word="yes", start=1.6, end=2.0, is_filler=False),
    ]
    seg = Segment(
        start=1.0,
        end=2.0,
        speaker_id="spk1",
        speaker_name="Alice",
        text="um yes",
        words=words,
    )
    result = PolyphonResult(
        duration=2.0,
        speakers=[SpeakerInfo(id="spk1", name="Alice")],
        segments=[seg],
    )

    res_dict = result.to_dict()
    assert res_dict["segments"][0]["words"][0]["is_filler"] is True
    assert res_dict["segments"][0]["words"][1]["is_filler"] is False

    reconstructed = PolyphonResult.from_dict(res_dict)
    assert reconstructed.segments[0].words[0].is_filler is True
    assert reconstructed.segments[0].words[1].is_filler is False

    # Historical JSON simulation (where is_filler was not stored)
    historical_dict = {
        "duration": 2.0,
        "speakers": [{"id": "spk1", "name": "Alice"}],
        "segments": [
            {
                "start": 1.0,
                "end": 2.0,
                "speaker_id": "spk1",
                "speaker_name": "Alice",
                "text": "uh huh yes",
                "words": [
                    {"word": "uh", "start": 1.0, "end": 1.2},
                    {"word": "huh", "start": 1.25, "end": 1.4},
                    {"word": "yes", "start": 1.5, "end": 1.8},
                ],
            }
        ],
    }
    hist_result = PolyphonResult.from_dict(historical_dict)
    assert hist_result.segments[0].words[0].is_filler is True
    assert hist_result.segments[0].words[1].is_filler is True
    assert hist_result.segments[0].words[2].is_filler is False
    assert hist_result.segments[0].get_clean_text() == "yes"
