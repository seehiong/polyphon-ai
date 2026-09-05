import tempfile
from unittest.mock import MagicMock, patch

import numpy as np

from polyphon.engine import PolyphonEngine
from polyphon.types import SpeakerInterval, WordToken
from polyphon.voicedb.base import VoiceDB


def test_engine_infer_names_and_auto_enroll():
    with tempfile.TemporaryDirectory() as tmpdir:
        vdb = VoiceDB(db_path=tmpdir)

        mock_asr = MagicMock()
        mock_words = [
            WordToken(word="Hello", start=0.0, end=0.5, score=0.99),
            WordToken(word="everybody,", start=0.6, end=1.0, score=0.99),
            WordToken(word="I'm", start=1.1, end=1.3, score=0.99),
            WordToken(word="Sarah,", start=1.4, end=1.8, score=0.99),
            WordToken(word="project", start=1.9, end=2.3, score=0.99),
            WordToken(word="manager", start=2.4, end=2.9, score=0.99),
        ]
        mock_asr.transcribe.return_value = ("en", mock_words)

        mock_diarizer = MagicMock()
        mock_diarizer.diarize.return_value = [
            SpeakerInterval(
                speaker_id="SPEAKER_00",
                start=0.0,
                end=3.0,
                is_exclusive=True,
                confidence=1.0,
            )
        ]
        # Return dummy embedding vector for auto-enroll
        mock_diarizer.extract_speaker_embeddings.return_value = {"SPEAKER_00": np.ones(64, dtype=np.float32)}

        engine = PolyphonEngine(asr=mock_asr, diarizer=mock_diarizer, voicedb=vdb)

        # Process with infer_names and auto_enroll
        with patch("polyphon.engine.get_audio_duration", return_value=3.0):
            result = engine.process(
                audio_path="dummy.wav",
                diarize=True,
                identify_speakers=True,
                infer_names=True,
                auto_enroll=True,
                verbose=False,
            )

        # 1. Check that speaker name was inferred in segments and speaker info
        assert len(result.segments) == 1
        assert result.segments[0].speaker_name == "Sarah (Project Manager)"
        assert result.speakers[0].name == "Sarah (Project Manager)"

        # 2. Check that speaker was automatically enrolled into VoiceDB!
        enrolled_speakers = vdb.list_speakers()
        assert len(enrolled_speakers) == 1
        assert enrolled_speakers[0]["name"] == "Sarah (Project Manager)"
