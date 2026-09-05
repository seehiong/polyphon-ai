from unittest.mock import MagicMock

import numpy as np

from polyphon.diarization.base import PyannoteDiarizer


def test_diarization_extract_single_embedding():
    diarizer = PyannoteDiarizer()
    mock_inference = MagicMock()
    # Return dummy 128-dim embedding vector
    mock_emb = np.ones(128, dtype=np.float32)
    mock_inference.return_value = mock_emb
    diarizer._embedding_inference = mock_inference

    emb = diarizer.extract_single_embedding("dummy.wav")
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (128,)
    # Should be L2-normalized
    assert np.isclose(np.linalg.norm(emb), 1.0)


def test_diarization_extract_speaker_embeddings():
    diarizer = PyannoteDiarizer()
    mock_inference = MagicMock()
    mock_emb = np.ones(128, dtype=np.float32)
    mock_inference.crop.return_value = mock_emb
    mock_inference.return_value = mock_emb
    diarizer._embedding_inference = mock_inference

    # Simulate last diarization result with itertracks
    mock_turn1 = MagicMock()
    mock_turn1.start = 0.0
    mock_turn1.end = 2.0

    mock_turn2 = MagicMock()
    mock_turn2.start = 2.5
    mock_turn2.end = 5.0

    mock_annotation = MagicMock()
    mock_annotation.itertracks.return_value = [
        (mock_turn1, None, "SPEAKER_00"),
        (mock_turn2, None, "SPEAKER_01"),
    ]

    mock_result = MagicMock()
    mock_result.speaker_diarization = mock_annotation
    mock_result.speaker_embeddings = None
    diarizer._last_result = mock_result

    embeddings = diarizer.extract_speaker_embeddings("dummy.wav")
    assert "SPEAKER_00" in embeddings
    assert "SPEAKER_01" in embeddings
    assert embeddings["SPEAKER_00"].shape == (128,)
    assert np.isclose(np.linalg.norm(embeddings["SPEAKER_00"]), 1.0)


def test_diarization_extract_speaker_embeddings_from_cached_numpy_array():
    diarizer = PyannoteDiarizer()

    mock_annotation = MagicMock()
    mock_annotation.labels.return_value = ["SPEAKER_00", "SPEAKER_01"]

    mock_result = MagicMock()
    mock_result.speaker_diarization = mock_annotation
    # Pyannote 3.3+ provides speaker_embeddings as a (num_speakers, dim) numpy array
    mock_result.speaker_embeddings = np.ones((2, 256), dtype=np.float32)
    diarizer._last_result = mock_result

    embeddings = diarizer.extract_speaker_embeddings("dummy.wav")
    assert "SPEAKER_00" in embeddings
    assert "SPEAKER_01" in embeddings
    assert embeddings["SPEAKER_00"].shape == (256,)
    assert embeddings["SPEAKER_01"].shape == (256,)
    assert np.isclose(np.linalg.norm(embeddings["SPEAKER_00"]), 1.0)
