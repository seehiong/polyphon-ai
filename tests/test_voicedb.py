import tempfile

import numpy as np
import pytest

from polyphon.voicedb.base import VoiceDB


def test_voicedb_enroll_and_identify():
    with tempfile.TemporaryDirectory() as tmpdir:
        vdb = VoiceDB(db_path=tmpdir)

        # Create two distinct 64-dim dummy embeddings
        np.random.seed(42)
        alice_vec = np.random.randn(64)
        bob_vec = np.random.randn(64)

        # Enroll Alice and Bob
        vdb.enroll("Alice", alice_vec)
        vdb.enroll("Bob", bob_vec)

        speakers = vdb.list_speakers()
        assert len(speakers) == 2
        names = {s["name"] for s in speakers}
        assert "Alice" in names
        assert "Bob" in names

        # Query with slightly perturbed Alice vector
        query_alice = alice_vec + 0.05 * np.random.randn(64)
        matched_name, score = vdb.identify(query_alice, threshold=0.7)
        assert matched_name == "Alice"
        assert score > 0.8

        # Query with arbitrary noise should fail to match high threshold
        noise_vec = np.random.randn(64)
        matched_name, score = vdb.identify(noise_vec, threshold=0.95)
        assert matched_name is None

        # Test get_speaker
        assert vdb.get_speaker("Alice") is not None
        assert vdb.get_speaker("alice") is not None
        assert vdb.get_speaker("unknown") is None

        # Test delete_speaker
        assert vdb.delete_speaker("Alice") is True
        assert len(vdb.list_speakers()) == 1
        assert vdb.get_speaker("Alice") is None
        assert vdb.delete_speaker("NonExistent") is False


def test_identify_skips_mismatched_embedding_sizes():
    """A voiceprint enrolled with a different model must not crash identify()."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vdb = VoiceDB(db_path=tmpdir)

        np.random.seed(7)
        legacy_vec = np.random.randn(128).astype(np.float32)
        current_vec = np.random.randn(256).astype(np.float32)

        vdb.enroll(name="Legacy Speaker", embedding=legacy_vec)
        vdb.enroll(name="Current Speaker", embedding=current_vec)

        # Querying with the current model's size must ignore the 128-dim entry
        # rather than raising "shapes (1,128) and (256,) not aligned".
        with pytest.warns(RuntimeWarning, match="embedding size"):
            name, score = vdb.identify(current_vec)

        assert name == "Current Speaker"
        assert score > 0.99


def test_identify_returns_no_match_when_all_sizes_mismatch():
    with tempfile.TemporaryDirectory() as tmpdir:
        vdb = VoiceDB(db_path=tmpdir)
        vdb.enroll(name="Legacy Speaker", embedding=np.random.randn(128).astype(np.float32))

        with pytest.warns(RuntimeWarning, match="embedding size"):
            name, score = vdb.identify(np.random.randn(256).astype(np.float32))

        assert name is None
        assert score == 0.0
