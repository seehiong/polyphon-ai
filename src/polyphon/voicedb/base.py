"""VoiceDB: Persistent local speaker identity and enrollment database."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np


class VoiceDB:
    """Persistent local database for speaker voice embeddings and identity matching."""

    def __init__(self, db_path: str | Path = "~/.polyphon/voicedb"):
        self.db_dir = Path(db_path).expanduser()
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.db_dir / "registry.json"
        self.embeddings_dir = self.db_dir / "embeddings"
        self.embeddings_dir.mkdir(exist_ok=True)
        self._embedding_cache: dict[str, np.ndarray] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        self._embedding_cache.clear()
        if self.registry_path.exists():
            try:
                with open(self.registry_path, encoding="utf-8") as f:
                    self.registry: dict[str, dict] = json.load(f)
            except (json.JSONDecodeError, OSError):
                self.registry = {}
        else:
            self.registry = {}

        # Preload embeddings into memory cache
        for spk_id, meta in self.registry.items():
            emb_file = Path(meta.get("embedding_file", ""))
            if emb_file.exists():
                try:
                    vec = np.load(emb_file)
                    norm = np.linalg.norm(vec)
                    self._embedding_cache[spk_id] = (vec / norm) if norm > 0 else vec
                except (OSError, ValueError):
                    pass

    def _save_registry(self) -> None:
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self.registry, f, indent=2)

    def enroll(
        self,
        name: str,
        embedding: np.ndarray,
        speaker_id: str | None = None,
        audio_path: str | None = None,
    ) -> str:
        """Enroll a new speaker or update an existing profile with an embedding."""
        spk_id = speaker_id or name.lower().replace(" ", "_")
        embedding_file = self.embeddings_dir / f"{spk_id}.npy"

        # Normalize embedding vector
        norm = np.linalg.norm(embedding)
        norm_emb = (embedding / norm) if norm > 0 else embedding

        if embedding_file.exists():
            try:
                existing = np.load(embedding_file)
                # Running average of centroids
                combined = 0.7 * existing + 0.3 * norm_emb
                combined_norm = np.linalg.norm(combined)
                norm_emb = combined / combined_norm if combined_norm > 0 else combined
            except (OSError, ValueError):
                pass

        np.save(embedding_file, norm_emb)
        self._embedding_cache[spk_id] = norm_emb.copy()

        self.registry[spk_id] = {
            "name": name,
            "id": spk_id,
            "embedding_file": str(embedding_file),
            "sample_audio": str(audio_path) if audio_path else None,
        }
        self._save_registry()
        return spk_id

    def get_speaker(self, name_or_id: str) -> dict | None:
        """Retrieve speaker metadata by ID or display name."""
        target = name_or_id.strip().lower()
        for k, v in self.registry.items():
            if k.lower() == target or v.get("name", "").strip().lower() == target:
                return v
        return None

    def delete_speaker(self, name_or_id: str) -> bool:
        """Delete an enrolled speaker profile and remove their embedding file."""
        spk = self.get_speaker(name_or_id)
        if not spk:
            return False
        spk_id = spk["id"]
        emb_file = Path(spk.get("embedding_file", ""))
        if emb_file.exists():
            try:
                emb_file.unlink()
            except OSError:
                pass
        self.registry.pop(spk_id, None)
        self._embedding_cache.pop(spk_id, None)
        self._save_registry()
        return True

    def list_speakers(self) -> list[dict[str, str]]:
        """List all enrolled speakers in the database."""
        return [
            {
                "id": k,
                "name": v["name"],
                "sample_audio": v.get("sample_audio"),
            }
            for k, v in self.registry.items()
        ]

    def identify(self, query_embedding: np.ndarray, threshold: float = 0.65) -> tuple[str | None, float]:
        """Match a query embedding against the enrolled voiceprints using vectorized cosine similarity.

        Returns:
            Tuple of (speaker_name, similarity_score). If no match passes threshold, returns (None, score).
        """
        if not self.registry:
            return None, 0.0

        q_norm = np.linalg.norm(query_embedding)
        if q_norm == 0:
            return None, 0.0
        q_vec = query_embedding / q_norm

        # Ensure cache is up to date if files exist
        if not self._embedding_cache and self.registry:
            for spk_id, meta in self.registry.items():
                emb_file = Path(meta.get("embedding_file", ""))
                if emb_file.exists():
                    try:
                        vec = np.load(emb_file)
                        norm = np.linalg.norm(vec)
                        self._embedding_cache[spk_id] = (vec / norm) if norm > 0 else vec
                    except (OSError, ValueError):
                        pass

        if not self._embedding_cache:
            return None, 0.0

        # Enrolments made with a different embedding model have a different
        # dimensionality and cannot be compared. Skip them rather than letting
        # np.dot raise a shape error that says nothing about the cause.
        dim = q_vec.shape[-1]
        spk_ids = [sid for sid, vec in self._embedding_cache.items() if vec.shape[-1] == dim]
        skipped = len(self._embedding_cache) - len(spk_ids)
        if skipped:
            warnings.warn(
                f"Ignoring {skipped} enrolled voiceprint(s) whose embedding size does not match "
                f"the current model ({dim}). Re-enrol them to make them matchable again.",
                RuntimeWarning,
                stacklevel=2,
            )
        if not spk_ids:
            return None, 0.0

        emb_matrix = np.stack([self._embedding_cache[sid] for sid in spk_ids])
        similarities = np.dot(emb_matrix, q_vec)

        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])
        best_spk_id = spk_ids[best_idx]
        best_name = self.registry.get(best_spk_id, {}).get("name")

        if best_sim >= threshold and best_name:
            return best_name, best_sim

        return None, max(0.0, best_sim)
