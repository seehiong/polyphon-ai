"""VoiceDB: Persistent local speaker identity and enrollment database."""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
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
        source: str | None = None,
        replace: bool = False,
    ) -> str:
        """Enroll a new speaker or update an existing profile with an embedding.

        Each call that supplies `audio_path` appends a new entry to that speaker's
        `samples` list (with `source`, e.g. a meeting name, and a timestamp) rather
        than overwriting the previous one, so re-enrolling the same person from a
        different meeting keeps every sample browsable instead of discarding it.

        By default a re-enrollment blends into the existing embedding (0.7 old / 0.3
        new), which only partially corrects a prior enrollment that captured the
        wrong voice — it takes several rounds to dilute out bad data. Pass
        `replace=True` to discard the existing embedding and sample history outright
        and start clean from this enrollment instead.
        """
        spk_id = speaker_id or name.lower().replace(" ", "_")
        embedding_file = self.embeddings_dir / f"{spk_id}.npy"

        # Normalize embedding vector
        norm = np.linalg.norm(embedding)
        norm_emb = (embedding / norm) if norm > 0 else embedding

        if not replace and embedding_file.exists():
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

        existing_entry = {} if replace else self.registry.get(spk_id, {})
        samples = list(existing_entry.get("samples", []))
        if not samples and existing_entry.get("sample_audio"):
            # Migrate a pre-existing single sample_audio into the samples list.
            samples.append({"path": existing_entry["sample_audio"]})
        if audio_path:
            sample_entry: dict[str, str] = {
                "path": str(audio_path),
                "added_at": datetime.now(timezone.utc).isoformat(),
            }
            if source:
                sample_entry["source"] = source
            samples.append(sample_entry)

        self.registry[spk_id] = {
            "name": name,
            "id": spk_id,
            "embedding_file": str(embedding_file),
            "sample_audio": str(audio_path) if audio_path else existing_entry.get("sample_audio"),
            "samples": samples,
        }
        self._save_registry()
        return spk_id

    def set_embedding(self, speaker_id: str, embedding: np.ndarray) -> bool:
        """Directly overwrite a speaker's stored embedding (no blending)."""
        if speaker_id not in self.registry:
            return False
        norm = np.linalg.norm(embedding)
        norm_emb = (embedding / norm) if norm > 0 else embedding
        embedding_file = self.embeddings_dir / f"{speaker_id}.npy"
        np.save(embedding_file, norm_emb)
        self._embedding_cache[speaker_id] = norm_emb.copy()
        return True

    def delete_sample(self, speaker_id: str, sample_index: int) -> dict | None:
        """Remove one enrollment sample from a speaker's history.

        Returns the removed sample entry (so the caller can delete its audio file
        and/or recompute the embedding from whatever samples remain), or None if
        the speaker/index doesn't exist.
        """
        entry = self.registry.get(speaker_id)
        if not entry:
            return None
        samples = list(entry.get("samples", []))
        if not samples and entry.get("sample_audio"):
            samples = [{"path": entry["sample_audio"]}]
        if sample_index < 0 or sample_index >= len(samples):
            return None

        removed = samples.pop(sample_index)
        entry["samples"] = samples
        entry["sample_audio"] = samples[-1]["path"] if samples else None
        self.registry[speaker_id] = entry
        self._save_registry()
        return removed

    def get_speaker(self, name_or_id: str) -> dict | None:
        """Retrieve speaker metadata by ID or display name."""
        target = name_or_id.strip().lower()
        for k, v in self.registry.items():
            if k.lower() == target or v.get("name", "").strip().lower() == target:
                return v
        return None

    def rename_speaker(self, speaker_id: str, new_name: str) -> bool:
        """Update an enrolled speaker's display name, keeping their id and voiceprint unchanged."""
        if speaker_id not in self.registry:
            return False
        self.registry[speaker_id]["name"] = new_name
        self._save_registry()
        return True

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

    def list_speakers(self) -> list[dict]:
        """List all enrolled speakers in the database, with every enrollment sample."""
        result = []
        for k, v in self.registry.items():
            samples = v.get("samples")
            if not samples and v.get("sample_audio"):
                # Older entries enrolled before multi-sample tracking existed.
                samples = [{"path": v["sample_audio"]}]
            result.append(
                {
                    "id": k,
                    "name": v["name"],
                    "sample_audio": v.get("sample_audio"),
                    "samples": samples or [],
                }
            )
        return result

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
