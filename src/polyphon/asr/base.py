"""ASR Backend Interface and Implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from polyphon.types import WordToken


class ASRBackend(ABC):
    """Abstract interface for speech recognition engines."""

    @abstractmethod
    def transcribe(
        self,
        audio_path: str,
        progress_callback: Callable[[float, float], None] | None = None,
        language: str | None = None,
    ) -> tuple[str, list[WordToken]]:
        """Transcribe audio file and return detected language and word tokens with timestamps."""


class FasterWhisperASR(ASRBackend):
    """ASR implementation using faster-whisper with native word-level cross attention."""

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "auto",
        compute_type: str = "auto",
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel

                self._model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                )
            except ImportError as err:
                raise ImportError(
                    'faster-whisper is not installed. Install with `pip install "polyphon-ai[asr]"` or `uv sync --all-extras`'
                ) from err
        return self._model

    def transcribe(
        self,
        audio_path: str,
        progress_callback: Callable[[float, float], None] | None = None,
        language: str | None = None,
    ) -> tuple[str, list[WordToken]]:
        model = self._get_model()
        transcribe_kwargs: dict[str, Any] = {
            "word_timestamps": True,
            "vad_filter": True,
            "condition_on_previous_text": False,
        }
        if language:
            transcribe_kwargs["language"] = language

        segments, info = model.transcribe(
            audio_path,
            **transcribe_kwargs,
        )

        words: list[WordToken] = []
        for segment in segments:
            if progress_callback and info.duration > 0:
                progress_callback(segment.end, info.duration)

            if segment.words:
                for w in segment.words:
                    clean_word = w.word.strip()
                    if clean_word:
                        from polyphon.fillers import is_filler_word

                        words.append(
                            WordToken(
                                word=clean_word,
                                start=w.start,
                                end=w.end,
                                score=w.probability,
                                is_filler=is_filler_word(clean_word),
                            )
                        )

        if progress_callback and info.duration > 0:
            progress_callback(info.duration, info.duration)

        return info.language, words
