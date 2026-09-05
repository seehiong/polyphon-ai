"""PolyphonEngine: Unified local orchestration engine."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from polyphon.asr.base import ASRBackend, FasterWhisperASR
from polyphon.audio import get_audio_duration
from polyphon.diarization.base import DiarizationBackend, PyannoteDiarizer, SortformerDiarizer
from polyphon.llm.base import LLMBackend, OpenAICompatibleLLM
from polyphon.reconciliation.base import ReconciliationEngine
from polyphon.types import Insights, PolyphonResult, SpeakerInfo
from polyphon.voicedb.base import VoiceDB

console = Console()


class PolyphonEngine:
    """Unified engine for transcribe, diarize, identify, and structure."""

    def __init__(
        self,
        asr: ASRBackend | str | None = None,
        diarizer: DiarizationBackend | str | None = None,
        voicedb: VoiceDB | None = None,
        llm: LLMBackend | None = None,
        reconciler: ReconciliationEngine | None = None,
        asr_model: str | None = None,
        device: str = "auto",
    ):
        if isinstance(asr, str):
            self.asr = FasterWhisperASR(model_size=asr, device=device)
        elif asr is not None:
            self.asr = asr
        elif asr_model is not None:
            self.asr = FasterWhisperASR(model_size=asr_model, device=device)
        else:
            self.asr = FasterWhisperASR(device=device)

        if isinstance(diarizer, str):
            if "sortformer" in diarizer.lower():
                model_name = diarizer if diarizer.startswith("diar_") else "diar_sortformer_4spk-v1"
                self.diarizer = SortformerDiarizer(model_name=model_name)
            else:
                model_name = diarizer if "/" in diarizer else "pyannote/speaker-diarization-3.1"
                self.diarizer = PyannoteDiarizer(model_name=model_name)
        else:
            self.diarizer = diarizer or PyannoteDiarizer()

        self.voicedb = voicedb or VoiceDB()
        self.llm = llm or OpenAICompatibleLLM()
        self.reconciler = reconciler or ReconciliationEngine()

    def process(
        self,
        audio_path: str | Path,
        diarize: bool = True,
        identify_speakers: bool = True,
        speaker_map: dict[str, str] | None = None,
        summarize: bool = False,
        infer_names: bool = True,
        auto_enroll: bool = False,
        language: str | None = "en",
        asr_progress_callback: Callable[[float, float], None] | None = None,
        show_diarize_progress: bool = True,
        verbose: bool = True,
        clean_fillers: bool = False,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> PolyphonResult:
        """Run the full end-to-end pipeline on an audio/video file."""
        audio_str = str(audio_path)
        duration = get_audio_duration(audio_str)

        # 1. Automatic Speech Recognition (ASR) - Isolated progress context
        lang_arg = None if language in ("auto", "", None) else language
        if verbose and asr_progress_callback is None:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold cyan][1/4][/bold cyan] 🎙️  Transcribing audio (Whisper large-v3)"),
                BarColumn(bar_width=28),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console,
                transient=False,
            ) as asr_progress:
                asr_task = asr_progress.add_task("Transcribing", total=int(duration))

                def default_cb(current: float, total: float):
                    asr_progress.update(asr_task, completed=min(current, total))

                language, words = self.asr.transcribe(audio_str, progress_callback=default_cb, language=lang_arg)
        else:
            language, words = self.asr.transcribe(audio_str, progress_callback=asr_progress_callback, language=lang_arg)

        # 2. Speaker Diarization - Isolated progress context
        speaker_map_dict: dict[str, str] = {}
        speaker_infos: list[SpeakerInfo] = []

        # Normalize any provided speaker_map keys (e.g. '0', '00', 'SPEAKER_00')
        explicit_speaker_map: dict[str, str] = {}
        if speaker_map:
            for k, v in speaker_map.items():
                clean_k = str(k).strip()
                if not clean_k.startswith("SPEAKER_"):
                    if clean_k.isdigit():
                        clean_k = f"SPEAKER_{clean_k.zfill(2)}"
                    else:
                        clean_k = f"SPEAKER_{clean_k}"
                explicit_speaker_map[clean_k] = str(v).strip()

        if diarize:
            intervals = self.diarizer.diarize(
                audio_str,
                show_progress=(verbose and show_diarize_progress),
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
            detected_speakers = sorted({i.speaker_id for i in intervals})
            extracted_embeddings: dict[str, np.ndarray] | None = None

            # 3. Speaker Identification (VoiceDB & explicit mapping)
            if (identify_speakers or auto_enroll) and self.voicedb:
                extracted_embeddings = self.diarizer.extract_speaker_embeddings(audio_str)

            for spk_id in detected_speakers:
                # 3a. Explicit mapping takes top priority
                if spk_id in explicit_speaker_map:
                    assigned_name = explicit_speaker_map[spk_id]
                    speaker_map_dict[spk_id] = assigned_name
                    speaker_infos.append(SpeakerInfo(id=spk_id, name=assigned_name, confidence=1.0))
                    continue

                # 3b. VoiceDB identification
                if identify_speakers and self.voicedb and extracted_embeddings:
                    emb = extracted_embeddings.get(spk_id)
                    if emb is not None:
                        matched_name, conf = self.voicedb.identify(emb)
                        if matched_name:
                            speaker_map_dict[spk_id] = matched_name
                            speaker_infos.append(SpeakerInfo(id=spk_id, name=matched_name, confidence=conf))
                            continue

                # 3c. Fallback to display name
                disp_name = spk_id.replace("SPEAKER_", "Speaker ")
                speaker_map_dict[spk_id] = disp_name
                speaker_infos.append(SpeakerInfo(id=spk_id, name=disp_name, confidence=1.0))

            # Auto-enroll explicitly mapped speakers if auto_enroll is enabled
            if auto_enroll and self.voicedb and extracted_embeddings:
                for spk_id, assigned_name in explicit_speaker_map.items():
                    if spk_id in detected_speakers:
                        emb = extracted_embeddings.get(spk_id)
                        if emb is not None:
                            spk_slug = self.voicedb.enroll(name=assigned_name, embedding=emb, audio_path=audio_str)
                            if verbose:
                                console.print(
                                    f"[bold green]✔ Enrolled assigned speaker into VoiceDB:[/bold green] "
                                    f"[cyan]{assigned_name}[/cyan] ([yellow]{spk_slug}[/yellow])"
                                )
        else:
            intervals = []
            extracted_embeddings = None
            default_name = explicit_speaker_map.get("SPEAKER_00", "Speaker")
            speaker_map_dict["SPEAKER_00"] = default_name
            speaker_infos.append(SpeakerInfo(id="SPEAKER_00", name=default_name, confidence=1.0))

        # 4. Reconciliation & Alignment
        if verbose:
            console.print("[bold cyan][3/4][/bold cyan] 🔄 Reconciling word timestamps with speaker turns...")
        segments = self.reconciler.reconcile(words, intervals, speaker_map_dict)

        # 4b. In-Context Verbal Speaker Name Inference & Auto-Enrollment
        if infer_names:
            from polyphon.nlp.speaker_inference import SpeakerNameInferer

            inferred = SpeakerNameInferer.infer_names(segments, llm=self.llm if summarize else None)
            newly_inferred: dict[str, str] = {}
            for spk_id, inf_name in inferred.items():
                current_name = speaker_map_dict.get(spk_id, "")
                # Only replace generic placeholder names (e.g. 'Speaker 00', 'Speaker')
                if current_name.startswith("Speaker") or current_name == spk_id:
                    speaker_map_dict[spk_id] = inf_name
                    newly_inferred[spk_id] = inf_name
                    for spk_info in speaker_infos:
                        if spk_info.id == spk_id:
                            spk_info.name = inf_name
                            spk_info.confidence = 0.9

            if newly_inferred:
                for seg in segments:
                    if seg.speaker_id in newly_inferred:
                        seg.speaker_name = newly_inferred[seg.speaker_id]

                if verbose:
                    names_str = ", ".join(f"{k} → [bold cyan]{v}[/bold cyan]" for k, v in newly_inferred.items())
                    console.print(f"[bold green]💬 Inferred speaker identity from dialogue:[/bold green] {names_str}")

                # Auto-enroll newly inferred speakers into VoiceDB if enabled
                if auto_enroll and self.voicedb and diarize:
                    if extracted_embeddings is None:
                        extracted_embeddings = self.diarizer.extract_speaker_embeddings(audio_str)
                    for spk_id, inf_name in newly_inferred.items():
                        emb = extracted_embeddings.get(spk_id)
                        if emb is not None:
                            spk_slug = self.voicedb.enroll(name=inf_name, embedding=emb, audio_path=audio_str)
                            if verbose:
                                console.print(
                                    f"[bold green]✔ Auto-enrolled into VoiceDB:[/bold green] "
                                    f"[cyan]{inf_name}[/cyan] ([yellow]{spk_slug}[/yellow])"
                                )

        if clean_fillers:
            for seg in segments:
                seg.text = seg.get_clean_text()

        # 5. Semantic / LLM Insights
        insights: Insights | None = None
        if summarize and self.llm:
            temp_result = PolyphonResult(
                duration=duration,
                language=language,
                speakers=speaker_infos,
                segments=segments,
            )
            md = temp_result.to_markdown(clean_fillers=clean_fillers)
            if verbose:
                with console.status(
                    "[bold cyan][4/4][/bold cyan] 🧠 Querying local LLM for executive summary & action items...",
                    spinner="dots",
                ):
                    insights = self.llm.extract_insights(md)
                console.print("[bold cyan][4/4][/bold cyan] 🧠 Executive summary & action items generated")
            else:
                insights = self.llm.extract_insights(md)

            insights.speaker_stats = self.reconciler.compute_speaker_stats(segments, duration)

        return PolyphonResult(
            duration=duration,
            language=language,
            speakers=speaker_infos,
            segments=segments,
            insights=insights,
        )
