"""Batch processing engine for directory-wide transcription and intelligence extraction."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from polyphon.engine import PolyphonEngine
from polyphon.types import PolyphonResult

console = Console()

SUPPORTED_EXTENSIONS = {
    ".mp4",
    ".wav",
    ".m4a",
    ".mp3",
    ".mkv",
    ".flac",
    ".aac",
    ".webm",
    ".ogg",
}


@dataclass
class BatchFileSummary:
    """Summary record for an individual file in a batch run."""

    filename: str
    duration_seconds: float
    language: str
    speaker_count: int
    speakers: list[str]
    summary: str
    decision_count: int
    action_item_count: int
    output_files: dict[str, str]


@dataclass
class BatchResult:
    """Overall result container for a batch run."""

    total_files: int
    total_duration_seconds: float
    successful_files: int
    failed_files: int
    items: list[BatchFileSummary]

    def to_dict(self) -> dict:
        return asdict(self)

    def to_markdown(self) -> str:
        m, s = divmod(int(self.total_duration_seconds), 60)
        dur_str = f"{m}m {s:02d}s" if m > 0 else f"{s:.1f}s"

        lines = [
            "# Polyphon Batch Intelligence Index",
            "",
            f"- **Total Files Processed:** {self.total_files} ({self.successful_files} succeeded, {self.failed_files} failed)",
            f"- **Total Audio Duration:** {dur_str}",
            "",
            "## Meeting Summaries",
            "",
            "| File | Duration | Speakers | Summary | Actions |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]

        for item in self.items:
            m_i, s_i = divmod(int(item.duration_seconds), 60)
            d_str = f"{m_i}m {s_i:02d}s"
            spk_str = ", ".join(item.speakers) if item.speakers else f"{item.speaker_count}"
            short_sum = (item.summary[:90] + "...") if len(item.summary) > 90 else (item.summary or "N/A")
            # Replace pipe characters to preserve markdown table formatting
            clean_sum = short_sum.replace("|", "/")
            lines.append(f"| `{item.filename}` | {d_str} | {spk_str} | {clean_sum} | {item.action_item_count} items |")

        lines.append("")
        return "\n".join(lines)


class BatchProcessor:
    """Discovers media files in directories and orchestrates sequential processing."""

    def __init__(self, engine: PolyphonEngine | None = None):
        self.engine = engine or PolyphonEngine()

    def discover_files(self, input_dir: str | Path, recursive: bool = False) -> list[Path]:
        """Scan input directory for supported media files."""
        dir_path = Path(input_dir).expanduser().resolve()
        if not dir_path.is_dir():
            raise ValueError(f"Input path is not a valid directory: {dir_path}")

        pattern = "**/*" if recursive else "*"
        files = [p for p in dir_path.glob(pattern) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
        return sorted(files, key=lambda p: p.name.lower())

    def process_directory(
        self,
        input_dir: str | Path,
        output_dir: str | Path | None = None,
        diarize: bool = True,
        identify: bool = True,
        summarize: bool = False,
        infer_names: bool = True,
        auto_enroll: bool = False,
        formats: list[str] | None = None,
        recursive: bool = False,
        clean_fillers: bool = False,
    ) -> BatchResult:
        """Run batch transcription and intelligence extraction across discovered files."""
        media_files = self.discover_files(input_dir, recursive=recursive)
        if not media_files:
            console.print(f"[yellow]No supported media files found in {input_dir}[/yellow]")
            return BatchResult(
                total_files=0,
                total_duration_seconds=0.0,
                successful_files=0,
                failed_files=0,
                items=[],
            )

        if output_dir:
            out_path = Path(output_dir).expanduser().resolve()
        else:
            default_env = os.environ.get("POLYPHON_OUTPUT_DIR")
            out_path = (
                Path(default_env).expanduser().resolve() if default_env else Path(input_dir).expanduser().resolve()
            )
        out_path.mkdir(parents=True, exist_ok=True)

        selected_formats = formats or ["markdown"]
        if "all" in selected_formats:
            selected_formats = ["markdown", "json", "srt", "html"]

        console.print()
        console.rule(
            f"[bold cyan]Polyphon Batch[/bold cyan] | Found [bold yellow]{len(media_files)}[/bold yellow] files in [cyan]{input_dir}[/cyan]"
        )
        console.print()

        items: list[BatchFileSummary] = []
        total_duration = 0.0
        success_count = 0
        failed_count = 0

        with Progress(
            TextColumn("[bold green]Batch Queue[/bold green] [bold cyan]({task.completed}/{task.total})[/bold cyan]"),
            BarColumn(bar_width=30),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        ) as batch_progress:
            batch_task = batch_progress.add_task("Batch progress", total=len(media_files))

            for idx, media_file in enumerate(media_files, start=1):
                console.print(
                    f"\n[bold magenta]━━━ [{idx}/{len(media_files)}] Processing: {media_file.name} ━━━[/bold magenta]"
                )
                try:
                    res: PolyphonResult = self.engine.process(
                        audio_path=media_file,
                        diarize=diarize,
                        identify_speakers=identify,
                        summarize=summarize,
                        infer_names=infer_names,
                        auto_enroll=auto_enroll,
                        show_diarize_progress=diarize,
                        verbose=True,
                        clean_fillers=clean_fillers,
                    )

                    total_duration += res.duration
                    success_count += 1

                    output_files: dict[str, str] = {}
                    base_name = media_file.stem

                    if "markdown" in selected_formats:
                        f_path = out_path / f"{base_name}.md"
                        f_path.write_text(res.to_markdown(clean_fillers=clean_fillers), encoding="utf-8")
                        output_files["markdown"] = str(f_path)

                    if "json" in selected_formats:
                        f_path = out_path / f"{base_name}.json"
                        f_path.write_text(res.to_json(), encoding="utf-8")
                        output_files["json"] = str(f_path)

                    if "srt" in selected_formats:
                        f_path = out_path / f"{base_name}.srt"
                        f_path.write_text(res.to_srt(clean_fillers=clean_fillers), encoding="utf-8")
                        output_files["srt"] = str(f_path)

                    if "html" in selected_formats:
                        f_path = out_path / f"{base_name}.html"
                        f_path.write_text(
                            res.to_html(title=f"Polyphon Report: {media_file.name}", clean_fillers=clean_fillers),
                            encoding="utf-8",
                        )
                        output_files["html"] = str(f_path)

                    spk_names = [s.name for s in res.speakers] if res.speakers else []
                    summary_text = res.insights.summary if res.insights else ""
                    decisions = res.insights.decisions if res.insights else []
                    actions = res.insights.action_items if res.insights else []

                    items.append(
                        BatchFileSummary(
                            filename=media_file.name,
                            duration_seconds=res.duration,
                            language=res.language,
                            speaker_count=len(res.speakers),
                            speakers=spk_names,
                            summary=summary_text,
                            decision_count=len(decisions),
                            action_item_count=len(actions),
                            output_files=output_files,
                        )
                    )

                    console.print(
                        f"[green]✔ Finished:[/green] {media_file.name} (Saved: {', '.join(output_files.keys())})"
                    )

                except Exception as err:
                    failed_count += 1
                    console.print(f"[red]✖ Error processing {media_file.name}: {err}[/red]")

                batch_progress.advance(batch_task)

        batch_result = BatchResult(
            total_files=len(media_files),
            total_duration_seconds=total_duration,
            successful_files=success_count,
            failed_files=failed_count,
            items=items,
        )

        # Write aggregate index files
        index_json_path = out_path / "batch_index.json"
        index_md_path = out_path / "batch_index.md"

        index_json_path.write_text(json.dumps(batch_result.to_dict(), indent=2), encoding="utf-8")
        index_md_path.write_text(batch_result.to_markdown(), encoding="utf-8")

        console.print()
        console.print(
            f"[bold green]✔ Batch Complete![/bold green] Processed {success_count}/{len(media_files)} files successfully."
        )
        console.print(f"  📄 Index: [bold cyan]{index_md_path}[/bold cyan]")
        console.print(f"  📊 Manifest: [bold cyan]{index_json_path}[/bold cyan]\n")

        return batch_result
