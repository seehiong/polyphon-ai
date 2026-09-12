"""Command-line interface for Polyphon."""

from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from polyphon.config import load_env

load_env()

# Suppress known benign third-party library warnings during CLI runs
warnings.filterwarnings("ignore", message=".*degrees of freedom is <= 0.*", category=UserWarning)

app = typer.Typer(
    name="polyphon",
    help="Local, identity-aware transcription and diarization engine.",
    add_completion=False,
)
speakers_app = typer.Typer(help="Manage enrolled speakers in VoiceDB.")
app.add_typer(speakers_app, name="speakers")

console = Console()


@app.command()
def transcribe(
    audio_path: Path = typer.Argument(..., help="Path to audio or video file"),
    diarize: bool = typer.Option(True, "--diarize/--no-diarize", help="Enable speaker diarization"),
    identify: bool = typer.Option(False, "--identify", help="Match speakers against local VoiceDB"),
    infer_names: bool = typer.Option(
        True,
        "--infer-names/--no-infer-names",
        help="Infer speaker names from verbal self-introductions in dialogue",
    ),
    auto_enroll: bool = typer.Option(
        False,
        "--auto-enroll",
        help="Automatically enroll newly inferred speakers into local VoiceDB",
    ),
    summarize: bool = typer.Option(False, "--summarize", help="Generate structured summary with local LLM"),
    language: str = typer.Option(
        "en",
        "--language",
        "-l",
        help="Spoken language code (e.g. 'en', 'ms', 'zh', or 'auto' for auto-detection)",
    ),
    llm_endpoint: str | None = typer.Option(
        None,
        "--llm-endpoint",
        help="OpenAI-compatible LLM endpoint (default: http://localhost:8080/v1)",
    ),
    llm_model: str | None = typer.Option(
        None,
        "--llm-model",
        help="LLM model identifier (default: qwen3.8-27b)",
    ),
    output_format: str = typer.Option("markdown", "--format", "-f", help="Output formats: json, markdown, srt"),
    output_path: Path | None = typer.Option(None, "--output", "-o", help="Custom output file path"),
    speaker_map: list[str] | None = typer.Option(
        None,
        "--speaker-map",
        "-m",
        help="Explicit speaker mappings (e.g. 'SPEAKER_00=Alice,SPEAKER_01=Bob' or repeated -m 'SPEAKER_00=Alice')",
    ),
    clean_fillers: bool = typer.Option(
        False,
        "--clean-fillers",
        help="Remove vocal disfluencies (um, uh, er, ah) for clean verbatim output",
    ),
    num_speakers: int | None = typer.Option(
        None,
        "--num-speakers",
        help="Exact number of speakers to constrain during diarization (e.g. 1 for solo recording)",
    ),
    min_speakers: int | None = typer.Option(
        None,
        "--min-speakers",
        help="Minimum number of speakers to detect during diarization",
    ),
    max_speakers: int | None = typer.Option(
        None,
        "--max-speakers",
        help="Maximum number of speakers to detect during diarization",
    ),
    save: bool = typer.Option(True, "--save/--no-save", help="Automatically save output file to disk"),
):
    """Transcribe and diarize an audio file."""
    if not audio_path.exists():
        console.print(f"[red]Error:[/red] File not found: {audio_path}")
        raise typer.Exit(1)

    from polyphon.assign import parse_speaker_map
    from polyphon.audio import get_audio_duration
    from polyphon.engine import PolyphonEngine
    from polyphon.llm.base import OpenAICompatibleLLM

    duration = get_audio_duration(audio_path)
    m, s = divmod(int(duration), 60)
    dur_str = f"{m}m {s:02d}s" if m > 0 else f"{s:.1f}s"

    console.print()
    console.rule(
        f"[bold cyan]Polyphon[/bold cyan] 🎙️ Processing: [bold yellow]{audio_path.name}[/bold yellow] ({dur_str})"
    )
    console.print()

    llm = OpenAICompatibleLLM(endpoint=llm_endpoint, model=llm_model) if summarize else None
    engine = PolyphonEngine(llm=llm)

    parsed_map = parse_speaker_map(speaker_map)

    result = engine.process(
        audio_path=audio_path,
        diarize=diarize,
        identify_speakers=identify,
        speaker_map=parsed_map,
        summarize=summarize,
        infer_names=infer_names,
        auto_enroll=auto_enroll,
        language=language,
        show_diarize_progress=diarize,
        verbose=True,
        clean_fillers=clean_fillers,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    if output_format == "json":
        out_text = result.to_json()
        ext = ".json"
    elif output_format == "srt":
        out_text = result.to_srt(clean_fillers=clean_fillers)
        ext = ".srt"
    elif output_format == "html":
        out_text = result.to_html(title=f"Polyphon Report: {audio_path.name}", clean_fillers=clean_fillers)
        ext = ".html"
    else:
        out_text = result.to_markdown(clean_fillers=clean_fillers)
        ext = ".md"

    if save:
        if output_path is not None:
            target_path = output_path
        else:
            default_out_dir = os.environ.get("POLYPHON_OUTPUT_DIR", "outputs")
            out_dir = Path(default_out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            target_path = out_dir / f"{audio_path.stem}{ext}"

        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(out_text, encoding="utf-8")
            console.print(f"[green]✔ Saved to:[/green] [bold cyan]{target_path}[/bold cyan]")
        except OSError as err:
            console.print(f"[yellow]Warning: Could not save to {target_path}: {err}[/yellow]")

    console.print("\n" + out_text)


@app.command("enroll")
def enroll_speaker(
    name: str = typer.Option(..., "--name", "-n", help="Full human name of the speaker to enroll"),
    audio_path: Path = typer.Option(..., "--audio", "-a", help="Path to audio file containing speaker's voice"),
    speaker_id: str | None = typer.Option(None, "--id", help="Optional internal speaker ID (e.g. lex_fridman)"),
    speaker_cluster: str | None = typer.Option(
        None, "--speaker", "-s", help="Cluster ID from meeting (e.g. SPEAKER_00)"
    ),
    start: float | None = typer.Option(None, "--start", help="Start timestamp in seconds to extract from"),
    end: float | None = typer.Option(None, "--end", help="End timestamp in seconds to extract from"),
):
    """Enroll a speaker voiceprint into the local VoiceDB."""
    if not audio_path.exists():
        console.print(f"[red]Error:[/red] File not found: {audio_path}")
        raise typer.Exit(1)

    from polyphon.diarization.base import PyannoteDiarizer
    from polyphon.voicedb.base import VoiceDB

    diarizer = PyannoteDiarizer()
    voicedb = VoiceDB()

    with console.status(
        f"Extracting voiceprint embedding for [bold cyan]{name}[/bold cyan]...",
        spinner="dots",
    ):
        if speaker_cluster:
            clusters = diarizer.extract_speaker_embeddings(str(audio_path))
            target_cluster = (
                speaker_cluster if speaker_cluster.startswith("SPEAKER_") else f"SPEAKER_{speaker_cluster.zfill(2)}"
            )
            if target_cluster not in clusters:
                console.print(
                    f"[red]Error:[/red] Cluster {target_cluster} not found in {audio_path.name}. Found: {list(clusters.keys())}"
                )
                raise typer.Exit(1)
            embedding = clusters[target_cluster]
        else:
            embedding = diarizer.extract_single_embedding(str(audio_path), start=start, end=end)

    spk_id = voicedb.enroll(
        name=name,
        embedding=embedding,
        speaker_id=speaker_id,
        audio_path=str(audio_path),
    )
    console.print(
        f"[bold green]✔ Successfully enrolled:[/bold green] [cyan]{name}[/cyan] (ID: [yellow]{spk_id}[/yellow]) into VoiceDB!"
    )


@app.command("assign")
def assign_meeting_speakers(
    meeting: str = typer.Argument(
        ...,
        help="Path to meeting manifest JSON (e.g. outputs/meeting.json) or meeting ID (stem)",
    ),
    speaker_map: list[str] | None = typer.Option(
        None,
        "--map",
        "-m",
        help="Speaker mapping (e.g. --map SPEAKER_00='Marcus Vance' or '0=Marcus,1=David')",
    ),
    mapping_args: list[str] = typer.Argument(
        None,
        help="Optional positional speaker mappings (e.g. SPEAKER_00='Marcus Vance' SPEAKER_01='David')",
    ),
    enroll: bool = typer.Option(
        True,
        "--enroll/--no-enroll",
        help="Extract voice embeddings and register mapped speakers into VoiceDB",
    ),
    summarize: bool = typer.Option(
        False,
        "--summarize/--no-summarize",
        help="Re-run local LLM summarization with updated speaker names",
    ),
    save: bool = typer.Option(
        True,
        "--save/--no-save",
        help="Save updated JSON, HTML, Markdown, and SRT reports",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Directory where meeting outputs are stored",
    ),
    llm_endpoint: str | None = typer.Option(
        None,
        "--llm-endpoint",
        help="OpenAI-compatible LLM endpoint (default: http://localhost:8080/v1)",
    ),
    llm_model: str | None = typer.Option(
        None,
        "--llm-model",
        help="LLM model identifier (default: qwen3.8-27b)",
    ),
):
    """Assign real speaker identities to an existing meeting, enroll them in VoiceDB, and update reports."""
    from polyphon.assign import assign_speakers_to_manifest, parse_speaker_map
    from polyphon.llm.base import OpenAICompatibleLLM

    combined = (speaker_map or []) + (mapping_args or [])
    clean_map = parse_speaker_map(combined)
    if not clean_map:
        console.print(
            "[red]Error:[/red] No valid speaker mappings provided. Use [bold cyan]--map SPEAKER_00='Name'[/bold cyan] or positional arguments."
        )
        raise typer.Exit(1)

    llm = OpenAICompatibleLLM(endpoint=llm_endpoint, model=llm_model) if summarize else None

    console.print()
    console.rule(f"[bold cyan]Polyphon[/bold cyan] 👥 Assigning Speakers: [bold yellow]{meeting}[/bold yellow]")
    table = Table(title="Speaker Mappings to Apply", box=None)
    table.add_column("Cluster ID", style="yellow")
    table.add_column("Assigned Identity", style="bold green")
    for k, v in clean_map.items():
        table.add_row(k, v)
    console.print(table)
    console.print()

    try:
        manifest_data, enrolled = assign_speakers_to_manifest(
            meeting_path_or_id=meeting,
            speaker_map=clean_map,
            enroll=enroll,
            summarize=summarize,
            llm=llm,
            save=save,
            output_dir=output_dir,
        )
    except FileNotFoundError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        console.print(
            "\n[dim]`assign` edits an existing JSON manifest. Transcribe with "
            "[bold]--format json[/bold] first:[/dim]\n"
            "  [bold cyan]polyphon transcribe <media> --diarize --format json[/bold cyan]\n"
            "[dim]Other formats (markdown, html, srt) are reports, not manifests.[/dim]"
        )
        raise typer.Exit(1) from exc
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    if enrolled:
        for name in enrolled:
            console.print(f"[bold green]✔ Enrolled into VoiceDB:[/bold green] [cyan]{name}[/cyan]")

    if save:
        console.print(
            "[bold green]✔ Successfully updated meeting manifest and regenerated HTML/MD/SRT reports![/bold green]"
        )


@app.command("batch")
def batch_transcribe(
    directory: Path = typer.Argument(..., help="Path to directory containing audio/video files"),
    output_dir: Path | None = typer.Option(None, "--output-dir", "-o", help="Output directory for generated files"),
    diarize: bool = typer.Option(True, "--diarize/--no-diarize", help="Perform speaker diarization"),
    identify: bool = typer.Option(False, "--identify", help="Match speakers against local VoiceDB"),
    infer_names: bool = typer.Option(
        True,
        "--infer-names/--no-infer-names",
        help="Infer speaker names from dialogue introductions",
    ),
    auto_enroll: bool = typer.Option(False, "--auto-enroll", help="Auto-enroll newly inferred speakers into VoiceDB"),
    summarize: bool = typer.Option(False, "--summarize", help="Generate structured summary with local LLM"),
    output_formats: list[str] = typer.Option(
        ["markdown"], "--format", "-f", help="Formats: markdown, json, srt, html, all"
    ),
    clean_fillers: bool = typer.Option(
        False,
        "--clean-fillers",
        help="Remove vocal disfluencies (um, uh, er, ah) for clean verbatim output",
    ),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Scan subdirectories recursively"),
    llm_endpoint: str | None = typer.Option(None, "--llm-endpoint", help="OpenAI-compatible LLM endpoint"),
    llm_model: str | None = typer.Option(None, "--llm-model", help="LLM model identifier"),
):
    """Process an entire directory of audio and video files in batch."""
    from polyphon.batch import BatchProcessor
    from polyphon.engine import PolyphonEngine
    from polyphon.llm.base import OpenAICompatibleLLM

    llm = OpenAICompatibleLLM(endpoint=llm_endpoint, model=llm_model) if summarize else None
    engine = PolyphonEngine(llm=llm)
    processor = BatchProcessor(engine=engine)

    processor.process_directory(
        input_dir=directory,
        output_dir=output_dir,
        diarize=diarize,
        identify=identify,
        summarize=summarize,
        infer_names=infer_names,
        auto_enroll=auto_enroll,
        formats=output_formats,
        recursive=recursive,
        clean_fillers=clean_fillers,
    )


@app.command("stream")
def stream_audio(
    audio_path: Path | None = typer.Argument(None, help="Audio file to stream as simulation, or omit if using --mic"),
    mic: bool = typer.Option(False, "--mic", "-m", help="Stream live from microphone"),
    simulate: bool = typer.Option(False, "--simulate", "-s", help="Simulate real-time streaming from audio file"),
    speed: float = typer.Option(1.0, "--speed", help="Simulation playback speed multiplier"),
    diarizer_type: str = typer.Option(
        "sortformer",
        "--diarizer",
        "-d",
        help="Diarization backend: sortformer (EEND) or pyannote",
    ),
    identify: bool = typer.Option(False, "--identify", help="Match speakers against local VoiceDB"),
    language: str = typer.Option("en", "--language", "-l", help="Spoken language code for ASR (default: 'en')"),
    chunk_size: float = typer.Option(1.5, "--chunk-size", help="Streaming chunk size in seconds"),
    output_dir: Path | None = typer.Option(None, "--output-dir", "-o", help="Output directory for saved stream"),
    save: bool = typer.Option(True, "--save/--no-save", help="Save final transcript and audio to disk"),
    title: str | None = typer.Option(None, "--title", "-t", help="Meeting title or recording label"),
):
    """Stream live audio from microphone or simulate real-time file streaming."""
    if not mic and not audio_path:
        console.print("[red]Error:[/red] Please provide an audio file to stream or specify [cyan]--mic[/cyan].")
        raise typer.Exit(1)

    if audio_path and not audio_path.exists():
        console.print(f"[red]Error:[/red] File not found: {audio_path}")
        raise typer.Exit(1)

    from polyphon.diarization.base import PyannoteDiarizer, SortformerDiarizer
    from polyphon.streaming import (
        StreamingPolyphonEngine,
        capture_mic_stream,
        simulate_file_stream,
    )
    from polyphon.voicedb.base import VoiceDB

    diarizer = SortformerDiarizer() if diarizer_type.lower() == "sortformer" else PyannoteDiarizer()
    voicedb = VoiceDB() if identify else None
    lang_val = None if language in ("auto", "", None) else language
    engine = StreamingPolyphonEngine(diarizer=diarizer, voicedb=voicedb, language=lang_val, chunk_duration=chunk_size)

    console.print()
    if mic:
        console.rule("[bold cyan]Polyphon Stream[/bold cyan] 🎙️ Live Microphone Streaming")
        console.print("[dim]Press Ctrl+C to stop recording and finalize meeting.[/dim]\n")
        generator = capture_mic_stream(chunk_duration=chunk_size)
    else:
        mode_str = f"Real-time Simulation ({speed}x)" if simulate else "File Stream"
        console.rule(
            f"[bold cyan]Polyphon Stream[/bold cyan] 🎧 {mode_str}: [bold yellow]{audio_path.name}[/bold yellow]"
        )
        console.print("[dim]Press Ctrl+C to abort stream early.[/dim]\n")
        generator = simulate_file_stream(audio_path, chunk_duration=chunk_size, speed=speed)

    engine.start_stream()
    last_printed_count = 0

    try:
        for chunk in generator:
            engine.feed_audio(chunk)
            event = engine.process_step()
            if event and event.segments:
                for idx, seg in enumerate(event.segments):
                    if idx >= last_printed_count:
                        color = "green" if (idx % 2 == 0) else "cyan"
                        m, s = divmod(int(seg.start), 60)
                        t_str = f"{m:02d}:{s:02d}"
                        console.print(
                            f"[dim][{t_str}][/dim] [bold {color}]{seg.speaker_name}[/bold {color}]: {seg.text}"
                        )
                        last_printed_count = idx + 1
    except KeyboardInterrupt:
        console.print("\n[yellow]Stream interrupted by user. Finalizing...[/yellow]")

    out_dir = output_dir or Path(os.environ.get("POLYPHON_OUTPUT_DIR", "outputs"))
    stem = audio_path.stem if audio_path else f"mic_stream_{int(time.time())}"
    result = engine.stop_stream(output_dir=out_dir, session_name=stem)
    if title:
        result.title = title
        json_file = out_dir / f"{stem}.json"
        if json_file.exists():
            try:
                import json

                jdata = json.loads(json_file.read_text(encoding="utf-8"))
                jdata["title"] = title
                json_file.write_text(json.dumps(jdata, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass

    console.print()
    console.rule("[bold green]✔ Stream Session Finalized[/bold green]")
    if title:
        console.print(f"[bold]Meeting Title:[/bold] {title}")
    console.print(f"[bold]Total Duration:[/bold] {result.duration:.1f}s")
    console.print(f"[bold]Total Turns:[/bold] {len(result.segments)}")
    if save:
        console.print(f"[green]✔ Saved audio & JSON to:[/green] [cyan]{out_dir / stem}[/cyan]")


@speakers_app.command("list")
def list_speakers():
    """List all enrolled speakers in VoiceDB."""
    from polyphon.voicedb.base import VoiceDB

    db = VoiceDB()
    spks = db.list_speakers()

    if not spks:
        console.print("[yellow]No enrolled speakers found in VoiceDB.[/yellow]")
        return

    table = Table(title="Enrolled Speakers (VoiceDB)")
    table.add_column("Speaker ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Sample Audio", style="dim")

    for s in spks:
        table.add_row(s["id"], s["name"], s.get("sample_audio") or "N/A")

    console.print(table)


@speakers_app.command("remove")
def remove_speaker(
    name: str = typer.Argument(..., help="Name or speaker ID to remove from VoiceDB"),
):
    """Remove an enrolled speaker from VoiceDB."""
    from polyphon.voicedb.base import VoiceDB

    db = VoiceDB()
    if db.delete_speaker(name):
        console.print(f"[green]✔ Removed speaker:[/green] {name}")
    else:
        console.print(f"[red]Speaker not found in VoiceDB:[/red] {name}")
        raise typer.Exit(1)


def _get_or_create_self_signed_cert(ssl_dir: Path | None = None) -> tuple[Path, Path]:
    """Generate self-signed SSL certificates for secure LAN access and microphone support."""
    import subprocess

    target_dir = ssl_dir or Path("~/.polyphon/ssl").expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)
    cert_file = target_dir / "cert.pem"
    key_file = target_dir / "key.pem"

    if not (cert_file.exists() and key_file.exists() and cert_file.stat().st_size > 0 and key_file.stat().st_size > 0):
        # 1. Try pure-Python cryptography generation first if available
        try:
            import datetime
            import ipaddress

            from cryptography import x509
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import rsa
            from cryptography.x509.oid import NameOID

            key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
            )
            name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "PolyphonStudio")])
            alt_names = [
                x509.DNSName("localhost"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
                x509.IPAddress(ipaddress.IPv4Address("0.0.0.0")),
            ]
            now = datetime.datetime.now(datetime.timezone.utc)
            cert = (
                x509.CertificateBuilder()
                .subject_name(name)
                .issuer_name(name)
                .public_key(key.public_key())
                .serial_number(x509.random_serial_number())
                .not_valid_before(now)
                .not_valid_after(now + datetime.timedelta(days=365))
                .add_extension(x509.SubjectAlternativeName(alt_names), critical=False)
                .sign(key, hashes.SHA256())
            )

            key_bytes = key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
            cert_bytes = cert.public_bytes(serialization.Encoding.PEM)

            key_file.write_bytes(key_bytes)
            cert_file.write_bytes(cert_bytes)
            return cert_file, key_file
        except (ImportError, Exception):
            pass

        # 2. Fallback to openssl CLI with a bundled configuration (fixes missing openssl.cnf on Windows)
        cnf_content = """[ req ]
default_bits = 2048
prompt = no
default_md = sha256
distinguished_name = dn
x509_extensions = v3_req

[ dn ]
CN = PolyphonStudio

[ v3_req ]
basicConstraints = CA:FALSE
keyUsage = digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[ alt_names ]
DNS.1 = localhost
IP.1 = 127.0.0.1
IP.2 = 0.0.0.0
"""
        cnf_file = target_dir / "openssl.cnf"
        cnf_file.write_text(cnf_content, encoding="utf-8")

        cmd = [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-keyout",
            str(key_file),
            "-out",
            str(cert_file),
            "-days",
            "365",
            "-nodes",
            "-config",
            str(cnf_file),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except (subprocess.SubprocessError, FileNotFoundError) as err:
            if cert_file.exists():
                cert_file.unlink(missing_ok=True)
            if key_file.exists():
                key_file.unlink(missing_ok=True)
            stderr = getattr(err, "stderr", None) or ""
            err_msg = f"Could not generate self-signed certificate with openssl: {err}"
            if stderr:
                err_msg += f"\n{stderr.strip()}"
            raise RuntimeError(err_msg) from err

    return cert_file, key_file


@app.command("serve")
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Bind host address"),
    port: int = typer.Option(7860, "--port", "-p", help="Server port"),
    output_dir: Path | None = typer.Option(None, "--output-dir", "-o", help="Outputs and archive directory"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development"),
    ssl_certfile: Path | None = typer.Option(None, "--ssl-certfile", help="SSL certificate file for HTTPS"),
    ssl_keyfile: Path | None = typer.Option(None, "--ssl-keyfile", help="SSL private key file for HTTPS"),
    self_signed: bool = typer.Option(
        False,
        "--self-signed",
        "--ssl",
        envvar="POLYPHON_SELF_SIGNED",
        help="Auto-generate and use self-signed SSL for secure LAN HTTPS & microphone access",
    ),
):
    """Launch Polyphon Studio Web Dashboard & REST API server."""
    import uvicorn

    from polyphon.server.app import create_app

    out_path = output_dir or os.environ.get("POLYPHON_OUTPUT_DIR") or "outputs"
    server_app = create_app(output_dir=out_path)

    use_ssl = False
    cert_path = ssl_certfile
    key_path = ssl_keyfile

    if self_signed:
        cert_path, key_path = _get_or_create_self_signed_cert()
        use_ssl = True
    elif ssl_certfile and ssl_keyfile:
        use_ssl = True

    protocol = "https" if use_ssl else "http"

    console.print()
    console.rule("[bold cyan]Polyphon Studio 🎙️ Web Dashboard[/bold cyan]")
    console.print(f"[bold green]🚀 Server running at:[/bold green] [bold cyan]{protocol}://{host}:{port}[/bold cyan]")
    if host == "0.0.0.0":
        console.print(f"[dim]Local access: {protocol}://localhost:{port}[/dim]")
        console.print(f"[dim]Network access: {protocol}://<your-lan-ip>:{port}[/dim]")
    console.print(f"[dim]Archive directory: {out_path}[/dim]")
    if use_ssl:
        console.print("[dim green]🔒 HTTPS enabled (WebRTC microphone capture enabled across LAN)[/dim green]")
        console.print(
            "[dim yellow]ℹ️  On first browser visit, click 'Advanced' → 'Proceed' to accept the self-signed certificate.[/dim yellow]"
        )
    else:
        console.print(
            "[dim yellow]💡 Tip: For LAN microphone capture, run with --self-signed (or set POLYPHON_SELF_SIGNED=1).[/dim yellow]"
        )
    console.print()

    uv_kwargs: dict[str, Any] = {"host": host, "port": port, "reload": reload}
    if use_ssl and cert_path and key_path:
        uv_kwargs["ssl_certfile"] = str(cert_path)
        uv_kwargs["ssl_keyfile"] = str(key_path)

    uvicorn.run(server_app, **uv_kwargs)


@app.command()
def bot(
    meeting_url: str = typer.Argument(..., help="Meeting URL (Google Meet, Jitsi Meet, etc.)"),
    name: str = typer.Option("Polyphon Notetaker", "--name", "-n", help="Display name for the virtual attendee"),
    title: str | None = typer.Option(
        None, "--title", "-t", help="Custom meeting title or recording label (e.g. 'Sprint Review')"
    ),
    session: str | None = typer.Option(None, "--session", help="Custom session filename stem (e.g. 'sprint_review')"),
    duration: int | None = typer.Option(None, "--duration", "-d", help="Max duration in minutes to attend"),
    server: str = typer.Option(
        "ws://localhost:7860/api/stream/ws", "--server", "-s", help="Polyphon streaming WebSocket URL"
    ),
    headed: bool = typer.Option(False, "--headed", help="Run browser with visible UI for debugging or manual login"),
):
    """Launch headless virtual attendee to join online meeting and transcribe in real-time."""
    import asyncio
    import ssl
    import urllib.error
    import urllib.parse
    import urllib.request

    from polyphon.bot.meeting_bot import MeetingBot

    # If server points to localhost with unencrypted ws://, check if server has TLS enabled
    if server.startswith("ws://localhost:") or server.startswith("ws://127.0.0.1:"):
        parsed = urllib.parse.urlparse(server)
        test_url = f"https://{parsed.netloc}/api/system"
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        try:
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({}),
                urllib.request.HTTPSHandler(context=ctx),
            )
            req = urllib.request.Request(test_url, headers={"User-Agent": "PolyphonBot"})
            with opener.open(req, timeout=1.0) as resp:
                if resp.status == 200:
                    server = server.replace("ws://", "wss://", 1)
        except Exception:
            pass

    console.rule("[bold cyan]Polyphon Meeting Bot 🤖[/bold cyan]")
    console.print(f"[bold green]Target Meeting:[/bold green] {meeting_url}")
    console.print(f"[bold green]Bot Display Name:[/bold green] {name}")
    if title:
        console.print(f"[bold green]Meeting Title:[/bold green] {title}")
    if session:
        console.print(f"[bold green]Session Stem:[/bold green] {session}")
    console.print(f"[bold green]Streaming Endpoint:[/bold green] {server}")
    console.print(f"[dim]Browser Mode: {'Headed (visible)' if headed else 'Headless'}[/dim]")
    console.print(
        "[dim italic]Note: The bot transcribes spoken meeting audio (microphones). It does not monitor or respond to text chat.[/dim italic]"
    )
    console.print("[dim]Press Ctrl+C at any time to leave the meeting and finalize reports.[/dim]\n")

    def on_transcript(event: dict[str, Any]) -> None:
        segments = event.get("segments", [])
        if segments:
            latest = segments[-1]
            spk = latest.get("speaker_name", "Speaker")
            txt = latest.get("text", "")
            console.print(f"[bold blue]{spk}:[/bold blue] {txt}")

    meeting_bot = MeetingBot(
        meeting_url=meeting_url,
        bot_name=name,
        server_ws_url=server,
        session_title=title,
        session_name=session,
        headless=not headed,
        duration_mins=duration,
        on_transcript=on_transcript,
        on_status=console.print,
    )

    try:
        asyncio.run(meeting_bot.start())
        console.print("\n[bold green]✔ Meeting Bot session completed and reports saved.[/bold green]")
    except KeyboardInterrupt:
        console.print("\n[yellow]Meeting Bot interrupted by user (Ctrl+C). Finalizing session...[/yellow]")
        console.print("[bold green]✔ Meeting Bot session ended and reports saved.[/bold green]")


@app.command("rename")
def rename_meeting_cmd(
    meeting: str = typer.Argument(
        ..., help="Meeting ID or manifest JSON path (e.g. 'live_1788613915' or 'outputs/live_1788613915.json')"
    ),
    title: str = typer.Argument(..., help="New title for the meeting recording (e.g. 'Sprint Planning')"),
    output_dir: Path | None = typer.Option(None, "--output-dir", "-o", help="Outputs and archive directory"),
):
    """Rename the title of an existing meeting recording across JSON manifest and HTML/MD reports."""
    import json

    from polyphon.types import PolyphonResult

    out_dir = output_dir or Path(os.environ.get("POLYPHON_OUTPUT_DIR", "outputs"))
    clean_target = meeting.strip()

    # Search for json manifest
    cand_path = Path(clean_target)
    json_path = None
    if cand_path.exists() and cand_path.is_file() and cand_path.suffix == ".json":
        json_path = cand_path
    else:
        stem = cand_path.stem
        direct = out_dir / f"{stem}.json"
        if direct.exists():
            json_path = direct
        else:
            for f in out_dir.glob("*.json"):
                if f.stem == stem or clean_target.lower() in f.stem.lower():
                    json_path = f
                    break

    if not json_path or not json_path.exists():
        console.print(f"[red]Error:[/red] Meeting manifest not found for: {meeting}")
        raise typer.Exit(1)

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        old_title = data.get("title") or json_path.stem.replace("_", " ")
        data["title"] = title.strip()
        json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

        # Companion live file
        live_file = json_path.parent / f"{json_path.stem}_live.json"
        if live_file.exists():
            try:
                live_data = json.loads(live_file.read_text(encoding="utf-8"))
                live_data["title"] = title.strip()
                live_file.write_text(json.dumps(live_data, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass

        # Regenerate HTML and Markdown reports
        poly_res = PolyphonResult.from_dict(data)
        poly_res.title = title.strip()
        html_file = json_path.parent / f"{json_path.stem}.html"
        if html_file.exists():
            html_file.write_text(poly_res.to_html(title=title.strip()), encoding="utf-8")
        md_file = json_path.parent / f"{json_path.stem}.md"
        if md_file.exists():
            md_file.write_text(poly_res.to_markdown(title=title.strip()), encoding="utf-8")

        console.print(
            f"[bold green]✔ Successfully renamed meeting:[/bold green] [dim]{old_title}[/dim] ➔ [bold cyan]{title.strip()}[/bold cyan]"
        )
        console.print(f"[dim]Updated: {json_path.name}[/dim]")
    except Exception as exc:
        console.print(f"[red]Error renaming meeting:[/red] {exc}")
        raise typer.Exit(1) from exc


@app.command("doctor")
def doctor():
    """Check that runtime dependencies (FFmpeg shared libraries) are usable."""
    import shutil

    console.rule("[bold cyan]Polyphon Environment Check[/bold cyan]")

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        console.print(f"[green]OK[/green]  ffmpeg executable: [dim]{ffmpeg_bin}[/dim]")
    else:
        console.print("[yellow]--[/yellow]  ffmpeg executable not on PATH (only needed for sample generation)")

    # Actually decode, rather than merely importing. `import torchcodec` swallows
    # a missing-FFmpeg error to keep its image decoders usable, so an import
    # alone reports success on a broken install.
    try:
        import tempfile

        import torch
        from torchcodec.decoders import AudioDecoder
        from torchcodec.encoders import AudioEncoder

        with tempfile.TemporaryDirectory() as tmp:
            probe = Path(tmp) / "probe.wav"
            AudioEncoder(torch.zeros(1, 1600), sample_rate=16000).to_file(str(probe))
            AudioDecoder(str(probe)).get_all_samples()
    except Exception as err:  # noqa: BLE001 - report any failure to the user
        if "libtorchcodec" in str(err):
            console.print("[red]FAIL[/red]  FFmpeg shared libraries could not be loaded.")
            console.print()
            console.print(
                "      torchcodec loads avcodec/avformat/avutil as libraries at runtime;\n"
                "      having the `ffmpeg` command on PATH is not sufficient."
            )
            if sys.platform == "win32":
                console.print(
                    "\n      On Windows the common packages (winget Gyan.FFmpeg, choco/scoop\n"
                    "      ffmpeg) are *static* builds shipping no DLLs. Install a shared build:\n"
                    "        [bold]winget install Gyan.FFmpeg.Shared[/bold]\n"
                    "      or unpack ffmpeg-release-full-shared.7z from\n"
                    "      https://www.gyan.dev/ffmpeg/builds/ and add its `bin` to PATH."
                )
            elif sys.platform == "darwin":
                console.print("\n      Install FFmpeg with: [bold]brew install ffmpeg[/bold]")
            else:
                console.print("\n      Install FFmpeg with: [bold]sudo apt install -y ffmpeg[/bold]")
            raise typer.Exit(1) from err
        console.print(f"[red]FAIL[/red]  Audio decoding check failed: {err}")
        raise typer.Exit(1) from err

    console.print("[green]OK[/green]  FFmpeg shared libraries loadable (audio decoding works)")
    console.print()
    console.print("[bold green]Environment looks good.[/bold green]")


def main():
    app()


if __name__ == "__main__":
    main()
