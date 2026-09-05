"""Helper script to download multi-speaker sample meetings and generate MP4 videos for testing."""

import argparse
import shutil
import subprocess
import urllib.request
from pathlib import Path

DEFAULT_URL = "https://files.pyannote.ai/marklex1min.wav"
CHALLENGING_URL = "http://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2004a/audio/ES2004a.Mix-Headset.wav"

ROOT_DIR = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = ROOT_DIR / "examples"


# Common font locations per platform. drawtext needs an explicit fontfile on
# systems without a fontconfig config (notably Windows ffmpeg builds, which
# otherwise emit "Fontconfig error: Cannot load default config file").
FONT_CANDIDATES = [
    # Windows
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/tahoma.ttf",
    # macOS
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
]


def find_font() -> str | None:
    """Return the first font file present on this system, or None."""
    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return None


def escape_fontfile(path: str) -> str:
    """Escape a font path for use inside an ffmpeg filtergraph argument."""
    # ffmpeg filter syntax treats ':' as an option separator and a backslash as an
    # escape char, so both must be escaped in Windows paths like C:/Windows/...
    return path.replace("\\", "/").replace(":", r"\:")


def main():
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser(description="Fetch meeting audio/video samples for Polyphon testing.")
    parser.add_argument(
        "--challenging",
        "-c",
        action="store_true",
        help="Fetch challenging 4-speaker AMI meeting (ES2004a, ~17.5m with real crosstalk and interruptions)",
    )
    args = parser.parse_args()

    print("=" * 65)
    print(" Polyphon Sample Media Fetcher")
    print("=" * 65)

    if args.challenging:
        sample_url = CHALLENGING_URL
        wav_path = EXAMPLES_DIR / "meeting_challenging.wav"
        mp4_path = EXAMPLES_DIR / "meeting_challenging.mp4"
        mp4_3min_path = EXAMPLES_DIR / "meeting_challenging_3min.mp4"
        title_text = "Polyphon Stress Test - AMI Meeting (4 Speakers)"
        subtitle_text = "Project Kickoff and Industrial Design (ES2004a)"
    else:
        sample_url = DEFAULT_URL
        wav_path = EXAMPLES_DIR / "meeting.wav"
        mp4_path = EXAMPLES_DIR / "meeting.mp4"
        mp4_3min_path = None
        title_text = "Polyphon Test Meeting"
        subtitle_text = "Lex Fridman and Mark Zuckerberg"
        print("🎯 Mode: Standard 2-Speaker Benchmark (~87s)")

    # 1. Download WAV
    if not wav_path.exists():
        print("⬇️  Downloading meeting audio...")
        print(f"    Source: {sample_url}")
        urllib.request.urlretrieve(sample_url, wav_path)
        print(f"✔ Saved: {wav_path.name}")
    else:
        print(f"✔ {wav_path.name} already exists.")

    # 2. Convert to MP4 using ffmpeg if installed
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin and not mp4_path.exists():
        print(f"🎬 Generating {mp4_path.name} with ffmpeg visual title card...")
        input_args = [
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(wav_path),
            "-f",
            "lavfi",
            "-i",
            "color=c=0x0f172a:s=1280x720:r=10",
        ]
        output_args = [
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(mp4_path),
        ]

        font_path = find_font()
        vf_args = []
        if font_path:
            fontfile = escape_fontfile(font_path)
            vf_filter = (
                f"drawtext=fontfile='{fontfile}':text='{title_text}':fontcolor=white:"
                f"fontsize=40:x=(w-text_w)/2:y=(h-text_h)/2-30,"
                f"drawtext=fontfile='{fontfile}':text='{subtitle_text}':fontcolor=0x94a3b8:"
                f"fontsize=24:x=(w-text_w)/2:y=(h-text_h)/2+40"
            )
            vf_args = ["-vf", vf_filter]
        else:
            print("ℹ️  No system font found; generating a solid background instead.")

        try:
            subprocess.run(input_args + vf_args + output_args, check=True)
            suffix = "" if vf_args else " (solid background)"
            print(f"✔ Generated: {mp4_path.name}{suffix}")
        except subprocess.CalledProcessError:
            if not vf_args:
                print("⚠️  Failed to generate MP4 via ffmpeg.")
            else:
                try:
                    subprocess.run(input_args + output_args, check=True)
                    print(f"✔ Generated: {mp4_path.name} (solid background)")
                except subprocess.CalledProcessError as e:
                    print(f"⚠️  Failed to generate MP4 via ffmpeg: {e}")

    # Also generate a 3-minute clip for faster iteration if challenging
    if mp4_3min_path and not mp4_3min_path.exists() and mp4_path.exists():
        if ffmpeg_bin:
            print(f"✂️  Extracting first 3 minutes for quick testing: {mp4_3min_path.name}...")
            clip_cmd = [
                ffmpeg_bin,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(mp4_path),
                "-t",
                "180",
                "-c",
                "copy",
                str(mp4_3min_path),
            ]
            try:
                subprocess.run(clip_cmd, check=True)
                print(f"✔ Generated: {mp4_3min_path.name}")
            except subprocess.CalledProcessError as e:
                print(f"⚠️  Failed to extract 3min clip: {e}")

    print("\n🎉 Ready to test! Run:")
    if args.challenging:
        print("  ▶ Quick Stress Test (First 3 minutes):")
        print("      uv run polyphon transcribe examples/meeting_challenging_3min.mp4 --diarize --summarize\n")
        print("  ▶ Full Marathon Stress Test (All ~17.5 minutes):")
        print("      uv run polyphon transcribe examples/meeting_challenging.mp4 --diarize --summarize")
    else:
        print("      uv run polyphon transcribe examples/meeting.mp4 --diarize --summarize")
    print("=" * 65)


if __name__ == "__main__":
    main()
