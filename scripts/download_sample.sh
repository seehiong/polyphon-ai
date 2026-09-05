#!/usr/bin/env bash
# scripts/download_sample.sh
# Downloads sample multi-speaker meetings and generates MP4 video files for testing.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CHALLENGING=false
for arg in "$@"; do
    if [ "$arg" = "--challenging" ] || [ "$arg" = "-c" ]; then
        CHALLENGING=true
    fi
done

echo "============================================================"
echo " Polyphon Sample Media Fetcher"
echo "============================================================"

EXAMPLES_DIR="${ROOT_DIR}/examples"
mkdir -p "${EXAMPLES_DIR}"

if [ "$CHALLENGING" = true ]; then
    SAMPLE_URL="http://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2004a/audio/ES2004a.Mix-Headset.wav"
    TARGET_WAV="${EXAMPLES_DIR}/meeting_challenging.wav"
    TARGET_MP4="${EXAMPLES_DIR}/meeting_challenging.mp4"
    TARGET_3MIN="${EXAMPLES_DIR}/meeting_challenging_3min.mp4"
    TITLE="Polyphon Stress Test - AMI Meeting (4 Speakers)"
    SUBTITLE="Project Kickoff and Industrial Design (ES2004a)"
    echo "🎯 Mode: CHALLENGING 4-Speaker Meeting (AMI Corpus ES2004a)"
    echo "   Features: 4 participants, overlaps, cross-talk, budget and design debates"
else
    SAMPLE_URL="https://files.pyannote.ai/marklex1min.wav"
    TARGET_WAV="${EXAMPLES_DIR}/meeting.wav"
    TARGET_MP4="${EXAMPLES_DIR}/meeting.mp4"
    TARGET_3MIN=""
    TITLE="Polyphon Test Meeting"
    SUBTITLE="Lex Fridman and Mark Zuckerberg"
    echo "🎯 Mode: Standard 2-Speaker Benchmark (~87s)"
fi

# 1. Download WAV
if [ ! -f "${TARGET_WAV}" ]; then
    echo "⬇️  Downloading meeting audio..."
    echo "    Source: ${SAMPLE_URL}"
    curl -fL -o "${TARGET_WAV}" "${SAMPLE_URL}"
    echo "✔ Saved to: ${TARGET_WAV}"
else
    echo "✔ ${TARGET_WAV} already exists."
fi

# 2. Generate MP4 with ffmpeg
if [ ! -f "${TARGET_MP4}" ]; then
    if command -v ffmpeg &>/dev/null; then
        echo "🎬 Generating ${TARGET_MP4} using ffmpeg..."
        ffmpeg -y -hide_banner -loglevel error \
            -i "${TARGET_WAV}" \
            -f lavfi -i color=c=0x0f172a:s=1280x720:r=10 \
            -vf "drawtext=text='${TITLE}':fontcolor=white:fontsize=40:x=(w-text_w)/2:y=(h-text_h)/2-30,drawtext=text='${SUBTITLE}':fontcolor=0x94a3b8:fontsize=24:x=(w-text_w)/2:y=(h-text_h)/2+40" \
            -shortest -c:v libx264 -pix_fmt yuv420p -c:a aac -b:a 192k \
            "${TARGET_MP4}" || \
        ffmpeg -y -hide_banner -loglevel error \
            -i "${TARGET_WAV}" \
            -f lavfi -i color=c=0x0f172a:s=1280x720:r=10 \
            -shortest -c:v libx264 -pix_fmt yuv420p -c:a aac -b:a 192k \
            "${TARGET_MP4}"
        echo "✔ Generated video: ${TARGET_MP4}"

        if [ -n "${TARGET_3MIN}" ] && [ ! -f "${TARGET_3MIN}" ]; then
            echo "✂️  Extracting first 3 minutes for quick stress testing..."
            ffmpeg -y -hide_banner -loglevel error -i "${TARGET_MP4}" -t 180 -c copy "${TARGET_3MIN}"
            echo "✔ Generated 3min clip: ${TARGET_3MIN}"
        fi
    else
        echo "⚠️  ffmpeg not found in PATH; skipping MP4 creation."
    fi
else
    echo "✔ ${TARGET_MP4} already exists."
fi

echo ""
echo "🎉 Ready to test! Run:"
if [ "$CHALLENGING" = true ]; then
    echo "  ▶ Quick Stress Test (First 3 minutes):"
    echo "      uv run polyphon transcribe examples/meeting_challenging_3min.mp4 --diarize --summarize"
    echo ""
    echo "  ▶ Full Marathon Test (All ~17.5 minutes):"
    echo "      uv run polyphon transcribe examples/meeting_challenging.mp4 --diarize --summarize"
else
    echo "      uv run polyphon transcribe examples/meeting.mp4 --diarize --summarize"
fi
echo "============================================================"
