# Polyphon Examples & Benchmarks

This directory contains sample multi-speaker recordings and generated intelligence reports for benchmarking and testing Polyphon.

---

## 🎧 Sample Recordings

| File | Speakers | Duration | Description |
| :--- | :--- | :--- | :--- |
| **`meeting.mp4`** / **`meeting.wav`** | 2 | ~87s | Standard dialogue benchmark between Lex Fridman and Mark Zuckerberg. Clean speech, natural turn-taking. |
| **`meeting_challenging_3min.mp4`** | 4 | 3m 00s | Rapid stress test clip with multiple overlapping interruptions, crosstalk, and speaker transitions. |
| **`meeting_challenging.mp4`** / **`meeting_challenging.wav`** | 4 | ~17.5m | AMI Meeting Corpus (ES2004a) project kickoff meeting with 4 participants discussing industrial design and budget. |

---

## 🚀 Downloading Samples

To fetch or regenerate these samples, run the included helper script:

```bash
# 1. Download standard 2-speaker benchmark (~87s)
uv run python scripts/download_sample.py

# 2. Download challenging 4-speaker benchmark (~17.5m & 3min clip)
uv run python scripts/download_sample.py --challenging
```

---

## 💻 Quick Test Commands

### 1. Transcribe & Diarize
```bash
uv run polyphon transcribe examples/meeting.mp4 --diarize
```

### 2. Full Intelligence Report (HTML Export)
```bash
uv run polyphon transcribe examples/meeting_challenging_3min.mp4 \
  --diarize \
  --infer-names \
  --auto-enroll \
  --summarize \
  --format html
```

### 3. Real-Time Streaming Simulation
```bash
uv run polyphon stream examples/meeting.mp4 --simulate --speed 1.0
```

