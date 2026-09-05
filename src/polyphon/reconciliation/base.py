"""Reconciliation and alignment engine for fusing ASR tokens with diarization intervals."""

from __future__ import annotations

from polyphon.types import Segment, SpeakerInterval, SpeakerStat, WordToken


class ReconciliationEngine:
    """Fuses word-level ASR timestamps with diarization speaker intervals."""

    def __init__(self, gap_tolerance_sec: float = 0.75, continuity_weight: float = 0.2):
        self.gap_tolerance_sec = gap_tolerance_sec
        self.continuity_weight = continuity_weight

    def reconcile(
        self,
        words: list[WordToken],
        intervals: list[SpeakerInterval],
        speaker_name_map: dict[str, str] | None = None,
    ) -> list[Segment]:
        """Align word tokens to speaker intervals and group into dialogue segments."""
        if not words:
            return []

        from polyphon.fillers import tag_filler_tokens

        tag_filler_tokens(words)

        if not intervals:
            # Fallback when no diarization intervals are available
            full_text = " ".join(w.word for w in words)
            return [
                Segment(
                    start=words[0].start,
                    end=words[-1].end,
                    speaker_id="SPEAKER_00",
                    speaker_name=(speaker_name_map.get("SPEAKER_00", "Speaker 0") if speaker_name_map else "Speaker 0"),
                    text=full_text,
                    words=words,
                )
            ]

        name_map = speaker_name_map or {}
        assigned_words: list[tuple[WordToken, str]] = []
        sorted_intervals = sorted(intervals, key=lambda x: x.start)
        n_intervals = len(sorted_intervals)
        last_speaker = sorted_intervals[0].speaker_id
        max_gap = self.gap_tolerance_sec
        window_start = 0

        for word in words:
            word_len = max(0.001, word.end - word.start)
            best_spk = None
            best_score = -1.0
            min_dist = float("inf")
            fallback_spk = None

            # Advance window pointer past intervals that ended well before word start
            while window_start < n_intervals and sorted_intervals[window_start].end < word.start - max_gap:
                window_start += 1

            # Search only candidate intervals within the temporal neighborhood
            for i in range(window_start, n_intervals):
                interval = sorted_intervals[i]
                if interval.start > word.end + max_gap:
                    break

                # 1. Overlap evaluation
                overlap = max(0.0, min(word.end, interval.end) - max(word.start, interval.start))
                if overlap > 0.0:
                    score = overlap / word_len
                    if interval.speaker_id == last_speaker:
                        score += self.continuity_weight

                    if score > best_score:
                        best_score = score
                        best_spk = interval.speaker_id

                # 2. Distance evaluation for gap fallback
                if word.start >= interval.end:
                    dist = word.start - interval.end
                elif word.end <= interval.start:
                    dist = interval.start - word.end
                else:
                    dist = 0.0

                if dist < min_dist:
                    min_dist = dist
                    if dist <= self.gap_tolerance_sec:
                        fallback_spk = interval.speaker_id

            # Default to previous speaker if still unassigned
            assigned_spk = best_spk or fallback_spk or last_speaker
            assigned_words.append((word, assigned_spk))
            last_speaker = assigned_spk

        # 3. Group contiguous words into dialogue segments
        segments: list[Segment] = []
        current_spk = None
        current_words: list[WordToken] = []

        for word, spk_id in assigned_words:
            if current_spk is None:
                current_spk = spk_id
                current_words = [word]
            elif spk_id == current_spk:
                current_words.append(word)
            else:
                # Flush previous segment
                seg_text = " ".join(w.word for w in current_words)
                segments.append(
                    Segment(
                        start=current_words[0].start,
                        end=current_words[-1].end,
                        speaker_id=current_spk,
                        speaker_name=name_map.get(current_spk, current_spk),
                        text=seg_text,
                        words=current_words,
                    )
                )
                current_spk = spk_id
                current_words = [word]

        if current_words and current_spk is not None:
            seg_text = " ".join(w.word for w in current_words)
            segments.append(
                Segment(
                    start=current_words[0].start,
                    end=current_words[-1].end,
                    speaker_id=current_spk,
                    speaker_name=name_map.get(current_spk, current_spk),
                    text=seg_text,
                    words=current_words,
                )
            )

        return segments

    @staticmethod
    def compute_speaker_stats(segments: list[Segment], total_duration: float) -> dict[str, SpeakerStat]:
        """Compute talk-time statistics for all speakers in the segments."""
        stats: dict[str, dict[str, float]] = {}

        for seg in segments:
            label = seg.speaker_name or seg.speaker_id
            dur = max(0.0, seg.end - seg.start)
            if label not in stats:
                stats[label] = {"talk_time": 0.0, "count": 0}
            stats[label]["talk_time"] += dur
            stats[label]["count"] += 1

        result: dict[str, SpeakerStat] = {}
        safe_duration = max(total_duration, 0.001)

        for label, data in stats.items():
            talk_sec = round(data["talk_time"], 2)
            pct = round((talk_sec / safe_duration) * 100, 1)
            result[label] = SpeakerStat(
                talk_time_seconds=talk_sec,
                percentage=pct,
                segment_count=int(data["count"]),
            )

        return result
