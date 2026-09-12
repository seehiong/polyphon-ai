"""Core data types and schemas for Polyphon (zero-dependency using standard dataclasses)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class WordToken:
    """Word-level timestamp and confidence."""

    word: str
    start: float
    end: float
    score: float = 1.0
    is_filler: bool = False


@dataclass
class SpeakerInterval:
    """Raw diarization speaker interval."""

    speaker_id: str
    start: float
    end: float
    is_exclusive: bool = True
    confidence: float = 1.0


@dataclass
class SpeakerInfo:
    """Enrolled or identified speaker profile."""

    id: str
    name: str
    confidence: float = 1.0


@dataclass
class Segment:
    """Reconciled transcript segment labeled with speaker."""

    start: float
    end: float
    speaker_id: str
    speaker_name: str
    text: str
    words: list[WordToken] = field(default_factory=list)

    def get_clean_text(self) -> str:
        """Return segment text with filler words excluded, preserving original words."""
        from polyphon.fillers import clean_text_from_words, is_filler_word

        if not self.words:
            return " ".join(w for w in self.text.split() if not is_filler_word(w)).strip()

        return clean_text_from_words(self.words)


@dataclass
class ActionItem:
    """Action item extracted from conversation."""

    task: str
    assignee: str | None = None
    due_date: str | None = None


@dataclass
class SpeakerStat:
    """Talk time analytics per speaker."""

    talk_time_seconds: float
    percentage: float
    segment_count: int


@dataclass
class DiscussionPoint:
    """A topic discussed in the meeting, with a short summary of what was covered."""

    topic: str
    summary: str


@dataclass
class Insights:
    """Structured semantic insights extracted by LLM."""

    summary: str = ""
    topics: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    action_items: list[ActionItem] = field(default_factory=list)
    speaker_stats: dict[str, SpeakerStat] = field(default_factory=dict)
    discussion_points: list[DiscussionPoint] = field(default_factory=list)


@dataclass
class PolyphonResult:
    """Top-level result container for Polyphon."""

    duration: float
    language: str = "en"
    speakers: list[SpeakerInfo] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    insights: Insights | None = None
    title: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert result to standard Python dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolyphonResult:
        """Construct a PolyphonResult instance from a raw dictionary or JSON payload."""
        from polyphon.fillers import is_filler_word

        speakers = [
            SpeakerInfo(
                id=s.get("id", ""),
                name=s.get("name", ""),
                confidence=s.get("confidence", 1.0),
            )
            for s in data.get("speakers", [])
        ]
        segments = []
        for s in data.get("segments", []):
            words = [
                WordToken(
                    word=w.get("word", ""),
                    start=w.get("start", 0.0),
                    end=w.get("end", 0.0),
                    score=w.get("score", 1.0),
                    is_filler=bool(w.get("is_filler", False)) or is_filler_word(w.get("word", "")),
                )
                for w in s.get("words", [])
            ]
            segments.append(
                Segment(
                    start=s.get("start", 0.0),
                    end=s.get("end", 0.0),
                    speaker_id=s.get("speaker_id", ""),
                    speaker_name=s.get("speaker_name", ""),
                    text=s.get("text", ""),
                    words=words,
                )
            )
        insights = None
        if data.get("insights"):
            ins = data["insights"]
            act_items = [
                ActionItem(
                    task=a.get("task", ""),
                    assignee=a.get("assignee"),
                    due_date=a.get("due_date"),
                )
                for a in ins.get("action_items", [])
            ]
            spk_stats = {
                k: SpeakerStat(
                    talk_time_seconds=v.get("talk_time_seconds", 0.0),
                    percentage=v.get("percentage", 0.0),
                    segment_count=v.get("segment_count", 0),
                )
                for k, v in ins.get("speaker_stats", {}).items()
            }
            disc_points = [
                DiscussionPoint(topic=d.get("topic", ""), summary=d.get("summary", ""))
                for d in ins.get("discussion_points", [])
            ]
            insights = Insights(
                summary=ins.get("summary", ""),
                topics=ins.get("topics", []),
                decisions=ins.get("decisions", []),
                action_items=act_items,
                speaker_stats=spk_stats,
                discussion_points=disc_points,
            )
        return cls(
            duration=data.get("duration", 0.0),
            language=data.get("language", "en"),
            speakers=speakers,
            segments=segments,
            insights=insights,
            title=data.get("title"),
        )

    def to_json(self, indent: int = 2) -> str:
        """Export result as formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_markdown(self, title: str | None = None, clean_fillers: bool = False) -> str:
        """Export result as human-readable Markdown dialogue."""
        doc_title = title or self.title or f"Transcript ({self.duration:.1f}s)"
        lines = [f"# {doc_title}", ""]
        if self.speakers:
            lines.append("### Speakers")
            seen_names = set()
            for spk in self.speakers:
                spk_name = spk.name or spk.id
                if spk_name not in seen_names:
                    seen_names.add(spk_name)
                    matched_ids = [s.id for s in self.speakers if (s.name or s.id) == spk_name]
                    ids_str = ", ".join(f"`{i}`" for i in matched_ids)
                    lines.append(f"- **{spk_name}** ({ids_str})")
            lines.append("")

        lines.append("### Dialogue")
        current_speaker = None
        for seg in self.segments:
            text = (seg.get_clean_text() if clean_fillers else seg.text).strip()
            if not text:
                continue
            speaker_label = seg.speaker_name or seg.speaker_id
            m, s = divmod(int(seg.start), 60)
            time_str = f"{m:02d}:{s:02d}"

            if speaker_label != current_speaker:
                lines.append(f"\n**[{time_str}] {speaker_label}:**")
                current_speaker = speaker_label

            lines.append(f"{text}")

        if self.insights:
            if self.insights.summary:
                lines.append("\n---\n### Summary")
                lines.append(self.insights.summary)

            if self.insights.topics:
                lines.append("\n### Topics Discussed")
                lines.extend(f"- {topic}" for topic in self.insights.topics)

            if self.insights.decisions:
                lines.append("\n### Key Decisions")
                lines.extend(f"- {decision}" for decision in self.insights.decisions)

            if self.insights.action_items:
                lines.append("\n### Action Items")
                for item in self.insights.action_items:
                    assignee_str = f" ({item.assignee})" if item.assignee else ""
                    lines.append(f"- [ ] {item.task}{assignee_str}")

            if self.insights.speaker_stats:
                lines.append("\n### Speaker Statistics")
                for spk_name, stat in self.insights.speaker_stats.items():
                    lines.append(
                        f"- **{spk_name}**: {stat.talk_time_seconds:.1f}s ({stat.percentage:.1f}%) across {stat.segment_count} turns"
                    )

        return "\n".join(lines)

    def to_srt(self, clean_fillers: bool = False) -> str:
        """Export result as standard SubRip (.srt) subtitle format."""

        def format_srt_time(seconds: float) -> str:
            millis = int((seconds - int(seconds)) * 1000)
            mins, secs = divmod(int(seconds), 60)
            hours, mins = divmod(mins, 60)
            return f"{hours:02d}:{mins:02d}:{secs:02d},{millis:03d}"

        srt_blocks = []
        idx = 1
        for seg in self.segments:
            seg_text = (seg.get_clean_text() if clean_fillers else seg.text).strip()
            if not seg_text:
                continue
            speaker_label = seg.speaker_name or seg.speaker_id
            start_str = format_srt_time(seg.start)
            end_str = format_srt_time(seg.end)
            text = f"[{speaker_label}] {seg_text}"
            srt_blocks.append(f"{idx}\n{start_str} --> {end_str}\n{text}\n")
            idx += 1

        return "\n".join(srt_blocks)

    def to_html(self, title: str | None = None, clean_fillers: bool = False) -> str:
        """Export result as a self-contained, interactive modern HTML document."""
        import base64
        import html as html_escape
        from pathlib import Path

        doc_title = title or self.title or "Polyphon Meeting Report"
        logo_path = Path(__file__).parent / "server" / "static" / "logo.jpg"
        if not logo_path.exists():
            logo_path = Path("assets/logo.jpg")

        logo_img_tag = ""
        if logo_path.exists():
            try:
                b64 = base64.b64encode(logo_path.read_bytes()).decode("utf-8")
                logo_img_tag = (
                    f'<img src="data:image/jpeg;base64,{b64}" alt="Polyphon Logo" '
                    'style="width: 52px; height: 52px; border-radius: 12px; object-fit: cover; '
                    'box-shadow: 0 4px 12px rgba(0,0,0,0.4); flex-shrink: 0;">'
                )
            except Exception:
                pass

        m, s = divmod(int(self.duration), 60)
        dur_str = f"{m}m {s:02d}s" if m > 0 else f"{s:.1f}s"

        palette = [
            {"color": "#3b82f6", "light": "rgba(59, 130, 246, 0.15)"},
            {"color": "#10b981", "light": "rgba(16, 185, 129, 0.15)"},
            {"color": "#8b5cf6", "light": "rgba(139, 92, 246, 0.15)"},
            {"color": "#f59e0b", "light": "rgba(245, 158, 11, 0.15)"},
            {"color": "#ec4899", "light": "rgba(236, 72, 153, 0.15)"},
            {"color": "#06b6d4", "light": "rgba(6, 182, 212, 0.15)"},
            {"color": "#f97316", "light": "rgba(249, 115, 22, 0.15)"},
        ]

        unique_speakers = []
        seen_names = set()
        for seg in self.segments:
            sname = seg.speaker_name or seg.speaker_id
            if sname not in seen_names:
                seen_names.add(sname)
                unique_speakers.append((seg.speaker_id, sname))

        for spk in self.speakers:
            sname = spk.name or spk.id
            if sname not in seen_names:
                seen_names.add(sname)
                unique_speakers.append((spk.id, sname))

        spk_color_map = {}
        for idx, (sid, sname) in enumerate(unique_speakers):
            col = palette[idx % len(palette)]
            spk_color_map[sname] = col
            spk_color_map[sid] = col
            for s in self.segments:
                if (s.speaker_name or s.speaker_id) == sname:
                    spk_color_map[s.speaker_id] = col
            for s in self.speakers:
                if (s.name or s.id) == sname:
                    spk_color_map[s.id] = col

        # Speaker Talk-Time Bar
        bar_segments = []
        stats = self.insights.speaker_stats if (self.insights and self.insights.speaker_stats) else {}
        for sid, sname in unique_speakers:
            stat = stats.get(sname) or stats.get(sid)
            pct = stat.percentage if stat else (100.0 / max(1, len(unique_speakers)))
            color = spk_color_map.get(sname, palette[0])["color"]
            bar_segments.append(
                f'<div class="progress-segment" style="width: {pct:.1f}%; background-color: {color};" '
                f'title="{html_escape.escape(sname)}: {pct:.1f}%"></div>'
            )

        # Insights blocks
        summary_html = ""
        topics_html = ""
        decisions_html = ""
        actions_html = ""

        if self.insights:
            if self.insights.summary:
                summary_html = (
                    f'<div class="card summary-card">'
                    f"<h3>💡 Executive Summary</h3>"
                    f"<p>{html_escape.escape(self.insights.summary)}</p>"
                    f"</div>"
                )

            if self.insights.topics:
                tags = "".join(f'<span class="topic-tag">{html_escape.escape(t)}</span>' for t in self.insights.topics)
                topics_html = f'<div class="meta-section"><h4>🏷️ Topics</h4><div class="topic-list">{tags}</div></div>'

            if self.insights.decisions:
                items = "".join(f"<li>{html_escape.escape(d)}</li>" for d in self.insights.decisions)
                decisions_html = (
                    f'<div class="meta-section"><h4>🎯 Key Decisions</h4><ul class="decisions-list">{items}</ul></div>'
                )

            if self.insights.action_items:
                act_items = []
                for idx, a in enumerate(self.insights.action_items):
                    assignee = (
                        f'<span class="assignee-badge">{html_escape.escape(a.assignee)}</span>' if a.assignee else ""
                    )
                    act_items.append(
                        f'<div class="action-item">'
                        f'<input type="checkbox" id="act_{idx}" onclick="this.nextElementSibling.classList.toggle(\'done\')">'
                        f'<label for="act_{idx}">{html_escape.escape(a.task)} {assignee}</label>'
                        f"</div>"
                    )
                actions_html = f'<div class="meta-section"><h4>✅ Action Items</h4><div class="action-list">{"".join(act_items)}</div></div>'

        # Dialogue Turns
        dialogue_cards = []
        for seg in self.segments:
            verbatim_text = seg.text.strip()
            clean_text = seg.get_clean_text().strip()
            seg_text = clean_text if clean_fillers else verbatim_text
            if not seg_text:
                continue
            m_start, s_start = divmod(int(seg.start), 60)
            time_str = f"{m_start:02d}:{s_start:02d}"
            spk_color = spk_color_map.get(seg.speaker_name, spk_color_map.get(seg.speaker_id, palette[0]))
            sname = html_escape.escape(seg.speaker_name or seg.speaker_id)
            clean_empty_attr = "true" if not clean_text else "false"
            dialogue_cards.append(
                f'<div class="turn-card" data-speaker="{html_escape.escape(seg.speaker_name or seg.speaker_id)}" data-speaker-id="{html_escape.escape(seg.speaker_id)}" data-clean-empty="{clean_empty_attr}">'
                f'<div class="turn-header">'
                f'<span class="speaker-badge" style="background-color: {spk_color["light"]}; color: {spk_color["color"]}; border: 1px solid {spk_color["color"]};">{sname}</span>'
                f'<span class="time-stamp">[{time_str}]</span>'
                f"</div>"
                f'<div class="turn-body" data-verbatim="{html_escape.escape(verbatim_text)}" data-clean="{html_escape.escape(clean_text)}">{html_escape.escape(seg_text)}</div>'
                f"</div>"
            )

        filter_buttons = [
            '<button class="filter-btn active" onclick="filterSpeaker(\'all\', this)">All Speakers</button>'
        ]
        for _sid, sname in unique_speakers:
            col = spk_color_map.get(sname, palette[0])["color"]
            filter_buttons.append(
                f'<button class="filter-btn" style="border-left: 4px solid {col};" onclick="filterSpeaker(\'{html_escape.escape(sname)}\', this)">'
                f"{html_escape.escape(sname)}</button>"
            )

        html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html_escape.escape(doc_title)}</title>
<style>
  :root {{
    --bg-main: #0f172a;
    --bg-card: #1e293b;
    --bg-card-hover: #334155;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --accent: #38bdf8;
    --border: #334155;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background-color: var(--bg-main);
    color: var(--text-primary);
    line-height: 1.6;
    padding: 2rem 1rem;
  }}
  .container {{ max-width: 960px; margin: 0 auto; }}
  header {{ margin-bottom: 2rem; border-bottom: 1px solid var(--border); padding-bottom: 1.5rem; }}
  h1 {{ font-size: 2rem; font-weight: 700; color: var(--text-primary); margin-bottom: 0.5rem; }}
  .header-metrics {{ display: flex; gap: 1rem; flex-wrap: wrap; color: var(--text-secondary); font-size: 0.95rem; }}
  .badge {{ background: var(--bg-card); padding: 0.25rem 0.75rem; border-radius: 9999px; border: 1px solid var(--border); }}
  
  .talktime-container {{ margin: 1.5rem 0; }}
  .talktime-label {{ font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 0.5rem; display: flex; justify-content: space-between; }}
  .talktime-bar {{ display: flex; height: 12px; border-radius: 6px; overflow: hidden; background: var(--border); }}
  .progress-segment {{ transition: opacity 0.2s; }}
  .progress-segment:hover {{ opacity: 0.8; cursor: pointer; }}

  .card {{ background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; }}
  .summary-card h3 {{ color: var(--accent); margin-bottom: 0.75rem; font-size: 1.25rem; }}
  .meta-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem; margin-bottom: 1.5rem; }}
  .meta-section h4 {{ color: var(--accent); margin-bottom: 0.75rem; font-size: 1rem; border-bottom: 1px solid var(--border); padding-bottom: 0.25rem; }}
  .topic-list {{ display: flex; flex-wrap: wrap; gap: 0.5rem; }}
  .topic-tag {{ background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); padding: 0.25rem 0.75rem; border-radius: 6px; font-size: 0.85rem; }}
  .decisions-list {{ padding-left: 1.25rem; font-size: 0.95rem; }}
  .decisions-list li {{ margin-bottom: 0.4rem; }}
  .action-list {{ display: flex; flex-direction: column; gap: 0.5rem; }}
  .action-item {{ display: flex; align-items: center; gap: 0.5rem; font-size: 0.95rem; }}
  .action-item input[type="checkbox"] {{ accent-color: var(--accent); width: 18px; height: 18px; cursor: pointer; }}
  .action-item label.done {{ text-decoration: line-through; color: var(--text-secondary); }}
  .assignee-badge {{ background: var(--bg-card-hover); color: var(--accent); font-size: 0.75rem; padding: 0.15rem 0.5rem; border-radius: 4px; margin-left: 0.5rem; }}

  .controls {{ display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; align-items: center; }}
  .search-input {{ flex: 1; min-width: 240px; background: var(--bg-card); border: 1px solid var(--border); color: var(--text-primary); padding: 0.6rem 1rem; border-radius: 8px; font-size: 0.95rem; outline: none; }}
  .search-input:focus {{ border-color: var(--accent); }}
  .filter-btn-group {{ display: flex; gap: 0.5rem; flex-wrap: wrap; }}
  .filter-btn {{ background: var(--bg-card); border: 1px solid var(--border); color: var(--text-secondary); padding: 0.5rem 0.9rem; border-radius: 8px; font-size: 0.85rem; cursor: pointer; transition: all 0.2s; }}
  .filter-btn:hover, .filter-btn.active {{ background: var(--bg-card-hover); color: var(--text-primary); border-color: var(--accent); }}

  .dialogue-list {{ display: flex; flex-direction: column; gap: 1rem; }}
  .turn-card {{ background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px; padding: 1rem 1.25rem; transition: border-color 0.2s; }}
  .turn-card:hover {{ border-color: var(--bg-card-hover); }}
  .turn-header {{ display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem; }}
  .speaker-badge {{ font-size: 0.8rem; font-weight: 600; padding: 0.2rem 0.6rem; border-radius: 6px; }}
  .time-stamp {{ font-size: 0.8rem; color: var(--text-secondary); font-family: monospace; }}
  .turn-body {{ font-size: 0.95rem; color: #e2e8f0; }}
</style>
</head>
<body>
<div class="container">
  <header>
    <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 0.85rem;">
      {logo_img_tag}
      <div>
        <h1 style="margin-bottom: 0.15rem;">{html_escape.escape(doc_title)}</h1>
        <p style="color: var(--text-secondary); font-size: 0.85rem; font-weight: 500;">Polyphon Speech AI Engine</p>
      </div>
    </div>
    <div class="header-metrics">
      <span class="badge">⏱️ Duration: <strong>{dur_str}</strong></span>
      <span class="badge">🌐 Language: <strong>{html_escape.escape(self.language.upper())}</strong></span>
      <span class="badge">👥 Speakers: <strong>{len(unique_speakers)}</strong></span>
      <span class="badge">💬 Dialogue Turns: <strong>{len(self.segments)}</strong></span>
    </div>

    <div class="talktime-container">
      <div class="talktime-label"><span>Talk Time Distribution</span><span>{len(unique_speakers)} Participants</span></div>
      <div class="talktime-bar">
        {"".join(bar_segments)}
      </div>
    </div>
  </header>

  {summary_html}

  <div class="meta-grid">
    {topics_html}
    {decisions_html}
  </div>

  {actions_html}

  <div class="controls" style="margin-top: 2rem;">
    <input type="text" class="search-input" id="searchBox" placeholder="🔍 Search transcript keywords..." oninput="applyFilters()">
    <button class="filter-btn{' active' if clean_fillers else ''}" id="btnCleanVerbatimHtml" onclick="toggleCleanVerbatimHtml(this)">✨ Clean Verbatim</button>
    <div class="filter-btn-group" id="filterGroup">
      {"".join(filter_buttons)}
    </div>
  </div>

  <div class="dialogue-list" id="dialogueList">
    {"".join(dialogue_cards)}
  </div>
</div>

<script>
  let activeSpeaker = 'all';
  let isCleanVerbatim = {"true" if clean_fillers else "false"};

  function toggleCleanVerbatimHtml(btn) {{
    isCleanVerbatim = !isCleanVerbatim;
    if (btn) btn.classList.toggle('active', isCleanVerbatim);
    document.querySelectorAll('.turn-card').forEach(card => {{
      const body = card.querySelector('.turn-body');
      if (body) {{
        body.textContent = isCleanVerbatim ? (body.getAttribute('data-clean') || body.getAttribute('data-verbatim')) : body.getAttribute('data-verbatim');
      }}
    }});
    applyFilters();
  }}

  function filterSpeaker(spkId, btn) {{
    activeSpeaker = spkId;
    document.querySelectorAll('.filter-btn').forEach(b => {{
      if (b.id !== 'btnCleanVerbatimHtml') b.classList.remove('active');
    }});
    if (btn) btn.classList.add('active');
    applyFilters();
  }}

  function applyFilters() {{
    const query = document.getElementById('searchBox').value.toLowerCase();
    const cards = document.querySelectorAll('.turn-card');
    cards.forEach(card => {{
      const spk = card.getAttribute('data-speaker');
      const text = card.querySelector('.turn-body').innerText.toLowerCase();
      const matchSpk = (activeSpeaker === 'all' || spk === activeSpeaker || card.getAttribute('data-speaker-id') === activeSpeaker);
      const matchText = (!query || text.includes(query));
      const isCleanEmpty = (isCleanVerbatim && card.getAttribute('data-clean-empty') === 'true');
      if (matchSpk && matchText && !isCleanEmpty) {{
        card.style.display = 'block';
      }} else {{
        card.style.display = 'none';
      }}
    }});
  }}
</script>
</body>
</html>
"""
        return html_doc

    def to_mom_html(self, title: str | None = None, meeting_date: str | None = None) -> str:
        """Export a professional, shareable Minutes of Meeting document.

        Unlike to_html() (an interactive transcript viewer), this is a static,
        print-friendly summary document meant to be shared with participants:
        attendees, executive summary, discussion by topic, decisions, and an
        action-item table — no raw dialogue.
        """
        import html as html_escape

        def esc(s: object) -> str:
            return html_escape.escape(str(s or ""))

        doc_title = title or self.title or "Meeting Minutes"
        ins = self.insights

        attendee_names: list[str] = []
        seen: set[str] = set()
        for spk in self.speakers:
            name = spk.name or spk.id
            if name not in seen:
                seen.add(name)
                attendee_names.append(name)

        total_min = int(self.duration // 60)
        hrs, mins = divmod(total_min, 60)
        duration_str = f"{hrs} hr {mins} min" if hrs > 0 else f"{mins} min"

        meta_parts = []
        if meeting_date:
            meta_parts.append(esc(meeting_date))
        meta_parts.append(duration_str)
        meta_parts.append(f"{len(attendee_names)} attendee{'s' if len(attendee_names) != 1 else ''}")
        meta_line = " &nbsp;&bull;&nbsp; ".join(meta_parts)

        attendees_html = "".join(f'<span class="attendee-chip">{esc(n)}</span>' for n in attendee_names) or (
            '<span class="muted">No attendees recorded</span>'
        )

        summary_html = ""
        if ins and ins.summary:
            summary_html = f"<section><h2>Executive Summary</h2><p>{esc(ins.summary)}</p></section>"

        discussion_html = ""
        if ins and ins.discussion_points:
            items = "".join(
                f'<div class="discussion-item"><h3>{esc(dp.topic)}</h3><p>{esc(dp.summary)}</p></div>'
                for dp in ins.discussion_points
            )
            discussion_html = f"<section><h2>Discussion Summary</h2>{items}</section>"
        elif ins and ins.topics:
            # Older meetings extracted before per-topic discussion notes existed.
            tags = "".join(f'<span class="topic-tag">{esc(t)}</span>' for t in ins.topics)
            discussion_html = f'<section><h2>Topics Discussed</h2><div class="topic-tags">{tags}</div></section>'

        decisions_html = ""
        if ins and ins.decisions:
            items = "".join(f"<li>{esc(d)}</li>" for d in ins.decisions)
            decisions_html = f'<section><h2>Decisions</h2><ul class="decision-list">{items}</ul></section>'

        actions_html = ""
        if ins and ins.action_items:
            rows = "".join(
                f"<tr><td>{esc(a.task)}</td><td>{esc(a.assignee) if a.assignee else '&mdash;'}</td>"
                f"<td>{esc(a.due_date) if a.due_date else '&mdash;'}</td></tr>"
                for a in ins.action_items
            )
            actions_html = (
                '<section><h2>Action Items</h2><table class="action-table">'
                "<thead><tr><th>Task</th><th>Owner</th><th>Due Date</th></tr></thead>"
                f"<tbody>{rows}</tbody></table></section>"
            )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{esc(doc_title)} — Minutes of Meeting</title>
<style>
  :root {{ color-scheme: light; }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 0; background: #f4f5f7; color: #1f2328;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.55;
  }}
  .page {{
    max-width: 780px; margin: 32px auto; background: #ffffff;
    border: 1px solid #e2e4e8; border-radius: 12px; padding: 48px 56px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  }}
  header.doc-header {{ border-bottom: 3px solid #1f2328; padding-bottom: 20px; margin-bottom: 28px; }}
  .kicker {{
    text-transform: uppercase; letter-spacing: 0.08em; font-size: 11px;
    font-weight: 700; color: #6b7280; margin-bottom: 6px;
  }}
  h1.doc-title {{ font-size: 26px; font-weight: 700; margin: 0 0 8px 0; color: #111827; }}
  .meta-line {{ font-size: 13px; color: #6b7280; }}
  .attendees-block {{ margin-top: 16px; }}
  .attendees-label {{
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.06em; color: #6b7280; margin-bottom: 8px;
  }}
  .attendee-chip {{
    display: inline-block; background: #eef2ff; color: #3730a3; border: 1px solid #c7d2fe;
    border-radius: 999px; padding: 3px 12px; font-size: 12.5px; font-weight: 600; margin: 0 6px 6px 0;
  }}
  section {{ margin: 30px 0; }}
  h2 {{
    font-size: 15px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em;
    color: #111827; border-left: 4px solid #2563eb; padding-left: 10px; margin: 0 0 14px 0;
  }}
  section > p {{ margin: 0; font-size: 14px; color: #1f2328; }}
  .discussion-item {{ margin-bottom: 16px; }}
  .discussion-item:last-child {{ margin-bottom: 0; }}
  .discussion-item h3 {{ font-size: 14px; font-weight: 700; margin: 0 0 4px 0; color: #111827; }}
  .discussion-item p {{ margin: 0; font-size: 13.5px; color: #374151; }}
  .topic-tags {{ display: flex; flex-wrap: wrap; gap: 6px; }}
  .topic-tag {{
    background: #f3f4f6; color: #374151; border: 1px solid #e5e7eb;
    border-radius: 6px; padding: 3px 10px; font-size: 12.5px; font-weight: 500;
  }}
  ul.decision-list {{ margin: 0; padding-left: 20px; }}
  ul.decision-list li {{ font-size: 14px; font-weight: 600; color: #111827; margin-bottom: 8px; }}
  ul.decision-list li:last-child {{ margin-bottom: 0; }}
  table.action-table {{ width: 100%; border-collapse: collapse; font-size: 13.5px; margin-top: 4px; }}
  table.action-table th {{
    text-align: left; background: #f9fafb; color: #6b7280; text-transform: uppercase;
    font-size: 11px; letter-spacing: 0.04em; padding: 8px 10px; border-bottom: 2px solid #e5e7eb;
  }}
  table.action-table td {{ padding: 10px; border-bottom: 1px solid #f0f1f3; color: #1f2328; vertical-align: top; }}
  table.action-table tr:last-child td {{ border-bottom: none; }}
  .muted {{ color: #9ca3af; font-size: 13px; font-style: italic; }}
  footer.doc-footer {{
    margin-top: 40px; padding-top: 16px; border-top: 1px solid #e5e7eb;
    font-size: 11px; color: #9ca3af; text-align: center;
  }}
  @media print {{
    body {{ background: #fff; }}
    .page {{ box-shadow: none; border: none; margin: 0; max-width: 100%; padding: 0; }}
    section {{ page-break-inside: avoid; }}
  }}
</style>
</head>
<body>
  <div class="page">
    <header class="doc-header">
      <div class="kicker">Minutes of Meeting</div>
      <h1 class="doc-title">{esc(doc_title)}</h1>
      <div class="meta-line">{meta_line}</div>
      <div class="attendees-block">
        <div class="attendees-label">Attendees</div>
        {attendees_html}
      </div>
    </header>
    {summary_html}
    {discussion_html}
    {decisions_html}
    {actions_html}
    <footer class="doc-footer">Generated by Polyphon AI &middot; Local Meeting Intelligence</footer>
  </div>
</body>
</html>
"""
