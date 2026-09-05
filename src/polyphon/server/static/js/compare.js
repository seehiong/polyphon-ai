// Polyphon Studio Comparative Benchmarking Modal (Live vs Processed)


// Comparison Modal Logic
async function openCompareModal(meetingId) {
  let data = null;
  let targetId = meetingId || currentLoadedMeetingId;
  if (currentMeetingPayload && currentMeetingPayload.job_id === targetId && currentMeetingPayload.result.live_segments) {
    data = currentMeetingPayload.result;
  } else {
    try {
      const res = await fetch(`/api/meetings/${targetId}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const meeting = await res.json();
      data = meeting.result || meeting;
    } catch (err) {
      alert('Failed to load meeting details: ' + err.message);
      return;
    }
  }

  if (!data || !data.live_segments || data.live_segments.length === 0) {
    alert('This meeting does not have live stream comparison data.');
    return;
  }

  populateCompareModal(data, targetId);
  document.getElementById('modal-compare').classList.remove('hidden');
}

function openCompareModalCurrent() {
  if (currentLoadedMeetingId) {
    openCompareModal(currentLoadedMeetingId);
  }
}

function closeCompareModal() {
  document.getElementById('modal-compare').classList.add('hidden');
}

function populateCompareModal(data, meetingId) {
  document.getElementById('compare-modal-title').textContent = `Compare: ${meetingId.replace(/_/g, ' ')}`;

  const liveSegs = data.live_segments || [];
  const procSegs = data.segments || [];

  // Word counts
  const countWords = (segs) => {
    return segs.reduce((acc, s) => {
      if (s.words && s.words.length > 0) return acc + s.words.length;
      return acc + (s.text || '').trim().split(/\s+/).filter(Boolean).length;
    }, 0);
  };
  const liveWords = countWords(liveSegs);
  const procWords = countWords(procSegs);

  // Distinct speakers
  const liveSpkSet = new Set(liveSegs.map(s => s.speaker_name || s.speaker_id).filter(Boolean));
  const procSpkSet = new Set((data.speakers || []).map(s => s.name || s.id).filter(Boolean));
  if (procSpkSet.size === 0) {
    procSegs.forEach(s => procSpkSet.add(s.speaker_name || s.speaker_id));
  }

  // Populate KPIs
  document.getElementById('cmp-turns-live').textContent = `${liveSegs.length} live`;
  document.getElementById('cmp-turns-proc').textContent = `${procSegs.length} proc`;
  document.getElementById('cmp-words-live').textContent = `${liveWords}`;
  document.getElementById('cmp-words-proc').textContent = `${procWords}`;
  document.getElementById('cmp-spk-live').textContent = `${liveSpkSet.size} live`;
  document.getElementById('cmp-spk-proc').textContent = `${procSpkSet.size} identified`;

  const intelBadge = document.getElementById('cmp-intel-badge');
  if (data.insights && (data.insights.summary || (data.insights.action_items && data.insights.action_items.length > 0))) {
    intelBadge.textContent = "Insights & Actions Generated";
    intelBadge.className = "px-2.5 py-0.5 rounded-full bg-emerald-950/80 text-emerald-300 border border-emerald-800 font-medium";
  } else {
    intelBadge.textContent = "Insights Pending";
    intelBadge.className = "px-2.5 py-0.5 rounded-full bg-amber-950/80 text-amber-300 border border-amber-800 font-medium";
  }

  document.getElementById('cmp-live-count').textContent = `${liveSegs.length} turns`;
  document.getElementById('cmp-proc-count').textContent = `${procSegs.length} turns`;

  // Render Left: Live Turns
  const liveContainer = document.getElementById('cmp-live-turns');
  liveContainer.innerHTML = '';
  if (liveSegs.length === 0) {
    liveContainer.innerHTML = '<p class="text-xs text-slate-500 italic p-4">No live segments recorded.</p>';
  } else {
    liveSegs.forEach(seg => {
      const card = document.createElement('div');
      card.className = 'p-3 rounded-xl bg-slate-900/90 border border-slate-800 hover:border-red-500/40 transition cursor-pointer text-xs space-y-1';
      const m = Math.floor(seg.start / 60);
      const s = Math.floor(seg.start % 60);
      const timeStr = `[${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}]`;
      card.onclick = () => {
        const p = getActivePlayer();
        if (p) {
          p.currentTime = seg.start;
          p.play().catch(()=>{});
        }
      };
      card.innerHTML = `
        <div class="flex items-center justify-between text-[11px]">
          <span class="font-semibold text-red-400 bg-red-950/60 px-2 py-0.5 rounded border border-red-800/60">
            ${escapeHtml(seg.speaker_name || seg.speaker_id || 'SPEAKER')}
          </span>
          <span class="font-mono text-slate-500">${timeStr}</span>
        </div>
        <p class="text-slate-300 leading-relaxed">${escapeHtml(seg.text)}</p>
      `;
      liveContainer.appendChild(card);
    });
  }

  // Render Right: Processed Turns
  const procContainer = document.getElementById('cmp-proc-turns');
  procContainer.innerHTML = '';
  if (procSegs.length === 0) {
    procContainer.innerHTML = '<p class="text-xs text-slate-500 italic p-4">No processed segments recorded.</p>';
  } else {
    procSegs.forEach(seg => {
      const card = document.createElement('div');
      card.className = 'p-3 rounded-xl bg-slate-900/90 border border-slate-800 hover:border-emerald-500/40 transition cursor-pointer text-xs space-y-1';
      const m = Math.floor(seg.start / 60);
      const s = Math.floor(seg.start % 60);
      const timeStr = `[${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}]`;
      card.onclick = () => {
        const p = getActivePlayer();
        if (p) {
          p.currentTime = seg.start;
          p.play().catch(()=>{});
        }
      };
      card.innerHTML = `
        <div class="flex items-center justify-between text-[11px]">
          <span class="font-semibold text-emerald-400 bg-emerald-950/60 px-2 py-0.5 rounded border border-emerald-800/60">
            ${escapeHtml(seg.speaker_name || seg.speaker_id || 'SPEAKER')}
          </span>
          <span class="font-mono text-slate-500">${timeStr}</span>
        </div>
        <p class="text-slate-300 leading-relaxed">${escapeHtml(seg.text)}</p>
      `;
      procContainer.appendChild(card);
    });
  }
}
