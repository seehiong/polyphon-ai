// Polyphon Studio Meeting Archive & Batch Processing


async function processMeetingById(meetingId) {
  if (!meetingId) return;
  switchTab('studio');
  document.getElementById('upload-panel').classList.add('hidden');
  document.getElementById('results-panel').classList.add('hidden');
  document.getElementById('progress-panel').classList.remove('hidden');

  const formData = new FormData();
  formData.append('saved_filename', meetingId);
  formData.append('diarize', 'true');
  formData.append('identify', 'true');
  formData.append('infer_names', 'true');
  formData.append('auto_enroll', 'true');
  formData.append('summarize', 'true');
  formData.append('clean_fillers', document.getElementById('opt-clean-fillers')?.checked ? 'true' : 'false');
  formData.append('language', document.getElementById('opt-language')?.value || 'en');

  try {
    const res = await fetch('/api/process', { method: 'POST', body: formData });
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `HTTP ${res.status}`);
    }
    const jobData = await res.json();
    connectSSE(jobData.job_id);
  } catch (err) {
    alert('Failed to start meeting processing: ' + err.message);
    document.getElementById('progress-panel').classList.add('hidden');
    if (currentMeetingPayload) {
      document.getElementById('results-panel').classList.remove('hidden');
    } else {
      document.getElementById('upload-panel').classList.remove('hidden');
    }
  }
}

function processCurrentMeeting() {
  if (currentLoadedMeetingId) {
    processMeetingById(currentLoadedMeetingId);
  }
}

function processLastLiveMeeting() {
  if (lastRecordedLiveSession) {
    processMeetingById(lastRecordedLiveSession);
  }
}

async function processArchiveMeeting(meetingId, isLiveOnly = false) {
  if (!meetingId) return;
  const msg = isLiveOnly
    ? `Run full offline AI processing (Whisper large-v3 + Diarization + VoiceDB + LLM) on live session "${meetingId}"?`
    : `Re-process meeting "${meetingId}" with full AI pipeline (Whisper large-v3 + Diarization + VoiceDB + LLM)? This will update transcripts and insights.`;
  if (!confirm(msg)) return;
  processMeetingById(meetingId);
}


let cachedArchiveItems = [];

// Format unix timestamp in seconds to human-readable date & time
function formatMeetingDate(timestampSec, itemId = '') {
  let ts = timestampSec;
  if (!ts && itemId && itemId.startsWith('live_')) {
    const rawTs = parseInt(itemId.replace('live_', ''), 10);
    if (!isNaN(rawTs) && rawTs > 1000000000) {
      ts = rawTs;
    }
  }
  if (!ts) return '';
  const date = new Date(ts * 1000);
  if (isNaN(date.getTime())) return '';
  const dateStr = date.toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    year: 'numeric'
  });
  const timeStr = date.toLocaleTimeString(undefined, {
    hour: '2-digit',
    minute: '2-digit'
  });
  return `${dateStr} • ${timeStr}`;
}

// Client-Side Search, Filter & Sorting for Archived Meetings
function filterArchiveMeetings() {
  const query = (document.getElementById('archive-search')?.value || '').trim().toLowerCase();
  const sortBy = document.getElementById('archive-sort')?.value || 'newest';
  const grid = document.getElementById('archive-grid');
  if (!grid) return;

  if (!cachedArchiveItems.length) {
    grid.innerHTML = '<p class="text-xs text-slate-500">No archived meetings found in outputs directory.</p>';
    return;
  }

  let filtered = cachedArchiveItems.filter(item => {
    if (!query) return true;
    const title = (item.title || '').toLowerCase();
    const id = (item.id || '').toLowerCase();
    const snippet = (item.summary_snippet || '').toLowerCase();
    return title.includes(query) || id.includes(query) || snippet.includes(query);
  });

  // Sort based on selected option
  filtered.sort((a, b) => {
    const timeA = a.created_at || (a.id && a.id.startsWith('live_') ? parseInt(a.id.replace('live_', ''), 10) : 0) || 0;
    const timeB = b.created_at || (b.id && b.id.startsWith('live_') ? parseInt(b.id.replace('live_', ''), 10) : 0) || 0;
    const durA = a.duration || 0;
    const durB = b.duration || 0;
    const titleA = (a.title || '').toLowerCase();
    const titleB = (b.title || '').toLowerCase();

    switch (sortBy) {
      case 'oldest':
        return timeA - timeB;
      case 'duration_desc':
        return durB - durA;
      case 'duration_asc':
        return durA - durB;
      case 'title_asc':
        return titleA.localeCompare(titleB);
      case 'newest':
      default:
        return timeB - timeA;
    }
  });

  renderArchiveCards(filtered, query);
}

function renderArchiveCards(items, query = '') {
  const grid = document.getElementById('archive-grid');
  grid.innerHTML = '';
  if (items.length === 0) {
    grid.innerHTML = query
      ? `<p class="text-xs text-slate-500">No meetings match "${escapeHtml(query)}". Try another search keyword.</p>`
      : '<p class="text-xs text-slate-500">No archived meetings found in outputs directory.</p>';
    return;
  }
  items.forEach(item => {
    const card = document.createElement('div');
    card.className = 'bg-dark-900 border border-slate-800 hover:border-slate-700 rounded-2xl p-5 space-y-3 transition shadow-sm';
    const m = Math.floor(item.duration / 60);
    const s = Math.floor(item.duration % 60);
    const dateFormatted = formatMeetingDate(item.created_at, item.id);
    const dateBadge = dateFormatted
      ? `<span class="inline-flex items-center gap-1 text-[10px] px-2 py-0.5 rounded bg-slate-800/90 text-slate-300 border border-slate-700/60 font-medium" title="Meeting recorded / saved at ${escapeHtml(dateFormatted)}"><span>📅</span> ${escapeHtml(dateFormatted)}</span>`
      : '';
    const liveBadge = item.is_live_only
      ? `<span class="inline-block text-[10px] px-2 py-0.5 rounded bg-red-950/80 text-red-300 border border-red-800/60 font-medium">🔴 Live Only</span>`
      : '';
    const cmpBadge = item.has_comparison
      ? `<span class="inline-block text-[10px] px-2 py-0.5 rounded bg-purple-950/80 text-purple-300 border border-purple-800/60 font-medium">⚖️ Compared</span>`
      : '';

    const processBtn = item.is_live_only
      ? `<button onclick="processArchiveMeeting(this.getAttribute('data-meeting-id'), true)" data-meeting-id="${escapeHtml(item.id)}" title="Process with full offline AI Pipeline" class="text-[11px] px-2.5 py-1 rounded bg-emerald-600 hover:bg-emerald-500 text-white font-medium transition flex items-center gap-1 shadow"><span>🚀</span> Process</button>`
      : `<button onclick="processArchiveMeeting(this.getAttribute('data-meeting-id'), false)" data-meeting-id="${escapeHtml(item.id)}" title="Re-process meeting with full AI Pipeline (Whisper + Diarization + VoiceDB + LLM)" class="text-[11px] px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition flex items-center gap-1"><span>🔄</span> Reprocess</button>`;

    const compareBtn = item.has_comparison
      ? `<button onclick="openCompareModal(this.getAttribute('data-meeting-id'))" data-meeting-id="${escapeHtml(item.id)}" title="Compare Live vs Processed" class="text-[11px] px-2.5 py-1 rounded bg-purple-900/60 hover:bg-purple-900/90 text-purple-300 border border-purple-700 transition flex items-center gap-1"><span>⚖️</span> Compare</button>`
      : '';

    card.innerHTML = `
      <div class="flex items-start justify-between gap-2">
        <div class="flex-1 min-w-0">
          <div class="flex items-center gap-1.5 group min-w-0">
            <h4 class="font-semibold text-sm text-white capitalize truncate cursor-pointer hover:text-blue-300 transition" onclick="promptRenameMeeting(this.getAttribute('data-meeting-id'), this.getAttribute('data-title'))" data-meeting-id="${escapeHtml(item.id)}" data-title="${escapeHtml(item.title)}" title="Click to rename meeting">${escapeHtml(item.title)}</h4>
            <button onclick="promptRenameMeeting(this.getAttribute('data-meeting-id'), this.getAttribute('data-title'))" data-meeting-id="${escapeHtml(item.id)}" data-title="${escapeHtml(item.title)}" title="Rename meeting title" class="text-xs text-slate-500 hover:text-blue-400 p-0.5 rounded transition opacity-70 hover:opacity-100 flex-shrink-0">
              ✏️
            </button>
          </div>
          <div class="flex flex-wrap items-center gap-1.5 mt-1.5">
            ${dateBadge}
            <span class="inline-block text-[10px] px-2 py-0.5 rounded bg-slate-800 text-slate-400 font-mono">⏱️ ${m}m ${s}s</span>
            ${liveBadge}
            ${cmpBadge}
          </div>
        </div>
        <button onclick="promptDeleteMeeting(this.getAttribute('data-meeting-id'))" data-meeting-id="${escapeHtml(item.id)}" title="Delete meeting" class="text-xs text-red-400 hover:text-red-300 p-1.5 rounded-lg hover:bg-red-500/10 transition flex-shrink-0">
          🗑️
        </button>
      </div>
      <p class="text-xs text-slate-400 line-clamp-3">${escapeHtml(item.summary_snippet)}</p>
      <div class="flex items-center justify-between pt-2 border-t border-slate-800/80">
        <span class="text-[11px] text-slate-500">👥 ${item.speakers_count} • 💬 ${item.turns_count} turns</span>
        <div class="flex items-center gap-1.5">
          ${processBtn}
          ${compareBtn}
          <button onclick="openMeetingInStudio(this.getAttribute('data-meeting-id'))" data-meeting-id="${escapeHtml(item.id)}" class="text-[11px] px-2.5 py-1 rounded bg-blue-600 hover:bg-blue-500 text-white font-medium transition flex items-center gap-1">
            <span>▶️</span> Play in Studio
          </button>
          ${item.html_url ? `<a href="${item.html_url}" target="_blank" class="text-[11px] px-2 py-1 rounded bg-blue-900/40 hover:bg-blue-900/60 text-blue-300 border border-blue-800 transition">View Report</a>` : ''}
          ${item.json_url ? `<a href="${item.json_url}" download class="text-[11px] px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition">JSON</a>` : ''}
        </div>
      </div>
    `;
    grid.appendChild(card);
  });
}

// Load Archive Tab
async function loadArchive() {
  const grid = document.getElementById('archive-grid');
  grid.innerHTML = '<p class="text-xs text-slate-400">Loading archived meetings...</p>';
  try {
    const res = await fetch('/api/meetings');
    cachedArchiveItems = await res.json();
    filterArchiveMeetings();
  } catch (err) {
    grid.innerHTML = `<p class="text-xs text-red-400">Failed to load archive: ${err.message}</p>`;
  }
}

// In-app Modal Delete Handlers (Bypasses any browser dialog suppression)
let pendingDeleteMeetingId = null;

function promptDeleteMeeting(meetingId) {
  if (!meetingId) {
    console.warn('[Polyphon Archive] promptDeleteMeeting called with empty ID');
    showNotification('No meeting ID provided to delete', 'warning');
    return;
  }
  pendingDeleteMeetingId = meetingId;
  const modal = document.getElementById('modal-confirm-delete');
  const label = document.getElementById('modal-delete-meeting-id');
  if (modal && label) {
    label.textContent = meetingId;
    modal.classList.remove('hidden');
  } else {
    // Fallback if modal DOM element is not present
    if (confirm(`Delete meeting "${meetingId}" and all its reports?`)) {
      executeDeleteMeeting();
    }
  }
}

function closeDeleteModal() {
  pendingDeleteMeetingId = null;
  const modal = document.getElementById('modal-confirm-delete');
  if (modal) modal.classList.add('hidden');
}

async function executeDeleteMeeting() {
  const meetingId = pendingDeleteMeetingId;
  closeDeleteModal();
  if (!meetingId) return;

  try {
    console.log(`[Polyphon Archive] Executing DELETE for meeting: ${meetingId}`);
    const res = await fetch(`/api/meetings/${encodeURIComponent(meetingId)}`, { method: 'DELETE' });
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `HTTP ${res.status}`);
    }
    showNotification(`🗑️ Deleted meeting.`);
    if (currentLoadedMeetingId === meetingId) {
      currentMeetingPayload = null;
      currentLoadedMeetingId = null;
      activeResultData = null;
      if (typeof getActivePlayer === 'function') {
        const p = getActivePlayer();
        if (p) p.pause();
      }
      document.getElementById('results-panel')?.classList.add('hidden');
      document.getElementById('upload-panel')?.classList.remove('hidden');
    }
    loadArchive();
    if (typeof checkCurrentJob === 'function') checkCurrentJob();
  } catch (err) {
    console.error('[Polyphon Archive] Delete failed:', err);
    alert('Failed to delete meeting: ' + err.message);
  }
}

// Backward-compatibility alias
function deleteMeeting(meetingId) {
  promptDeleteMeeting(meetingId);
}

// In-app Modal Rename Handlers
let pendingRenameMeetingId = null;

function promptRenameMeeting(meetingId, currentTitle = '') {
  if (!meetingId) {
    console.warn('[Polyphon Archive] promptRenameMeeting called with empty ID');
    return;
  }
  pendingRenameMeetingId = meetingId;
  const modal = document.getElementById('modal-rename-meeting');
  const idLabel = document.getElementById('modal-rename-meeting-id');
  const input = document.getElementById('modal-rename-input');
  const fallbackTitle = currentTitle || meetingId.replace(/_/g, ' ');

  if (modal && input) {
    if (idLabel) idLabel.textContent = meetingId;
    input.value = fallbackTitle;
    modal.classList.remove('hidden');
    setTimeout(() => {
      input.focus();
      input.select();
    }, 50);
  } else {
    const entered = prompt(`Enter new title for meeting "${meetingId}":`, fallbackTitle);
    if (entered && entered.trim()) {
      executeRenameMeetingDirect(meetingId, entered.trim());
    }
  }
}

function closeRenameModal() {
  pendingRenameMeetingId = null;
  const modal = document.getElementById('modal-rename-meeting');
  if (modal) modal.classList.add('hidden');
}

async function executeRenameMeeting() {
  const meetingId = pendingRenameMeetingId;
  const input = document.getElementById('modal-rename-input');
  const newTitle = input ? input.value.trim() : '';
  closeRenameModal();
  if (!meetingId || !newTitle) return;
  await executeRenameMeetingDirect(meetingId, newTitle);
}

async function executeRenameMeetingDirect(meetingId, newTitle) {
  try {
    const res = await fetch(`/api/meetings/${encodeURIComponent(meetingId)}/rename`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title: newTitle })
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    showNotification(`✔ Renamed meeting to "${newTitle}"`, 'success');

    // Update in cachedArchiveItems
    const cached = cachedArchiveItems.find(it => it.id === meetingId || it.id === data.id);
    if (cached) {
      cached.title = newTitle;
      filterArchiveMeetings();
    } else {
      loadArchive();
    }

    // Update Studio header if current meeting is loaded
    if (currentLoadedMeetingId === meetingId || currentLoadedMeetingId === data.id) {
      const titleEl = document.getElementById('res-meeting-title');
      if (titleEl) titleEl.textContent = newTitle;
      if (currentMeetingPayload) {
        currentMeetingPayload.title = newTitle;
        if (currentMeetingPayload.result) currentMeetingPayload.result.title = newTitle;
      }
    }
  } catch (err) {
    console.error('[Polyphon Archive] Rename failed:', err);
    alert('Failed to rename meeting: ' + err.message);
  }
}

// Bind keyboard events for rename modal
document.addEventListener('DOMContentLoaded', () => {
  const renameInput = document.getElementById('modal-rename-input');
  if (renameInput) {
    renameInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        executeRenameMeeting();
      } else if (e.key === 'Escape') {
        closeRenameModal();
      }
    });
  }
});

async function openMeetingInStudio(meetingId) {
  try {
    const res = await fetch(`/api/meetings/${encodeURIComponent(meetingId)}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    switchTab('studio');
    document.getElementById('upload-panel').classList.add('hidden');
    document.getElementById('progress-panel').classList.add('hidden');
    const payload = {
      job_id: data.id || meetingId,
      title: data.title,
      result: data.result || data,
      media_url: data.media_url || `/media/${meetingId}.mp4`,
      html_report_url: data.html_report_url || `/reports/${meetingId}.html`
    };
    renderResultsUI(payload);
  } catch (err) {
    alert('Failed to load meeting: ' + err.message);
  }
}
