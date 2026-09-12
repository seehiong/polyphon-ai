// Polyphon Studio VoiceDB Biometric Profile Management

// Load VoiceDB Tab
async function loadVoiceDB() {
  const grid = document.getElementById('voicedb-grid');
  grid.innerHTML = '<p class="text-xs text-slate-400">Loading VoiceDB profiles...</p>';
  try {
    const res = await fetch('/api/speakers');
    const speakers = await res.json();
    grid.innerHTML = '';
    if (speakers.length === 0) {
      grid.innerHTML = '<p class="text-xs text-slate-500">No enrolled speaker profiles found in VoiceDB.</p>';
      return;
    }
    speakers.forEach(spk => {
      const card = document.createElement('div');
      card.className = 'bg-dark-900 border border-slate-800 rounded-2xl p-5 space-y-3';

      const samples = spk.samples || [];
      const samplesHtml = samples.length === 0
        ? '<p class="text-[11px] text-slate-500">No enrollment sample on file.</p>'
        : samples.map((sample, idx) => {
            const source = sample.source ? escapeHtml(sample.source) : `Sample ${idx + 1}`;
            const when = sample.added_at ? formatMeetingDate(Math.floor(new Date(sample.added_at).getTime() / 1000)) : '';
            const audioUrl = `/api/speakers/${encodeURIComponent(spk.id)}/samples/${idx}/audio`;
            return `
              <div class="flex items-center justify-between gap-2 text-[11px]">
                <span class="text-slate-400 truncate" title="${source}${when ? ' • ' + escapeHtml(when) : ''}">${source}${when ? ` <span class="text-slate-600">• ${escapeHtml(when)}</span>` : ''}</span>
                <div class="flex items-center gap-1 flex-shrink-0">
                  <audio controls preload="none" class="h-7 w-40" src="${audioUrl}"
                    onerror="this.outerHTML='<span class=&quot;text-amber-500&quot;>⚠️ unavailable</span>'"></audio>
                  ${actionIconButton({
                    icon: '🗑️',
                    tooltip: "Delete this sample and re-derive the voiceprint from what's left",
                    variant: 'danger',
                    onClick: "deleteSample(this.getAttribute('data-spk-id'), this.getAttribute('data-idx'))",
                    dataAttrs: { 'spk-id': spk.id, idx: idx },
                  })}
                </div>
              </div>
            `;
          }).join('');

      card.innerHTML = `
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-2.5">
            <span class="w-8 h-8 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 flex items-center justify-center font-bold text-xs">
              ${spk.name.charAt(0).toUpperCase()}
            </span>
            <div>
              <h4 class="font-semibold text-sm text-white">${spk.name}</h4>
              <span class="text-[10px] font-mono text-slate-500">${spk.id}</span>
            </div>
          </div>
          <div class="flex items-center gap-1">
            ${actionIconButton({
              icon: '✏️',
              tooltip: 'Rename speaker',
              onClick: "promptRenameSpeaker(this.getAttribute('data-spk-id'), this.getAttribute('data-spk-name'))",
              dataAttrs: { 'spk-id': spk.id, 'spk-name': spk.name },
            })}
            ${actionIconButton({
              icon: '🗑️',
              tooltip: 'Delete speaker profile',
              variant: 'danger',
              onClick: "deleteSpeaker(this.getAttribute('data-spk-id'))",
              dataAttrs: { 'spk-id': spk.id },
            })}
          </div>
        </div>
        <div class="space-y-1.5 pt-2 border-t border-slate-800">
          <div class="flex items-center justify-between">
            <span class="text-[11px] text-slate-500">Enrollment Samples</span>
            <span class="text-[10px] font-mono text-slate-600">${samples.length}</span>
          </div>
          ${samplesHtml}
        </div>
      `;
      grid.appendChild(card);
    });
  } catch (err) {
    grid.innerHTML = `<p class="text-xs text-red-400">Failed to load VoiceDB: ${err.message}</p>`;
  }
}

async function deleteSpeaker(spkId) {
  if (!confirm(`Delete speaker profile "${spkId}" from VoiceDB?`)) return;
  try {
    await fetch(`/api/speakers/${spkId}`, { method: 'DELETE' });
    loadVoiceDB();
    checkSystemStatus();
  } catch (err) {
    alert('Failed to delete speaker: ' + err.message);
  }
}

async function deleteSample(spkId, sampleIndex) {
  if (!confirm('Delete this sample? The voiceprint will be re-derived from whatever samples remain.')) return;
  try {
    const res = await fetch(`/api/speakers/${encodeURIComponent(spkId)}/samples/${sampleIndex}`, { method: 'DELETE' });
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    if (data.remaining_samples > 0 && !data.embedding_reconciled) {
      showNotification('⚠️ Sample deleted, but the voiceprint could not be re-derived (remaining sample audio unavailable). Consider re-enrolling.', 'error');
    } else if (data.embedding_reconciled) {
      showNotification(`🗑️ Sample deleted — voiceprint re-derived from ${data.embedding_reconciled_from} remaining sample(s).`);
    } else {
      showNotification('🗑️ Sample deleted. No samples remain — the existing voiceprint was left as-is.', 'error');
    }
    loadVoiceDB();
  } catch (err) {
    alert('Failed to delete sample: ' + err.message);
  }
}

let pendingRenameSpeakerId = null;

function promptRenameSpeaker(spkId, currentName = '') {
  if (!spkId) return;
  pendingRenameSpeakerId = spkId;
  const modal = document.getElementById('modal-rename-speaker');
  const idLabel = document.getElementById('modal-rename-speaker-id');
  const input = document.getElementById('modal-rename-speaker-input');

  if (modal && input) {
    if (idLabel) idLabel.textContent = spkId;
    input.value = currentName;
    modal.classList.remove('hidden');
    setTimeout(() => {
      input.focus();
      input.select();
    }, 50);
  } else {
    const entered = prompt(`Enter new name for speaker "${spkId}":`, currentName);
    if (entered && entered.trim()) {
      executeRenameSpeakerDirect(spkId, entered.trim());
    }
  }
}

function closeRenameSpeakerModal() {
  pendingRenameSpeakerId = null;
  const modal = document.getElementById('modal-rename-speaker');
  if (modal) modal.classList.add('hidden');
}

async function executeRenameSpeaker() {
  const spkId = pendingRenameSpeakerId;
  const input = document.getElementById('modal-rename-speaker-input');
  const newName = input ? input.value.trim() : '';
  closeRenameSpeakerModal();
  if (!spkId || !newName) return;
  await executeRenameSpeakerDirect(spkId, newName);
}

async function executeRenameSpeakerDirect(spkId, newName) {
  try {
    const res = await fetch(`/api/speakers/${encodeURIComponent(spkId)}/rename`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: newName })
    });
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `HTTP ${res.status}`);
    }
    loadVoiceDB();
  } catch (err) {
    alert('Failed to rename speaker: ' + err.message);
  }
}
