// Polyphon Studio Navigation & System Status

function switchTab(tabId) {
  ['studio', 'stream', 'archive', 'voicedb'].forEach(t => {
    const viewEl = document.getElementById(`view-${t}`);
    const tabEl = document.getElementById(`tab-${t}`);
    if (viewEl) viewEl.classList.add('hidden');
    if (tabEl) {
      tabEl.classList.remove('bg-blue-600', 'text-white', 'shadow');
      tabEl.classList.add('text-slate-400');
    }
  });

  const targetView = document.getElementById(`view-${tabId}`);
  const targetTab = document.getElementById(`tab-${tabId}`);
  if (targetView) targetView.classList.remove('hidden');
  if (targetTab) {
    targetTab.classList.add('bg-blue-600', 'text-white', 'shadow');
    targetTab.classList.remove('text-slate-400');
  }

  if (tabId === 'stream' && typeof checkMicSupport === 'function') checkMicSupport();
  if (tabId === 'archive' && typeof loadArchive === 'function') loadArchive();
  if (tabId === 'voicedb' && typeof loadVoiceDB === 'function') loadVoiceDB();
}

async function checkSystemStatus() {
  try {
    const res = await fetch('/api/system');
    const data = await res.json();
    const pill = document.getElementById('system-status-pill');
    const txt = document.getElementById('system-status-text');

    if (pill && txt) {
      if (data.llm && data.llm.connected) {
        pill.className = 'flex items-center gap-2 px-3 py-1.5 rounded-full text-xs bg-emerald-950/60 border border-emerald-800 text-emerald-300';
        txt.textContent = `🟢 ${data.llm.model} Connected`;
      } else {
        pill.className = 'flex items-center gap-2 px-3 py-1.5 rounded-full text-xs bg-amber-950/60 border border-amber-800 text-amber-300';
        txt.textContent = '🟡 LLM Offline (Port 8080)';
      }
    }
  } catch (err) {
    const txt = document.getElementById('system-status-text');
    if (txt) txt.textContent = '🔴 Server Disconnected';
  }
}

