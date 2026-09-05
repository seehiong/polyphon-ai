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
          <button onclick="deleteSpeaker('${spk.id}')" class="text-xs text-red-400 hover:text-red-300 p-1.5 rounded-lg hover:bg-red-500/10 transition">
            🗑️
          </button>
        </div>
        <div class="text-[11px] text-slate-400 pt-2 border-t border-slate-800">
          <span class="text-slate-500">Sample Audio:</span> ${spk.sample_audio ? spk.sample_audio.split('/').pop() : 'N/A'}
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
