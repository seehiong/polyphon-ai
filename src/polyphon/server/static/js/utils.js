// Polyphon Studio Utility Functions

function escapeHtml(str) {
  return (str || '').replace(/[&<>"']/g, m => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#039;'
  }[m]));
}

// Shared icon-button markup for the small rename/delete controls repeated across
// Archive meeting cards and VoiceDB speaker cards, so tooltip/spacing/hover styling
// can't silently drift between the two (e.g. one growing a tooltip, the other not).
// Params go through data-* attributes rather than interpolated into the onclick
// string, so values containing quotes can't break the generated markup.
function actionIconButton({ icon, tooltip, onClick, variant = 'default', dataAttrs = {} }) {
  const variantClasses = variant === 'danger'
    ? 'text-red-400 hover:text-red-300 hover:bg-red-500/10'
    : 'text-slate-400 hover:text-white hover:bg-slate-700/60';
  const dataStr = Object.entries(dataAttrs)
    .map(([k, v]) => `data-${k}="${escapeHtml(String(v))}"`)
    .join(' ');
  return `<button onclick="${onClick}" ${dataStr} title="${escapeHtml(tooltip)}" class="text-xs p-1.5 rounded-lg ${variantClasses} transition">${icon}</button>`;
}

function hashString(str) {
  let hash = 0;
  for (let i = 0; i < (str || '').length; i++) {
    hash = (hash << 5) - hash + str.charCodeAt(i);
    hash |= 0;
  }
  return hash;
}

function showNotification(message, type = 'success') {
  let toast = document.getElementById('studio-toast');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'studio-toast';
    toast.className = 'fixed bottom-6 right-6 z-50 px-4 py-3 rounded-2xl shadow-2xl border text-xs font-semibold flex items-center gap-2 transition-all duration-300 transform translate-y-2 opacity-0 pointer-events-none';
    document.body.appendChild(toast);
  }
  toast.className = `fixed bottom-6 right-6 z-50 px-4 py-3 rounded-2xl shadow-2xl border text-xs font-semibold flex items-center gap-2 transition-all duration-300 transform translate-y-0 opacity-100 ${
    type === 'error' ? 'bg-red-950/90 border-red-800 text-red-200' : 'bg-emerald-950/90 border-emerald-800 text-emerald-200'
  }`;
  toast.textContent = message;
  setTimeout(() => {
    toast.classList.remove('opacity-100', 'translate-y-0');
    toast.classList.add('opacity-0', 'translate-y-2');
  }, 4000);
}

