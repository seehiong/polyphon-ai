// Polyphon Studio Audio/Video Player, Karaoke, Upload & Results UI

// File Upload Handler
async function handleFileSelected(file) {
  if (!file) return;
  document.getElementById('file-title').textContent = `Uploading: ${file.name}...`;
  document.getElementById('file-subtitle').textContent = `${(file.size / (1024*1024)).toFixed(1)} MB`;

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/api/upload', { method: 'POST', body: formData });
    const data = await res.json();
    currentUploadedFile = data;
    document.getElementById('file-title').textContent = file.name;
    document.getElementById('file-subtitle').textContent = `Ready to process • Duration: ${Math.round(data.duration)}s`;
    document.getElementById('btn-process').disabled = false;
  } catch (err) {
    alert('Upload failed: ' + err.message);
  }
}

// Check Current Job on load and periodically
async function checkCurrentJob() {
  try {
    const res = await fetch('/api/jobs/current');
    const data = await res.json();
    currentActiveJob = data;
    const banner = document.getElementById('job-banner');
    const dot = document.getElementById('banner-dot');
    const txt = document.getElementById('banner-text');
    const btn = document.getElementById('btn-banner-action');

    if (data.status === 'processing') {
      banner.classList.remove('hidden');
      dot.className = 'w-2.5 h-2.5 rounded-full bg-blue-500 animate-pulse';
      txt.innerHTML = `<strong>Engine Busy:</strong> ${data.filename} • ${data.stage_name} (${Math.round(data.percent)}%) • ${data.details || ''}`;
      btn.textContent = 'View Progress';
      btn.className = 'px-3 py-1 rounded-lg text-xs font-semibold bg-blue-600 hover:bg-blue-500 text-white transition';

      // If on studio tab and upload panel is still visible, resume progress view automatically
      if (document.getElementById('results-panel').classList.contains('hidden') && !document.getElementById('view-studio').classList.contains('hidden')) {
        document.getElementById('upload-panel').classList.add('hidden');
        document.getElementById('progress-panel').classList.remove('hidden');
        updateProgressUI(data);
        if (!activeEventSource) {
          connectSSE(data.job_id);
        }
      }
    } else if (data.status === 'completed' && data.result) {
      banner.classList.remove('hidden');
      dot.className = 'w-2.5 h-2.5 rounded-full bg-emerald-500';
      const m = Math.floor(data.result.duration / 60);
      const s = Math.floor(data.result.duration % 60);
      txt.innerHTML = `<strong>Latest Meeting Ready:</strong> ${data.filename} (${m}m ${s}s) • ${data.result.speakers.length} Speakers • ${data.details || 'Completed'}`;
      btn.textContent = '▶️ Open in Studio';
      btn.className = 'px-3 py-1 rounded-lg text-xs font-semibold bg-emerald-600 hover:bg-emerald-500 text-white transition';
    } else {
      banner.classList.add('hidden');
    }
  } catch (e) {
    console.warn('Could not query current job:', e);
  }
}

function handleBannerAction() {
  if (!currentActiveJob) return;
  switchTab('studio');
  if (currentActiveJob.status === 'processing') {
    document.getElementById('upload-panel').classList.add('hidden');
    document.getElementById('results-panel').classList.add('hidden');
    document.getElementById('progress-panel').classList.remove('hidden');
    updateProgressUI(currentActiveJob);
    connectSSE(currentActiveJob.job_id);
  } else if (currentActiveJob.status === 'completed') {
    document.getElementById('upload-panel').classList.add('hidden');
    document.getElementById('progress-panel').classList.add('hidden');
    renderResultsUI(currentActiveJob);
  }
}

function resetToUpload() {
  const p = getActivePlayer();
  if (p) p.pause();
  currentActiveCard = null;
  turnCardIndex = [];
  toggleAutoScroll(true);
  document.getElementById('results-panel').classList.add('hidden');
  document.getElementById('progress-panel').classList.add('hidden');
  document.getElementById('upload-panel').classList.remove('hidden');
  currentUploadedFile = null;
  document.getElementById('file-input').value = '';
  document.getElementById('file-title').textContent = 'Drop your meeting recording here';
  document.getElementById('file-subtitle').textContent = 'Supports MP4, WAV, MP3, M4A, FLAC, MKV';
  document.getElementById('btn-process').disabled = true;
}

function connectSSE(jobId) {
  if (activeEventSource) {
    activeEventSource.close();
  }
  activeEventSource = new EventSource(`/api/jobs/${jobId}/events`);
  activeEventSource.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    currentActiveJob = payload;
    updateProgressUI(payload);

    if (payload.status === 'completed') {
      activeEventSource.close();
      activeEventSource = null;
      setTimeout(() => {
        document.getElementById('progress-panel').classList.add('hidden');
        renderResultsUI(payload);
        checkCurrentJob();
      }, 600);
    } else if (payload.status === 'failed') {
      activeEventSource.close();
      activeEventSource = null;
      alert('Processing failed: ' + (payload.error || payload.details));
      resetToUpload();
      checkCurrentJob();
    }
  };
}

// Start Processing Pipeline
async function startProcessing() {
  if (!currentUploadedFile) return;

  document.getElementById('upload-panel').classList.add('hidden');
  document.getElementById('progress-panel').classList.remove('hidden');
  document.getElementById('results-panel').classList.add('hidden');

  const formData = new FormData();
  formData.append('saved_filename', currentUploadedFile.saved_filename);
  formData.append('diarize', document.getElementById('opt-diarize').checked);
  formData.append('identify', document.getElementById('opt-identify').checked);
  formData.append('infer_names', document.getElementById('opt-infer-names').checked);
  formData.append('auto_enroll', document.getElementById('opt-auto-enroll').checked);
  formData.append('summarize', document.getElementById('opt-summarize').checked);
  formData.append('clean_fillers', document.getElementById('opt-clean-fillers')?.checked || false);
  formData.append('language', document.getElementById('opt-language')?.value || 'en');

  const res = await fetch('/api/process', { method: 'POST', body: formData });
  const jobData = await res.json();
  connectSSE(jobData.job_id);
}

function updateProgressUI(payload) {
  document.getElementById('progress-stage-title').textContent = payload.stage_name || 'Processing...';
  document.getElementById('progress-percent').textContent = `${Math.round(payload.percent)}%`;
  document.getElementById('progress-bar-fill').style.width = `${payload.percent}%`;
  if (payload.details) {
    document.getElementById('progress-details').textContent = payload.details;
  }
}


function getActivePlayer() {
  return isVideoMode ? document.getElementById('video-player') : document.getElementById('audio-player');
}

function toggleAutoScroll(forceState) {
  if (typeof forceState === 'boolean') {
    isAutoScrollEnabled = forceState;
  } else {
    isAutoScrollEnabled = !isAutoScrollEnabled;
  }
  const label = document.getElementById('autoscroll-label');
  const dot = document.getElementById('autoscroll-dot');
  const btn = document.getElementById('btn-autoscroll');
  if (!label || !dot || !btn) return;
  if (isAutoScrollEnabled) {
    label.textContent = 'Auto-scroll: ON';
    dot.className = 'w-2 h-2 rounded-full bg-emerald-400 animate-pulse';
    btn.className = 'px-3 py-1.5 rounded-xl text-xs font-semibold bg-blue-600/15 text-blue-400 border border-blue-500/30 hover:bg-blue-600/25 whitespace-nowrap transition flex items-center gap-1.5 shadow-sm';
    if (currentActiveCard) {
      scrollCardIntoView(currentActiveCard, true);
    }
  } else {
    label.textContent = 'Auto-scroll: OFF';
    dot.className = 'w-2 h-2 rounded-full bg-slate-500';
    btn.className = 'px-3 py-1.5 rounded-xl text-xs font-semibold bg-slate-900 text-slate-400 border border-slate-800 hover:bg-slate-800 whitespace-nowrap transition flex items-center gap-1.5 shadow-sm';
  }
}

const STANDARD_FILLERS = new Set([
  'um', 'uh', 'er', 'ah', 'erm', 'umm', 'uhh', 'err', 'ahh',
  'hmm', 'mhm', 'mm', 'mm-hmm', 'mmhmm', 'uh-huh', 'uhhuh', 'uh-oh', 'uhoh', 'eh', 'huh', 'hum'
]);

function isFillerWordToken(w) {
  if (w.is_filler === true) return true;
  if (!w.word) return false;
  const cleaned = w.word.replace(/^[.,!?:;"'()\[\]{}<>\-~—/\\]+|[.,!?:;"'()\[\]{}<>\-~—/\\]+$/g, '').toLowerCase();
  return STANDARD_FILLERS.has(cleaned);
}

function toggleCleanVerbatim(forceState) {
  if (typeof forceState === 'boolean') {
    isCleanVerbatimEnabled = forceState;
  } else {
    isCleanVerbatimEnabled = !isCleanVerbatimEnabled;
  }

  const container = document.getElementById('dialogue-turns-list');
  const btn = document.getElementById('btn-clean-verbatim');
  const label = document.getElementById('clean-verbatim-label');
  const icon = document.getElementById('clean-verbatim-icon');

  if (!btn || !label) return;

  const totalFillers = container ? container.querySelectorAll('.k-word.filler').length : 0;

  if (isCleanVerbatimEnabled) {
    if (container) container.classList.add('clean-verbatim-active');
    btn.className = 'px-3 py-1.5 rounded-xl text-xs font-semibold bg-amber-500/15 text-amber-400 border border-amber-500/30 hover:bg-amber-500/25 whitespace-nowrap transition flex items-center gap-1.5 shadow-sm';
    label.textContent = totalFillers > 0 ? `Clean Verbatim (${totalFillers} hidden)` : 'Clean Verbatim: ON';
    if (icon) icon.textContent = '✨';
  } else {
    if (container) container.classList.remove('clean-verbatim-active');
    btn.className = 'px-3 py-1.5 rounded-xl text-xs font-semibold bg-slate-800/80 text-slate-300 border border-slate-700/80 hover:bg-slate-700 hover:text-white whitespace-nowrap transition flex items-center gap-1.5 shadow-sm';
    label.textContent = 'Clean Verbatim';
    if (icon) icon.textContent = '✨';
  }
}

function scrollCardIntoView(card, force = false) {
  if (!isAutoScrollEnabled && !force) return;
  const container = document.getElementById('dialogue-turns-list');
  if (!container || !card) return;

  const containerTop = container.scrollTop;
  const containerHeight = container.clientHeight;
  const cardTop = card.offsetTop - container.offsetTop;
  const cardHeight = card.clientHeight;

  const targetScroll = cardTop - (containerHeight / 2) + (cardHeight / 2);
  container.scrollTo({
    top: Math.max(0, targetScroll),
    behavior: 'smooth'
  });
}

function toggleVideoMode() {
  const vWrap = document.getElementById('video-wrapper');
  const vEl = document.getElementById('video-player');
  const aEl = document.getElementById('audio-player');
  const btnLabel = document.getElementById('btn-toggle-video-label');
  
  const currentTime = activePlayer ? activePlayer.currentTime : 0;
  const wasPlaying = activePlayer && !activePlayer.paused;
  
  if (activePlayer) activePlayer.pause();
  
  isVideoMode = !isVideoMode;
  if (isVideoMode) {
    vWrap.classList.remove('hidden');
    aEl.classList.add('hidden');
    activePlayer = vEl;
    btnLabel.textContent = 'Audio Only';
  } else {
    vWrap.classList.add('hidden');
    aEl.classList.remove('hidden');
    activePlayer = aEl;
    btnLabel.textContent = 'Show Video';
  }
  
  activePlayer.currentTime = currentTime;
  attachPlayerListeners(activePlayer);
  if (wasPlaying) activePlayer.play().catch(()=>{});
}

function setPlaybackRate(rate, btn) {
  currentPlaybackRate = parseFloat(rate);
  const a = document.getElementById('audio-player');
  const v = document.getElementById('video-player');
  if (a) a.playbackRate = currentPlaybackRate;
  if (v) v.playbackRate = currentPlaybackRate;
  if (activePlayer) activePlayer.playbackRate = currentPlaybackRate;

  document.querySelectorAll('.speed-btn').forEach(b => {
    b.className = 'speed-btn px-1.5 py-0.5 rounded text-slate-400 hover:text-white transition';
  });
  if (btn) {
    btn.className = 'speed-btn px-1.5 py-0.5 rounded bg-blue-600 text-white shadow transition font-bold';
  }
}

function attachPlayerListeners(player) {
  player.playbackRate = currentPlaybackRate;
  player.onplay = () => {
    player.playbackRate = currentPlaybackRate;
  };
  player.ontimeupdate = () => {
    const currentTime = player.currentTime;
    let activeCard = null;

    // Fast lookup via pre-indexed card list (avoids DOM tree querying on every tick)
    for (let i = 0; i < turnCardIndex.length; i++) {
      const item = turnCardIndex[i];
      if (currentTime >= item.start && currentTime <= item.end) {
        activeCard = item.card;
        break;
      }
    }

    // Trigger turn change & centering exactly ONCE per turn transition!
    if (activeCard && activeCard !== currentActiveCard) {
      if (currentActiveCard) {
        currentActiveCard.classList.remove('active');
        currentActiveCard.querySelectorAll('.k-word').forEach(wEl => {
          wEl.classList.remove('current');
          const wEnd = parseFloat(wEl.getAttribute('data-end'));
          if (currentTime > wEnd) {
            wEl.classList.add('past');
            wEl.classList.remove('future');
          } else {
            wEl.classList.remove('past');
            wEl.classList.add('future');
          }
        });
      }
      activeCard.classList.add('active');
      currentActiveCard = activeCard;
      scrollCardIntoView(activeCard);
    }

    // Word-level Karaoke illumination on the currently active card
    if (currentActiveCard) {
      currentActiveCard.querySelectorAll('.k-word').forEach(wEl => {
        const wStart = parseFloat(wEl.getAttribute('data-start'));
        const wEnd = parseFloat(wEl.getAttribute('data-end'));
        if (currentTime >= wStart && currentTime <= wEnd) {
          wEl.classList.add('current');
          wEl.classList.remove('past', 'future');
        } else if (currentTime > wEnd) {
          wEl.classList.add('past');
          wEl.classList.remove('current', 'future');
        } else {
          wEl.classList.add('future');
          wEl.classList.remove('current', 'past');
        }
      });
    }
  };
}

// Render Completed Meeting Results
function renderResultsUI(payload) {
  activeResultData = payload.result;
  currentLoadedMeetingId = payload.job_id;
  currentMeetingPayload = payload;
  currentTranscriptViewMode = 'processed';
  currentActiveCard = null;
  document.getElementById('results-panel').classList.remove('hidden');

  // Process/Reprocess button in Studio toolbar
  const processCurrentBtn = document.getElementById('btn-process-current');
  const isLiveOnly = !payload.result.insights || payload.result.is_live_only;
  if (processCurrentBtn) {
    processCurrentBtn.classList.remove('hidden');
    if (isLiveOnly) {
      processCurrentBtn.innerHTML = '<span>🚀</span> Process Offline';
      processCurrentBtn.className = 'px-3.5 py-1.5 rounded-lg text-xs font-semibold bg-emerald-600 hover:bg-emerald-500 text-white transition flex items-center gap-1.5 shadow';
      processCurrentBtn.title = 'Run full offline AI processing (Whisper large-v3 + Diarization + VoiceDB + LLM)';
    } else {
      processCurrentBtn.innerHTML = '<span>🔄</span> Reprocess';
      processCurrentBtn.className = 'px-3.5 py-1.5 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 transition flex items-center gap-1.5';
      processCurrentBtn.title = 'Re-run full AI pipeline (Whisper + Diarization + VoiceDB + LLM) on this recording';
    }
  }

  // Show/Hide Compare button and transcript toggle
  const compareBtn = document.getElementById('btn-compare-transcripts');
  const toggleGroup = document.getElementById('transcript-mode-toggle');
  const hasComparison = Boolean((payload.result.live_segments && payload.result.live_segments.length > 0) || payload.result.has_comparison);

  if (compareBtn) {
    if (hasComparison) {
      compareBtn.classList.remove('hidden');
    } else {
      compareBtn.classList.add('hidden');
    }
  }
  if (toggleGroup) {
    if (hasComparison) {
      toggleGroup.classList.remove('hidden');
      updateTranscriptToggleButtons('processed');
    } else {
      toggleGroup.classList.add('hidden');
    }
  }

  // Title & Badges
  const uniqueSpeakerNames = [...new Set((payload.result.speakers || []).map(s => s.name))];
  const meetingTitle = payload.title || payload.result?.title || (payload.job_id ? payload.job_id.replace(/_/g, ' ') : `Meeting: ${uniqueSpeakerNames.join(', ')}`);
  document.getElementById('res-meeting-title').textContent = meetingTitle;
  const m = Math.floor(payload.result.duration / 60);
  const s = Math.floor(payload.result.duration % 60);
  document.getElementById('res-dur-badge').textContent = `⏱️ ${m}m ${s.toString().padStart(2, '0')}s`;
  document.getElementById('res-spk-badge').textContent = `👥 ${uniqueSpeakerNames.length} Speaker${uniqueSpeakerNames.length === 1 ? '' : 's'}`;
  document.getElementById('res-turns-badge').textContent = `💬 ${payload.result.segments.length} Turns`;

  let totalFillers = 0;
  (payload.result.segments || []).forEach(seg => {
    if (seg.words) {
      seg.words.forEach(w => {
        if (isFillerWordToken(w)) totalFillers++;
      });
    }
  });
  const fillersBadge = document.getElementById('res-fillers-badge');
  if (fillersBadge) {
    if (totalFillers > 0) {
      fillersBadge.textContent = `✨ ${totalFillers} Filler${totalFillers === 1 ? '' : 's'}`;
      fillersBadge.classList.remove('hidden');
    } else {
      fillersBadge.classList.add('hidden');
    }
  }

  // Export Links
  const stem = payload.job_id || 'meeting';
  const htmlBtn = document.getElementById('btn-export-html');
  const mdBtn = document.getElementById('btn-export-md');
  const jsonBtn = document.getElementById('btn-export-json');
  const srtBtn = document.getElementById('btn-export-srt');

  htmlBtn.href = payload.html_report_url || `/reports/${stem}.html`;
  mdBtn.href = `/reports/${stem}.md`;
  jsonBtn.href = `/reports/${stem}.json`;
  srtBtn.href = `/reports/${stem}.srt`;

  const hasSegments = payload.result && payload.result.segments && payload.result.segments.length > 0;
  [htmlBtn, mdBtn, jsonBtn, srtBtn].forEach(el => {
    if (!el) return;
    if (!hasSegments) {
      el.classList.add('opacity-40', 'pointer-events-none');
      el.setAttribute('title', 'No transcript data available to export');
    } else {
      el.classList.remove('opacity-40', 'pointer-events-none');
    }
  });

  // Setup Unified Media Player
  const vEl = document.getElementById('video-player');
  const aEl = document.getElementById('audio-player');
  const vWrap = document.getElementById('video-wrapper');
  const toggleBtn = document.getElementById('btn-toggle-video');
  const btnLabel = document.getElementById('toggle-video-label');

  const mediaUrl = payload.media_url || '';
  const isVideoFile = /\.(mp4|webm|mov|mkv)$/i.test(mediaUrl);

  vEl.src = mediaUrl;
  aEl.src = mediaUrl;

  if (isVideoFile) {
    isVideoMode = true;
    vWrap.classList.remove('hidden');
    aEl.classList.add('hidden');
    toggleBtn.classList.remove('hidden');
    btnLabel.textContent = 'Audio Only';
    activePlayer = vEl;
  } else {
    isVideoMode = false;
    vWrap.classList.add('hidden');
    aEl.classList.remove('hidden');
    toggleBtn.classList.add('hidden');
    activePlayer = aEl;
  }

  activePlayer.load();
  attachPlayerListeners(activePlayer);

  // Speaker Map & Colors
  const spkColorMap = {};
  uniqueSpeakerNames.forEach((name, idx) => {
    spkColorMap[name] = SPEAKER_COLORS[idx % SPEAKER_COLORS.length];
  });
  payload.result.speakers.forEach(spk => {
    const name = spk.name || spk.id;
    spkColorMap[spk.id] = spkColorMap[name] || SPEAKER_COLORS[0];
  });

  // Talk Time Bar
  const talktimeBar = document.getElementById('talktime-bar');
  talktimeBar.innerHTML = '';
  const spkDurations = {};
  payload.result.segments.forEach(seg => {
    const spkName = seg.speaker_name || seg.speaker_id;
    spkDurations[spkName] = (spkDurations[spkName] || 0) + (seg.end - seg.start);
  });
  uniqueSpeakerNames.forEach((name, idx) => {
    const spkDur = spkDurations[name] || 0;
    const pct = payload.result.duration > 0 ? (spkDur / payload.result.duration) * 100 : 0;
    if (pct > 0) {
      const segEl = document.createElement('div');
      segEl.style.width = `${pct}%`;
      segEl.style.backgroundColor = SPEAKER_COLORS[idx % SPEAKER_COLORS.length];
      segEl.title = `${name}: ${pct.toFixed(1)}%`;
      talktimeBar.appendChild(segEl);
    }
  });
  document.getElementById('talktime-participant-count').textContent = `${uniqueSpeakerNames.length} Participant${uniqueSpeakerNames.length === 1 ? '' : 's'}`;

  // Speaker Filter Tabs (Grouped by unique speaker identity)
  const filterGroup = document.getElementById('speaker-filter-group');
  filterGroup.innerHTML = `<button onclick="filterSpeaker('all', this)" class="spk-btn active px-3 py-1.5 rounded-lg text-xs font-medium bg-blue-600 text-white transition">All Speakers</button>`;
  
  const seenSpkNames = new Set();
  (payload.result.speakers || []).forEach(spk => {
    if (seenSpkNames.has(spk.name)) return;
    seenSpkNames.add(spk.name);

    const wrap = document.createElement('div');
    wrap.className = 'inline-flex items-center rounded-lg bg-slate-800 border border-slate-700/80 overflow-hidden';

    const btn = document.createElement('button');
    btn.className = 'spk-btn px-3 py-1.5 text-xs font-medium text-slate-300 hover:text-white transition';
    btn.style.borderLeft = `4px solid ${spkColorMap[spk.name] || spkColorMap[spk.id] || '#94a3b8'}`;
    btn.textContent = spk.name;
    btn.onclick = function() { filterSpeaker(spk.name, this); };

    const enrollBtn = document.createElement('button');
    enrollBtn.type = 'button';
    enrollBtn.className = 'px-2 py-1.5 text-[10px] text-slate-400 hover:text-white hover:bg-slate-700 transition border-l border-slate-700/60';
    enrollBtn.title = `Enroll ${spk.name} in VoiceDB`;
    enrollBtn.innerHTML = '👤';
    enrollBtn.onclick = function(e) { openEnrollModal(e, spk.id, spk.name); };

    wrap.appendChild(btn);
    wrap.appendChild(enrollBtn);
    filterGroup.appendChild(wrap);
  });

  // Executive Summary & Insights
  const ins = payload.result.insights || {};
  document.getElementById('res-summary-text').textContent = ins.summary || "No executive summary generated.";

  // Decisions
  const decList = document.getElementById('res-decisions-list');
  decList.innerHTML = '';
  (ins.decisions || []).forEach(d => {
    const li = document.createElement('li');
    li.textContent = d;
    decList.appendChild(li);
  });

  // Action Items
  const actList = document.getElementById('res-actions-list');
  actList.innerHTML = '';
  (ins.action_items || []).forEach((act, idx) => {
    const div = document.createElement('div');
    div.className = 'flex items-center gap-2.5 p-2 rounded-lg bg-slate-950/60 border border-slate-800/80 transition';
    div.innerHTML = `
      <input type="checkbox" id="act_${idx}" onchange="toggleActionItem(this)" class="w-3.5 h-3.5 rounded text-amber-500 bg-slate-900 border-slate-700 cursor-pointer">
      <label for="act_${idx}" class="text-slate-300 cursor-pointer flex-1 transition">${escapeHtml(act.task)}</label>
      ${act.assignee ? `<span class="text-[10px] px-2 py-0.5 rounded bg-blue-900/60 text-blue-300 border border-blue-800 shrink-0">${escapeHtml(act.assignee)}</span>` : ''}
    `;
    actList.appendChild(div);
  });

  // Topics
  const topList = document.getElementById('res-topics-list');
  topList.innerHTML = '';
  (ins.topics || []).forEach(t => {
    const tag = document.createElement('span');
    tag.className = 'px-2.5 py-1 rounded-full text-xs font-medium bg-purple-950/60 border border-purple-800 text-purple-300';
    tag.textContent = t;
    topList.appendChild(tag);
  });

  // Initial dialogue turns rendering (processed by default)
  renderDialogueTurns(payload.result.segments || [], false);
  toggleCleanVerbatim(isCleanVerbatimEnabled);
}

// Render Dialogue Turns (shared for Processed and Live Stream modes)
function renderDialogueTurns(segments, isLiveMode = false) {
  const turnsList = document.getElementById('dialogue-turns-list');
  turnsList.innerHTML = '';
  turnCardIndex = [];

  if (!segments || segments.length === 0) {
    turnsList.innerHTML = '<p class="text-xs text-slate-500 italic p-4">No dialogue turns recorded in this view.</p>';
    return;
  }

  const uniqueNames = [...new Set((activeResultData?.speakers || []).map(s => s.name || s.id))];
  const spkColorMap = {};
  uniqueNames.forEach((name, idx) => {
    spkColorMap[name] = SPEAKER_COLORS[idx % SPEAKER_COLORS.length];
  });
  (activeResultData?.speakers || []).forEach(spk => {
    const name = spk.name || spk.id;
    spkColorMap[spk.id] = spkColorMap[name] || SPEAKER_COLORS[0];
  });

  // Coalesce adjacent segments by the same speaker identity
  const displaySegments = [];
  (segments || []).forEach(seg => {
    const spkName = seg.speaker_name || seg.speaker_id;
    if (displaySegments.length > 0) {
      const prev = displaySegments[displaySegments.length - 1];
      const prevName = prev.speaker_name || prev.speaker_id;
      const gap = seg.start - prev.end;
      if (spkName === prevName && gap <= 3.0) {
        prev.end = Math.max(prev.end, seg.end);
        prev.text = (prev.text + ' ' + seg.text).trim();
        if (prev.words && seg.words) {
          prev.words = prev.words.concat(seg.words);
        }
        return;
      }
    }
    displaySegments.push({
      ...seg,
      words: seg.words ? [...seg.words] : []
    });
  });

  displaySegments.forEach(seg => {
    const card = document.createElement('div');
    card.className = `turn-card p-3.5 rounded-xl bg-slate-950 border ${isLiveMode ? 'border-red-900/40' : 'border-slate-800/90'} transition cursor-pointer hover:border-slate-700`;
    card.setAttribute('data-speaker', seg.speaker_id);
    card.setAttribute('data-speaker-name', seg.speaker_name || seg.speaker_id);
    card.setAttribute('data-start', seg.start);
    card.setAttribute('data-end', seg.end);

    card.onclick = (e) => {
      if (e.target.closest('.btn-enroll-speaker')) return;
      const p = getActivePlayer();
      const wTarget = e.target.closest('.k-word');
      if (wTarget) {
        const wStart = parseFloat(wTarget.getAttribute('data-start'));
        p.currentTime = Math.max(0, wStart - 0.05);
        p.play().catch(()=>{});
      } else {
        p.currentTime = seg.start;
        p.play().catch(()=>{});
      }

      if (currentActiveCard && currentActiveCard !== card) {
        currentActiveCard.classList.remove('active');
      }
      card.classList.add('active');
      currentActiveCard = card;
      scrollCardIntoView(card, true);
    };

    const mStart = Math.floor(seg.start / 60);
    const sStart = Math.floor(seg.start % 60);
    const timeStr = `[${mStart.toString().padStart(2, '0')}:${sStart.toString().padStart(2, '0')}]`;
    const color = spkColorMap[seg.speaker_name] || spkColorMap[seg.speaker_id] || (isLiveMode ? '#ef4444' : '#3b82f6');

    let wordsHtml = '';
    if (seg.words && seg.words.length > 0) {
      wordsHtml = seg.words.map(w => {
        const safe = escapeHtml(w.word);
        const isFiller = isFillerWordToken(w);
        const fillerClass = isFiller ? ' filler' : '';
        const fillerAttr = isFiller ? ' data-filler="true"' : '';
        return `<span class="k-word future${fillerClass}" data-start="${w.start}" data-end="${w.end}"${fillerAttr}>${safe}</span>`;
      }).join(' ');
    } else {
      wordsHtml = `<span class="turn-text text-xs text-slate-300 leading-relaxed">${escapeHtml(seg.text)}</span>`;
    }

    const modeBadge = isLiveMode ? '<span class="text-[9px] px-1.5 py-0.5 rounded bg-red-950 text-red-400 border border-red-800 font-mono">LIVE</span>' : '';
    const spkLabel = escapeHtml(seg.speaker_name || seg.speaker_id);
    const spkId = escapeHtml(seg.speaker_id || '');

    card.innerHTML = `
      <div class="flex items-center justify-between gap-2 mb-1.5">
        <div class="flex items-center gap-2">
          <span class="text-[11px] font-semibold px-2 py-0.5 rounded" style="background-color: ${color}20; color: ${color}; border: 1px solid ${color}40;">
            ${spkLabel}
          </span>
          <span class="text-[11px] font-mono text-slate-500">${timeStr}</span>
          ${modeBadge}
        </div>
        <button type="button" onclick="openEnrollModal(event, '${spkId}', '${spkLabel}')" title="Enroll speaker voiceprint into VoiceDB" class="btn-enroll-speaker text-[10px] px-2 py-0.5 rounded bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-white border border-slate-800 transition flex items-center gap-1 shrink-0">
          <span>👤</span> Enroll
        </button>
      </div>
      <div class="turn-body text-xs text-slate-300 leading-relaxed">${wordsHtml}</div>
    `;
    turnsList.appendChild(card);
    turnCardIndex.push({
      card: card,
      start: seg.start,
      end: seg.end
    });
  });
}

// Toggle between Processed and Live Stream transcripts in Studio view
function switchTranscriptView(mode) {
  if (!currentMeetingPayload || !currentMeetingPayload.result) return;
  currentTranscriptViewMode = mode;
  updateTranscriptToggleButtons(mode);

  const segments = mode === 'live'
    ? (currentMeetingPayload.result.live_segments || [])
    : (currentMeetingPayload.result.segments || []);

  renderDialogueTurns(segments, mode === 'live');
  toggleCleanVerbatim(isCleanVerbatimEnabled);
}

function updateTranscriptToggleButtons(mode) {
  const btnProc = document.getElementById('btn-view-proc');
  const btnLive = document.getElementById('btn-view-live');
  if (!btnProc || !btnLive) return;

  if (mode === 'live') {
    btnProc.className = "px-2.5 py-1 rounded-lg font-medium text-slate-400 hover:text-slate-200 transition flex items-center gap-1";
    btnLive.className = "px-2.5 py-1 rounded-lg font-semibold bg-red-600 text-white transition flex items-center gap-1 shadow";
  } else {
    btnProc.className = "px-2.5 py-1 rounded-lg font-semibold bg-blue-600 text-white transition flex items-center gap-1 shadow";
    btnLive.className = "px-2.5 py-1 rounded-lg font-medium text-slate-400 hover:text-slate-200 transition flex items-center gap-1";
  }
}

// Interactive Transcript Filtering
function filterSpeaker(spkId, btn) {
  activeSpeakerFilter = spkId;
  document.querySelectorAll('.spk-btn').forEach(b => {
    b.classList.remove('active', 'bg-blue-600', 'text-white');
    b.classList.add('bg-slate-800', 'text-slate-300');
  });
  if (btn) {
    btn.classList.add('active', 'bg-blue-600', 'text-white');
    btn.classList.remove('bg-slate-800', 'text-slate-300');
  }
  applyTranscriptFilters();
}

function filterTranscript() {
  applyTranscriptFilters();
}

function applyTranscriptFilters() {
  const searchInput = document.getElementById('transcript-search');
  const q = searchInput ? searchInput.value.toLowerCase() : '';
  turnCardIndex.forEach(item => {
    const card = item.card;
    const spkId = card.getAttribute('data-speaker') || '';
    const spkName = card.getAttribute('data-speaker-name') || '';
    const bodyEl = card.querySelector('.turn-body') || card.querySelector('.turn-text');
    const text = bodyEl ? bodyEl.textContent.toLowerCase() : '';
    const matchSpk = (
      activeSpeakerFilter === 'all' ||
      spkId === activeSpeakerFilter ||
      spkName.toLowerCase() === activeSpeakerFilter.toLowerCase()
    );
    const matchText = (!q || text.includes(q));
    card.style.display = (matchSpk && matchText) ? 'block' : 'none';
  });
}

// Action Items Interactive Toggle
function toggleActionItem(chk) {
  const lbl = chk.parentElement.querySelector('label');
  if (lbl) {
    if (chk.checked) {
      lbl.classList.add('line-through', 'text-slate-500', 'opacity-60');
      lbl.classList.remove('text-slate-300');
    } else {
      lbl.classList.remove('line-through', 'text-slate-500', 'opacity-60');
      lbl.classList.add('text-slate-300');
    }
  }
}

// Copy Action Items Checklist to Clipboard
function copyTasksToClipboard() {
  if (!activeResultData || !activeResultData.insights || !activeResultData.insights.action_items) {
    showNotification('No action items available to copy', 'warning');
    return;
  }
  const items = activeResultData.insights.action_items;
  if (!items.length) {
    showNotification('No action items found', 'info');
    return;
  }
  const lines = items.map((act, idx) => {
    const chk = document.getElementById(`act_${idx}`);
    const isDone = chk && chk.checked;
    const mark = isDone ? '[x]' : '[ ]';
    const who = act.assignee ? ` (@${act.assignee})` : '';
    return `- ${mark} ${act.task}${who}`;
  });
  const text = `# Action Items\n\n${lines.join('\n')}\n`;
  navigator.clipboard.writeText(text).then(() => {
    showNotification(`Copied ${items.length} action items to clipboard!`, 'success');
  }).catch(() => {
    showNotification('Failed to copy to clipboard', 'error');
  });
}

// Copy Full Formatted Transcript to Clipboard
function copyTranscriptToClipboard() {
  if (!activeResultData) {
    showNotification('No transcript available to copy', 'warning');
    return;
  }
  const segments = (currentTranscriptViewMode === 'live' && activeResultData.live_segments)
    ? activeResultData.live_segments
    : (activeResultData.segments || []);

  if (!segments.length) {
    showNotification('No dialogue turns to copy', 'info');
    return;
  }

  const lines = [];
  segments.forEach(seg => {
    const spkName = seg.speaker_name || seg.speaker || 'Speaker';
    const m = Math.floor(seg.start / 60);
    const s = Math.floor(seg.start % 60);
    const timeStr = `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
    let text = (seg.text || '').trim();
    if (isCleanVerbatimEnabled && seg.words && seg.words.length > 0) {
      const nonFillerWords = seg.words.filter(w => !isFillerWordToken(w)).map(w => w.word);
      text = nonFillerWords.join(' ').trim();
    }
    if (text) {
      lines.push(`[${timeStr}] ${spkName}: ${text}`);
    }
  });

  const title = document.getElementById('res-meeting-title')?.textContent || 'Meeting Transcript';
  const text = `# ${title}\n\n${lines.join('\n\n')}\n`;
  navigator.clipboard.writeText(text).then(() => {
    const modeStr = isCleanVerbatimEnabled ? ' (Clean Verbatim)' : '';
    showNotification(`Copied ${lines.length} dialogue turns${modeStr} to clipboard!`, 'success');
  }).catch(() => {
    showNotification('Failed to copy to clipboard', 'error');
  });
}

function deleteCurrentMeeting() {
  const meetingId = currentLoadedMeetingId || (currentMeetingPayload ? (currentMeetingPayload.meeting_id || currentMeetingPayload.id || currentMeetingPayload.job_id) : null);
  if (!meetingId) {
    showNotification('No meeting is currently loaded to delete.', 'warning');
    return;
  }
  if (typeof promptDeleteMeeting === 'function') {
    promptDeleteMeeting(meetingId);
  } else if (confirm(`Delete meeting "${meetingId}" and all its reports? This cannot be undone.`)) {
    fetch(`/api/meetings/${encodeURIComponent(meetingId)}`, { method: 'DELETE' })
      .then(res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        showNotification('🗑️ Deleted meeting.');
        currentMeetingPayload = null;
        currentLoadedMeetingId = null;
        activeResultData = null;
        if (typeof getActivePlayer === 'function') {
          const p = getActivePlayer();
          if (p) p.pause();
        }
        document.getElementById('results-panel')?.classList.add('hidden');
        document.getElementById('upload-panel')?.classList.remove('hidden');
        if (typeof loadArchive === 'function') loadArchive();
        if (typeof checkCurrentJob === 'function') checkCurrentJob();
        switchTab('archive');
      })
      .catch(err => alert('Failed to delete meeting: ' + err.message));
  }
}

function renameCurrentMeeting() {
  const meetingId = currentLoadedMeetingId || (currentMeetingPayload ? (currentMeetingPayload.meeting_id || currentMeetingPayload.id || currentMeetingPayload.job_id) : null);
  if (!meetingId) {
    showNotification('No meeting currently loaded to rename', 'warning');
    return;
  }
  const titleEl = document.getElementById('res-meeting-title');
  const currentTitle = titleEl ? titleEl.textContent : meetingId.replace(/_/g, ' ');
  if (typeof promptRenameMeeting === 'function') {
    promptRenameMeeting(meetingId, currentTitle);
  }
}

