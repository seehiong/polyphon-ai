// Polyphon Studio One-Click VoiceDB Enrollment from Transcript


function openEnrollModal(e, speakerId, currentName) {
  if (e) e.stopPropagation();
  currentEnrollSpeakerId = speakerId;
  const modal = document.getElementById('modal-enroll-speaker');
  const clusterBadge = document.getElementById('enroll-speaker-cluster');
  const statsBadge = document.getElementById('enroll-speaker-stats');
  const nameInput = document.getElementById('enroll-speaker-name');
  const errorBox = document.getElementById('enroll-error-box');

  if (errorBox) errorBox.classList.add('hidden');
  if (clusterBadge) clusterBadge.textContent = speakerId;

  const segs = (currentMeetingPayload?.result?.segments || []).filter(s => s.speaker_id === speakerId || s.speaker_name === speakerId);
  const liveSegs = (currentMeetingPayload?.result?.live_segments || []).filter(s => s.speaker_id === speakerId || s.speaker_name === speakerId);
  const count = Math.max(segs.length, liveSegs.length);
  if (statsBadge) statsBadge.textContent = `${count} dialogue turns in meeting`;

  if (nameInput) {
    nameInput.value = (currentName && !currentName.startsWith('SPEAKER_')) ? currentName : '';
  }
  const replaceCheckbox = document.getElementById('enroll-replace-checkbox');
  if (replaceCheckbox) replaceCheckbox.checked = false;

  if (modal) {
    modal.classList.remove('hidden');
    setTimeout(() => { if (nameInput) nameInput.focus(); }, 100);
  }
}

function closeEnrollModal() {
  const modal = document.getElementById('modal-enroll-speaker');
  if (modal) modal.classList.add('hidden');
  currentEnrollSpeakerId = null;
}

async function submitEnrollSpeaker() {
  const nameInput = document.getElementById('enroll-speaker-name');
  const errorBox = document.getElementById('enroll-error-box');
  const submitBtn = document.getElementById('btn-submit-enroll');
  const submitLabel = document.getElementById('btn-submit-enroll-label');

  const name = nameInput ? nameInput.value.trim() : '';
  if (!name) {
    if (errorBox) {
      errorBox.textContent = "Please enter a name for this speaker.";
      errorBox.classList.remove('hidden');
    }
    return;
  }

  const meetingId = (currentMeetingPayload?.result?.audio_path ? currentMeetingPayload.result.audio_path.split('/').pop().replace(/\.[^/.]+$/, "") : null)
    || (currentMeetingPayload?.filename ? currentMeetingPayload.filename.replace(/\.[^/.]+$/, "") : null)
    || (currentMeetingPayload?.id || currentLoadedMeetingId);

  if (!meetingId || !currentEnrollSpeakerId) {
    if (errorBox) {
      errorBox.textContent = "Meeting or speaker identifier is missing.";
      errorBox.classList.remove('hidden');
    }
    return;
  }

  submitBtn.disabled = true;
  submitBtn.className = "px-4 py-2 rounded-xl text-xs font-semibold bg-slate-800 text-slate-400 cursor-not-allowed shadow transition flex items-center gap-2";
  submitLabel.textContent = "Extracting Voiceprint...";

  const replaceCheckbox = document.getElementById('enroll-replace-checkbox');
  const replace = replaceCheckbox ? replaceCheckbox.checked : false;

  try {
    const resp = await fetch('/api/speakers/enroll_from_meeting', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        meeting_id: meetingId,
        speaker_id: currentEnrollSpeakerId,
        name: name,
        replace: replace
      })
    });

    const data = await resp.json();
    if (!resp.ok) {
      throw new Error(data.detail || "Failed to enroll speaker");
    }

    // Update active payload in memory
    if (data.meeting) {
      currentMeetingPayload.result = data.meeting;
      activeResultData = data.meeting;
    } else {
      (currentMeetingPayload?.result?.speakers || []).forEach(spk => {
        if (spk.id === currentEnrollSpeakerId || spk.name === currentEnrollSpeakerId) {
          spk.name = name;
        }
      });
      (currentMeetingPayload?.result?.segments || []).forEach(seg => {
        if (seg.speaker_id === currentEnrollSpeakerId || seg.speaker_name === currentEnrollSpeakerId) {
          seg.speaker_name = name;
        }
      });
      (currentMeetingPayload?.result?.live_segments || []).forEach(seg => {
        if (seg.speaker_id === currentEnrollSpeakerId || seg.speaker_name === currentEnrollSpeakerId) {
          seg.speaker_name = name;
        }
      });
    }

    // Re-render dialogue turns
    const currentSegments = currentTranscriptViewMode === 'live'
      ? (currentMeetingPayload.result.live_segments || [])
      : (currentMeetingPayload.result.segments || []);
    renderDialogueTurns(currentSegments, currentTranscriptViewMode === 'live');

    // Update title & badges with deduplicated speaker names
    const uniqueSpeakerNames = [...new Set((currentMeetingPayload.result.speakers || []).map(s => s.name))];
    const titleEl = document.getElementById('res-meeting-title');
    if (titleEl && currentMeetingPayload.job_id) {
      titleEl.textContent = `Meeting: ${uniqueSpeakerNames.join(', ')}`;
    }
    const spkBadge = document.getElementById('res-spk-badge');
    if (spkBadge) {
      spkBadge.textContent = `👥 ${uniqueSpeakerNames.length} Speaker${uniqueSpeakerNames.length === 1 ? '' : 's'}`;
    }

    // Re-render Talk Time Bar & Participant count
    const talktimeBar = document.getElementById('talktime-bar');
    if (talktimeBar) {
      talktimeBar.innerHTML = '';
      const spkDurations = {};
      const segs = (currentMeetingPayload.result.segments || currentMeetingPayload.result.live_segments || []);
      segs.forEach(seg => {
        const spkName = seg.speaker_name || seg.speaker_id;
        spkDurations[spkName] = (spkDurations[spkName] || 0) + (seg.end - seg.start);
      });
      const totalDur = currentMeetingPayload.result.duration || 1;
      uniqueSpeakerNames.forEach((name, idx) => {
        const spkDur = spkDurations[name] || 0;
        const pct = totalDur > 0 ? (spkDur / totalDur) * 100 : 0;
        if (pct > 0) {
          const segEl = document.createElement('div');
          segEl.style.width = `${pct}%`;
          segEl.style.backgroundColor = SPEAKER_COLORS[idx % SPEAKER_COLORS.length];
          segEl.title = `${name}: ${pct.toFixed(1)}%`;
          talktimeBar.appendChild(segEl);
        }
      });
    }
    const partCount = document.getElementById('talktime-participant-count');
    if (partCount) {
      partCount.textContent = `${uniqueSpeakerNames.length} Participant${uniqueSpeakerNames.length === 1 ? '' : 's'}`;
    }

    // Re-render speaker filter buttons (Grouped by unique speaker identity)
    const spkColorMap = {};
    (currentMeetingPayload.result.speakers || []).forEach((spk, idx) => {
      spkColorMap[spk.id] = SPEAKER_COLORS[idx % SPEAKER_COLORS.length];
    });
    const filterGroup = document.getElementById('speaker-filter-group');
    if (filterGroup) {
      filterGroup.innerHTML = `<button onclick="filterSpeaker('all', this)" class="spk-btn active px-3 py-1.5 rounded-lg text-xs font-medium bg-blue-600 text-white transition">All Speakers</button>`;
      const seenSpkNames = new Set();
      (currentMeetingPayload.result.speakers || []).forEach(spk => {
        if (seenSpkNames.has(spk.name)) return;
        seenSpkNames.add(spk.name);

        const wrap = document.createElement('div');
        wrap.className = 'inline-flex items-center rounded-lg bg-slate-800 border border-slate-700/80 overflow-hidden';

        const btn = document.createElement('button');
        btn.className = 'spk-btn px-3 py-1.5 text-xs font-medium text-slate-300 hover:text-white transition';
        btn.style.borderLeft = `4px solid ${spkColorMap[spk.id] || '#94a3b8'}`;
        btn.textContent = spk.name;
        btn.onclick = function() { filterSpeaker(spk.name, btn); };

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
    }

    const participantCount = document.getElementById('talktime-participant-count');
    if (participantCount && currentMeetingPayload.result.speakers) {
      participantCount.textContent = `${uniqueSpeakerNames.length} Participant${uniqueSpeakerNames.length === 1 ? '' : 's'}`;
    }

    checkSystemStatus();
    closeEnrollModal();
    const replaceNote = replace ? ' (previous voiceprint & samples replaced)' : '';
    showNotification(`🎉 Enrolled "${name}" into VoiceDB${replaceNote}! Future meetings will automatically recognize this voice.`);

  } catch (err) {
    console.error("Enroll error:", err);
    if (errorBox) {
      errorBox.textContent = err.message || "Enrollment failed.";
      errorBox.classList.remove('hidden');
    }
  } finally {
    submitBtn.disabled = false;
    submitBtn.className = "px-4 py-2 rounded-xl text-xs font-semibold bg-blue-600 hover:bg-blue-500 text-white shadow-lg transition flex items-center gap-2";
    submitLabel.textContent = "Extract & Enroll Profile";
  }
}
