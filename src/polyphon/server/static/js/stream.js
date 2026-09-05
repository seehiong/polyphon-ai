// Polyphon Studio Live Streaming & Simulation Controller

function checkMicSupport() {
  const isSecure = window.isSecureContext || window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
  const hasMediaDevices = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  const warnBanner = document.getElementById('mic-insecure-warning');
  const originLabel = document.getElementById('current-origin-label');
  if (originLabel) originLabel.textContent = window.location.origin;

  if (!isSecure || !hasMediaDevices) {
    if (warnBanner && (streamSource === 'mic' || streamSource === 'dual')) warnBanner.classList.remove('hidden');
  } else {
    if (warnBanner) warnBanner.classList.add('hidden');
  }
}

function setStreamSource(source) {
  streamSource = source;
  const btnMic = document.getElementById('src-btn-mic');
  const btnDual = document.getElementById('src-btn-dual');
  const btnSim = document.getElementById('src-btn-sim');
  const simSelector = document.getElementById('sim-file-selector');
  const dualTip = document.getElementById('dual-guide-tip');
  const warnBanner = document.getElementById('mic-insecure-warning');
  const btnLabel = document.getElementById('stream-btn-label');
  const viewTitle = document.getElementById('stream-view-title');
  const viewSubtitle = document.getElementById('stream-view-subtitle');
  const levelLabel = document.getElementById('stream-level-label');

  const activeClass = "px-3 py-1 rounded-lg text-xs font-semibold bg-blue-600 text-white shadow transition";
  const inactiveClass = "px-3 py-1 rounded-lg text-xs font-semibold text-slate-400 hover:text-white transition";

  if (btnMic) btnMic.className = source === 'mic' ? activeClass : inactiveClass;
  if (btnDual) btnDual.className = source === 'dual' ? activeClass : inactiveClass;
  if (btnSim) btnSim.className = source === 'sim' ? activeClass : inactiveClass;

  if (source === 'mic') {
    if (simSelector) simSelector.classList.add('hidden');
    if (dualTip) dualTip.classList.add('hidden');
    if (btnLabel && !isLiveStreaming) btnLabel.textContent = "Start Recording";
    if (viewTitle) viewTitle.textContent = "Live Microphone Stream";
    if (viewSubtitle) viewSubtitle.textContent = "Real-time streaming speech recognition with NeMo Sortformer neural diarization.";
    if (levelLabel) levelLabel.textContent = "Mic Level:";
    checkMicSupport();
  } else if (source === 'dual') {
    if (simSelector) simSelector.classList.add('hidden');
    if (dualTip) dualTip.classList.remove('hidden');
    if (btnLabel && !isLiveStreaming) btnLabel.textContent = "Start Dual Recording";
    if (viewTitle) viewTitle.textContent = "Live Microphone & Meeting Stream";
    if (viewSubtitle) viewSubtitle.textContent = "Simultaneous local mic + meeting tab/system audio capture with real-time neural diarization.";
    if (levelLabel) levelLabel.textContent = "Mixed Level:";
    checkMicSupport();
  } else {
    if (simSelector) simSelector.classList.remove('hidden');
    if (dualTip) dualTip.classList.add('hidden');
    if (warnBanner) warnBanner.classList.add('hidden');
    if (btnLabel && !isLiveStreaming) btnLabel.textContent = "Start Simulation";
    if (viewTitle) viewTitle.textContent = "Real-Time File Stream Simulation";
    if (viewSubtitle) viewSubtitle.textContent = "Simulate real-time streaming audio ingestion with NeMo Sortformer neural diarization.";
    if (levelLabel) levelLabel.textContent = "Audio Level:";
  }
}

function handleSimSampleChange() {
  const select = document.getElementById('sim-sample-select');
  const browseBtn = document.getElementById('btn-browse-sim');
  if (select.value === 'custom') {
    browseBtn.classList.remove('hidden');
    if (!customSimFile) {
      document.getElementById('sim-custom-file').click();
    }
  } else {
    browseBtn.classList.add('hidden');
  }
}

function handleSimCustomFile(file) {
  if (!file) return;
  customSimFile = file;
  const browseBtn = document.getElementById('btn-browse-sim');
  browseBtn.textContent = file.name.length > 16 ? file.name.substring(0, 13) + '...' : file.name;
}

function showMicHelpModal(reason = 'insecure_origin') {
  const modal = document.getElementById('modal-mic-help');
  if (!modal) return;

  const flagOrigin = document.getElementById('modal-flag-origin');
  const httpsOrigin = document.getElementById('modal-https-origin');
  const lanOrigin = document.getElementById('modal-lan-origin');
  const alertTitle = document.getElementById('mic-help-alert-title');
  const alertMsg = document.getElementById('mic-help-alert-msg');
  const dualGuide = document.getElementById('mic-help-dual-guide');
  const generalSolutions = document.getElementById('mic-help-general-solutions');

  const origin = window.location.origin;
  const host = window.location.host;
  if (flagOrigin) flagOrigin.textContent = origin;
  if (lanOrigin) lanOrigin.textContent = origin;
  if (httpsOrigin) httpsOrigin.textContent = `https://${host}`;

  if (dualGuide) dualGuide.classList.add('hidden');
  if (generalSolutions) generalSolutions.classList.remove('hidden');

  if (reason === 'no_tab_audio') {
    alertTitle.textContent = "Missing Meeting Tab Audio:";
    alertMsg.textContent = "No audio track was received from your browser. When selecting your meeting tab or window, please ensure 'Also share tab audio' (or 'Share system audio') is checked in the browser dialog.";
    if (dualGuide) dualGuide.classList.remove('hidden');
    if (generalSolutions) generalSolutions.classList.add('hidden');
  } else if (reason === 'dual_info') {
    alertTitle.textContent = "Mic & Meeting Dual Capture Guide:";
    alertMsg.textContent = "Capture both your microphone and remote participants on Google Meet, Zoom web, Teams, or Webex without needing third-party bots.";
    if (dualGuide) dualGuide.classList.remove('hidden');
  } else if (reason === 'display_cancel') {
    alertTitle.textContent = "Meeting Audio Share Cancelled:";
    alertMsg.textContent = "Tab/screen selection was cancelled. Recording was not started.";
  } else if (reason === 'NotAllowedError' || reason === 'PermissionDeniedError' || reason === 'permission') {
    alertTitle.textContent = "Microphone Permission Blocked:";
    alertMsg.textContent = "Your browser blocked microphone permissions for this site. Click the lock/tune icon in your address bar and change Microphone to 'Allow'.";
  } else if (reason === 'NotFoundError' || reason === 'DevicesNotFoundError') {
    alertTitle.textContent = "No Microphone Found:";
    alertMsg.textContent = "No audio input device was detected by your browser. Connect a microphone or headset, or use File Simulation below.";
  } else if (reason === 'ws_error') {
    alertTitle.textContent = "Streaming Connection Disconnected:";
    alertMsg.textContent = "The real-time streaming WebSocket connection encountered an error or was closed. Please ensure Polyphon server is running.";
  } else {
    alertTitle.textContent = "Browser Security Restriction:";
    alertMsg.textContent = `Web browsers disable microphone access on unencrypted HTTP over local network IPs (${origin}). Use File Simulation or follow the HTTPS/flag steps below.`;
  }

  modal.classList.remove('hidden');
}

function hideMicHelpModal() {
  const modal = document.getElementById('modal-mic-help');
  if (modal) modal.classList.add('hidden');
}

async function toggleLiveStream() {
  if (isLiveStreaming) {
    stopLiveStream();
  } else {
    await startLiveStream();
  }
}

async function startLiveStream() {
  if (streamSource === 'sim') {
    await startFileSimulationStream();
    return;
  }

  const btn = document.getElementById('btn-stream-start');
  const btnLabel = document.getElementById('stream-btn-label');
  const btnDot = document.getElementById('stream-btn-dot');
  const pulseDot = document.getElementById('stream-pulse-dot');
  const statusBadge = document.getElementById('stream-status-badge');
  const diarizerSelect = document.getElementById('stream-diarizer');
  const identifyCheck = document.getElementById('stream-identify');

  // Check if browser restricts mediaDevices on insecure HTTP
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showMicHelpModal('insecure_origin');
    return;
  }
  if (streamSource === 'dual' && !navigator.mediaDevices.getDisplayMedia) {
    alert("Your browser does not support getDisplayMedia for tab/meeting audio capture. Please use Chrome, Edge, or Firefox.");
    return;
  }

  try {
    // 1. Acquire microphone audio
    try {
      micStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true,
        }
      });
    } catch (micErr) {
      console.error("Microphone capture failed:", micErr);
      showMicHelpModal(micErr.name || 'permission');
      return;
    }

    // 2. If dual capture, acquire meeting tab/system audio
    if (streamSource === 'dual') {
      try {
        displayStream = await navigator.mediaDevices.getDisplayMedia({
          video: true,
          audio: {
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl: false,
          }
        });
      } catch (dispErr) {
        console.warn("Display audio capture cancelled or failed:", dispErr);
        if (micStream) {
          micStream.getTracks().forEach(t => t.stop());
          micStream = null;
        }
        if (dispErr.name === 'NotAllowedError' || dispErr.name === 'AbortError') {
          showMicHelpModal('display_cancel');
        } else {
          showMicHelpModal(dispErr.name);
        }
        return;
      }

      const displayAudioTracks = displayStream.getAudioTracks();
      if (!displayAudioTracks || displayAudioTracks.length === 0) {
        displayStream.getTracks().forEach(t => t.stop());
        displayStream = null;
        if (micStream) {
          micStream.getTracks().forEach(t => t.stop());
          micStream = null;
        }
        showMicHelpModal('no_tab_audio');
        return;
      }

      // Discard video tracks immediately to eliminate CPU/GPU rendering
      displayStream.getVideoTracks().forEach(track => track.stop());

      // Gracefully stop live stream if user clicks "Stop sharing" on Chrome banner
      displayAudioTracks[0].onended = () => {
        if (isLiveStreaming) {
          console.log("Meeting tab audio sharing stopped by user; finalizing stream.");
          stopLiveStream();
        }
      };
    }

    // 3. Web Audio API Mixing Graph
    audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    if (audioContext.state === 'suspended') {
      await audioContext.resume();
    }

    const mixer = audioContext.createGain();

    // Connect local microphone
    const micSource = audioContext.createMediaStreamSource(micStream);
    const micGain = audioContext.createGain();
    micGain.gain.value = 1.0;
    micSource.connect(micGain);
    micGain.connect(mixer);

    // Connect meeting/tab audio if dual capture
    if (displayStream && displayStream.getAudioTracks().length > 0) {
      const displaySource = audioContext.createMediaStreamSource(displayStream);
      const displayGain = audioContext.createGain();
      displayGain.gain.value = 1.0;
      displaySource.connect(displayGain);
      displayGain.connect(mixer);
    }

    processorNode = audioContext.createScriptProcessor(4096, 1, 1);
    mixer.connect(processorNode);

    // Connect processor to zero-gain sink to prevent feedback echo while keeping graph active
    const silentSink = audioContext.createGain();
    silentSink.gain.value = 0;
    processorNode.connect(silentSink);
    silentSink.connect(audioContext.destination);

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/api/stream/ws`;
    streamSocket = new WebSocket(wsUrl);

    currentLiveSessionName = `live_${Date.now()}`;

    const langSelect = document.getElementById('stream-language');
    const modelSelect = document.getElementById('stream-model');

    const customTitle = (document.getElementById('stream-session-title')?.value || '').trim();

    streamSocket.onopen = () => {
      streamSocket.send(JSON.stringify({
        action: 'start',
        name: currentLiveSessionName,
        title: customTitle || undefined,
        diarizer: diarizerSelect ? diarizerSelect.value : 'sortformer',
        identify: identifyCheck ? identifyCheck.checked : false,
        language: langSelect ? langSelect.value : 'en',
        model: modelSelect ? modelSelect.value : 'base',
      }));
    };

    streamSocket.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        handleStreamMessage(data);
      } catch (err) {
        console.error("Stream message error:", err);
      }
    };

    streamSocket.onerror = (err) => {
      console.error("WebSocket stream error:", err);
      showMicHelpModal('ws_error');
      stopLiveStream();
    };

    streamSocket.onclose = (e) => {
      if (isLiveStreaming) {
        console.warn("WebSocket closed unexpectedly:", e);
        stopLiveStream();
      }
    };

    processorNode.onaudioprocess = (e) => {
      if (!isLiveStreaming || !streamSocket || streamSocket.readyState !== WebSocket.OPEN) return;
      const inputData = e.inputBuffer.getChannelData(0);

      let sum = 0;
      for (let i = 0; i < inputData.length; i++) {
        sum += inputData[i] * inputData[i];
      }
      const rms = Math.sqrt(sum / inputData.length);
      const vuPercent = Math.min(100, Math.round(rms * 400));
      const vuBar = document.getElementById('stream-vu-bar');
      if (vuBar) vuBar.style.width = `${vuPercent}%`;

      const buffer = new ArrayBuffer(inputData.length * 2);
      const view = new DataView(buffer);
      for (let i = 0; i < inputData.length; i++) {
        let s = Math.max(-1, Math.min(1, inputData[i]));
        view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
      }
      streamSocket.send(buffer);
    };

    isLiveStreaming = true;
    btn.className = "px-5 py-2.5 rounded-xl font-semibold text-sm bg-amber-600 hover:bg-amber-500 text-white shadow-lg transition flex items-center gap-2";
    btnLabel.textContent = "Stop & Finalize";
    btnDot.className = "w-2.5 h-2.5 rounded-full bg-white animate-ping";
    if (streamSource === 'dual') {
      pulseDot.className = "w-4 h-4 rounded-full bg-purple-500 animate-pulse";
      statusBadge.className = "px-2.5 py-0.5 rounded-full text-xs font-semibold bg-purple-950/80 text-purple-300 border border-purple-800 animate-pulse";
      statusBadge.textContent = "Dual Stream Live (Mic + Meeting)";
    } else {
      pulseDot.className = "w-4 h-4 rounded-full bg-red-500 animate-pulse";
      statusBadge.className = "px-2.5 py-0.5 rounded-full text-xs font-semibold bg-red-950/80 text-red-300 border border-red-800 animate-pulse";
      statusBadge.textContent = "Streaming Live";
    }

    const feed = document.getElementById('live-transcript-feed');
    feed.innerHTML = '';
    document.getElementById('live-turn-count').textContent = '0 turns';
    const wordsEl = document.getElementById('live-stat-words');
    const wpmEl = document.getElementById('live-stat-wpm');
    if (wordsEl) wordsEl.textContent = '0';
    if (wpmEl) wpmEl.textContent = '0';
    document.getElementById('live-speaker-list').innerHTML = '<p class="text-xs text-slate-500 italic">Listening for speech...</p>';
    document.getElementById('btn-stream-view-studio').disabled = true;
    document.getElementById('btn-stream-view-studio').className = "w-full py-2.5 px-4 rounded-xl text-xs font-semibold bg-slate-800 text-slate-500 transition cursor-not-allowed";
    const procPipelineBtn = document.getElementById('btn-stream-process-pipeline');
    if (procPipelineBtn) {
      procPipelineBtn.disabled = true;
      procPipelineBtn.classList.add('hidden');
    }

    streamStartTime = Date.now();
    if (streamTimerInterval) clearInterval(streamTimerInterval);
    streamTimerInterval = setInterval(() => {
      const elapsedSec = Math.floor((Date.now() - streamStartTime) / 1000);
      const m = String(Math.floor(elapsedSec / 60)).padStart(2, '0');
      const s = String(elapsedSec % 60).padStart(2, '0');
      document.getElementById('stream-timer').textContent = `${m}:${s}`;
    }, 500);

  } catch (err) {
    console.error("Failed to start audio stream:", err);
    showMicHelpModal(err.name || 'permission');
    stopLiveStream();
  }
}

async function startFileSimulationStream() {
  const btn = document.getElementById('btn-stream-start');
  const btnLabel = document.getElementById('stream-btn-label');
  const btnDot = document.getElementById('stream-btn-dot');
  const pulseDot = document.getElementById('stream-pulse-dot');
  const statusBadge = document.getElementById('stream-status-badge');
  const diarizerSelect = document.getElementById('stream-diarizer');
  const identifyCheck = document.getElementById('stream-identify');
  const sampleSelect = document.getElementById('sim-sample-select');

  btn.disabled = true;
  btnLabel.textContent = "Loading Audio...";

  try {
    let arrayBuffer;
    if (sampleSelect.value === 'custom') {
      if (!customSimFile) {
        document.getElementById('sim-custom-file').click();
        btn.disabled = false;
        btnLabel.textContent = "Start Simulation";
        return;
      }
      arrayBuffer = await customSimFile.arrayBuffer();
    } else {
      const resp = await fetch(sampleSelect.value);
      if (!resp.ok) throw new Error(`Could not load audio sample (${resp.status})`);
      arrayBuffer = await resp.arrayBuffer();
    }

    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const decoded = await audioContext.decodeAudioData(arrayBuffer);

    const targetSampleRate = 16000;
    const totalTargetSamples = Math.ceil(decoded.duration * targetSampleRate);
    const offlineCtx = new OfflineAudioContext(1, totalTargetSamples, targetSampleRate);
    const srcNode = offlineCtx.createBufferSource();
    srcNode.buffer = decoded;
    srcNode.connect(offlineCtx.destination);
    srcNode.start(0);
    const rendered = await offlineCtx.startRendering();
    const pcmFloats = rendered.getChannelData(0);

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/api/stream/ws`;
    streamSocket = new WebSocket(wsUrl);
    currentLiveSessionName = `sim_${Date.now()}`;

    const langSelect = document.getElementById('stream-language');
    const modelSelect = document.getElementById('stream-model');

    const customTitle = (document.getElementById('stream-session-title')?.value || '').trim();

    streamSocket.onopen = () => {
      streamSocket.send(JSON.stringify({
        action: 'start',
        name: currentLiveSessionName,
        title: customTitle || undefined,
        diarizer: diarizerSelect ? diarizerSelect.value : 'sortformer',
        identify: identifyCheck ? identifyCheck.checked : false,
        language: langSelect ? langSelect.value : 'en',
        model: modelSelect ? modelSelect.value : 'base',
      }));
    };

    streamSocket.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        handleStreamMessage(data);
      } catch (err) {
        console.error("Stream message error:", err);
      }
    };

    streamSocket.onerror = (err) => {
      console.error("WebSocket stream error:", err);
      showMicHelpModal('ws_error');
      stopLiveStream();
    };

    streamSocket.onclose = (e) => {
      if (isLiveStreaming) {
        console.warn("Simulation WebSocket closed unexpectedly:", e);
        stopLiveStream();
      }
    };

    isLiveStreaming = true;
    btn.disabled = false;
    btn.className = "px-5 py-2.5 rounded-xl font-semibold text-sm bg-amber-600 hover:bg-amber-500 text-white shadow-lg transition flex items-center gap-2";
    btnLabel.textContent = "Stop & Finalize";
    btnDot.className = "w-2.5 h-2.5 rounded-full bg-white animate-ping";
    pulseDot.className = "w-4 h-4 rounded-full bg-blue-500 animate-pulse";
    statusBadge.className = "px-2.5 py-0.5 rounded-full text-xs font-semibold bg-blue-950/80 text-blue-300 border border-blue-800 animate-pulse";
    statusBadge.textContent = "Simulating Stream";

    const feed = document.getElementById('live-transcript-feed');
    feed.innerHTML = '';
    document.getElementById('live-turn-count').textContent = '0 turns';
    const wordsEl = document.getElementById('live-stat-words');
    const wpmEl = document.getElementById('live-stat-wpm');
    if (wordsEl) wordsEl.textContent = '0';
    if (wpmEl) wpmEl.textContent = '0';
    document.getElementById('live-speaker-list').innerHTML = '<p class="text-xs text-slate-500 italic">Listening for speech...</p>';
    document.getElementById('btn-stream-view-studio').disabled = true;
    document.getElementById('btn-stream-view-studio').className = "w-full py-2.5 px-4 rounded-xl text-xs font-semibold bg-slate-800 text-slate-500 transition cursor-not-allowed";
    const procPipelineBtn = document.getElementById('btn-stream-process-pipeline');
    if (procPipelineBtn) {
      procPipelineBtn.disabled = true;
      procPipelineBtn.classList.add('hidden');
    }

    streamStartTime = Date.now();
    if (streamTimerInterval) clearInterval(streamTimerInterval);
    streamTimerInterval = setInterval(() => {
      const elapsedSec = Math.floor((Date.now() - streamStartTime) / 1000);
      const m = String(Math.floor(elapsedSec / 60)).padStart(2, '0');
      const s = String(elapsedSec % 60).padStart(2, '0');
      document.getElementById('stream-timer').textContent = `${m}:${s}`;
    }, 500);

    const chunkSize = 8000; // 0.5s chunks
    let offset = 0;
    const vuBar = document.getElementById('stream-vu-bar');

    simInterval = setInterval(() => {
      if (!isLiveStreaming || !streamSocket || streamSocket.readyState !== WebSocket.OPEN) {
        if (simInterval) clearInterval(simInterval);
        return;
      }

      if (offset >= pcmFloats.length) {
        clearInterval(simInterval);
        simInterval = null;
        setTimeout(() => {
          if (isLiveStreaming) stopLiveStream();
        }, 800);
        return;
      }

      const end = Math.min(offset + chunkSize, pcmFloats.length);
      const slice = pcmFloats.subarray(offset, end);
      offset = end;

      let sum = 0;
      for (let i = 0; i < slice.length; i++) {
        sum += slice[i] * slice[i];
      }
      const rms = Math.sqrt(sum / slice.length);
      const vuPercent = Math.min(100, Math.round(rms * 400));
      if (vuBar) vuBar.style.width = `${vuPercent}%`;

      const buffer = new ArrayBuffer(slice.length * 2);
      const view = new DataView(buffer);
      for (let i = 0; i < slice.length; i++) {
        let s = Math.max(-1, Math.min(1, slice[i]));
        view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
      }
      streamSocket.send(buffer);
    }, 500);

  } catch (err) {
    console.error("Simulation error:", err);
    btn.disabled = false;
    btnLabel.textContent = "Start Simulation";
    showMicHelpModal('simulation_failed');
    stopLiveStream();
  }
}

function stopLiveStream() {
  isLiveStreaming = false;
  if (streamTimerInterval) {
    clearInterval(streamTimerInterval);
    streamTimerInterval = null;
  }
  if (simInterval) {
    clearInterval(simInterval);
    simInterval = null;
  }

  if (streamSocket && streamSocket.readyState === WebSocket.OPEN) {
    streamSocket.send(JSON.stringify({ action: 'stop', name: currentLiveSessionName }));
  }

  if (processorNode) {
    try { processorNode.disconnect(); } catch (e) {}
    processorNode = null;
  }
  if (micStream) {
    micStream.getTracks().forEach(track => track.stop());
    micStream = null;
  }
  if (displayStream) {
    displayStream.getTracks().forEach(track => track.stop());
    displayStream = null;
  }
  if (audioContext && audioContext.state !== 'closed') {
    try { audioContext.close(); } catch (e) {}
    audioContext = null;
  }

  const btn = document.getElementById('btn-stream-start');
  const btnLabel = document.getElementById('stream-btn-label');
  const btnDot = document.getElementById('stream-btn-dot');
  const pulseDot = document.getElementById('stream-pulse-dot');
  const statusBadge = document.getElementById('stream-status-badge');
  const vuBar = document.getElementById('stream-vu-bar');
  if (vuBar) vuBar.style.width = '0%';

  btn.disabled = true;
  btn.className = "px-5 py-2.5 rounded-xl font-semibold text-sm bg-slate-800 text-slate-400 cursor-not-allowed shadow transition flex items-center gap-2";
  btnLabel.textContent = "Finalizing...";
  btnDot.className = "w-2.5 h-2.5 rounded-full bg-amber-400 animate-pulse";
  pulseDot.className = "w-4 h-4 rounded-full bg-slate-600";
  statusBadge.className = "px-2.5 py-0.5 rounded-full text-xs font-semibold bg-amber-950/80 text-amber-300 border border-amber-800 animate-pulse";
  statusBadge.textContent = "Finalizing Session...";

  if (streamSocket && streamSocket.readyState === WebSocket.OPEN) {
    streamSocket.send(JSON.stringify({ action: 'stop', name: currentLiveSessionName }));
  } else {
    btn.disabled = false;
    btn.className = "px-5 py-2.5 rounded-xl font-semibold text-sm bg-red-600 hover:bg-red-500 text-white shadow-lg transition flex items-center gap-2";
    btnLabel.textContent = streamSource === 'sim' ? "Start Simulation" : (streamSource === 'dual' ? "Start Dual Recording" : "Start Recording");
    btnDot.className = "w-2.5 h-2.5 rounded-full bg-white";
    statusBadge.className = "px-2.5 py-0.5 rounded-full text-xs font-semibold bg-slate-800 text-slate-400 border border-slate-700";
    statusBadge.textContent = "Ready";
  }
}

function handleStreamMessage(data) {
  if (data.status === 'completed') {
    lastRecordedLiveSession = data.session_name;
    const viewBtn = document.getElementById('btn-stream-view-studio');
    if (viewBtn) {
      viewBtn.disabled = false;
      viewBtn.className = "w-full py-2.5 px-4 rounded-xl text-xs font-semibold bg-blue-600 hover:bg-blue-500 text-white shadow transition cursor-pointer";
      viewBtn.textContent = `Open "${data.session_name}" in Polyphon Studio 🎙️`;
    }
    const procPipelineBtn = document.getElementById('btn-stream-process-pipeline');
    if (procPipelineBtn) {
      procPipelineBtn.disabled = false;
      procPipelineBtn.classList.remove('hidden');
    }

    const statusBadge = document.getElementById('stream-status-badge');
    if (statusBadge) {
      statusBadge.className = "px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-950/80 text-emerald-300 border border-emerald-800";
      statusBadge.textContent = "Finalized";
    }

    const btn = document.getElementById('btn-stream-start');
    const btnLabel = document.getElementById('stream-btn-label');
    const btnDot = document.getElementById('stream-btn-dot');
    if (btn) {
      btn.disabled = false;
      btn.className = "px-5 py-2.5 rounded-xl font-semibold text-sm bg-red-600 hover:bg-red-500 text-white shadow-lg transition flex items-center gap-2";
    }
    if (btnLabel) {
      btnLabel.textContent = streamSource === 'sim' ? "Start Simulation" : (streamSource === 'dual' ? "Start Dual Recording" : "Start Recording");
    }
    if (btnDot) {
      btnDot.className = "w-2.5 h-2.5 rounded-full bg-white";
    }
    return;
  }

  if (!isLiveStreaming) return;

  if (data.active_speaker) {
    const activeTag = document.getElementById('live-active-speaker');
    if (activeTag) {
      activeTag.classList.remove('hidden');
      activeTag.textContent = `Speaking: ${data.active_speaker}`;
    }
  }

  if (data.segments && data.segments.length > 0) {
    renderLiveTranscript(data.segments);
    updateLiveSpeakers(data.segments);
  }
}

function renderLiveTranscript(segments) {
  const feed = document.getElementById('live-transcript-feed');
  document.getElementById('live-turn-count').textContent = `${segments.length} turn${segments.length === 1 ? '' : 's'}`;

  let totalWords = 0;
  segments.forEach(seg => {
    if (seg.text) {
      totalWords += seg.text.trim().split(/\s+/).filter(Boolean).length;
    }
  });

  const elapsedMin = Math.max(0.1, (Date.now() - (streamStartTime || Date.now())) / 60000);
  const wpm = Math.round(totalWords / elapsedMin);

  const wordsEl = document.getElementById('live-stat-words');
  const wpmEl = document.getElementById('live-stat-wpm');
  if (wordsEl) wordsEl.textContent = totalWords;
  if (wpmEl) wpmEl.textContent = isNaN(wpm) ? 0 : wpm;

  feed.innerHTML = segments.map((seg, idx) => {
    const color = SPEAKER_COLORS[Math.abs(hashString(seg.speaker_id || seg.speaker_name)) % SPEAKER_COLORS.length];
    const m = Math.floor(seg.start / 60);
    const s = Math.floor(seg.start % 60);
    const timeStr = `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
    const isLatest = idx === segments.length - 1;
    const glowClass = isLatest ? 'border-blue-500/50 bg-blue-950/20' : 'border-slate-800 bg-dark-950/40';

    return `
      <div class="p-3 rounded-xl border ${glowClass} transition">
        <div class="flex items-center justify-between gap-2 mb-1">
          <span class="text-xs font-bold px-2 py-0.5 rounded-full" style="color: ${color}; background-color: ${color}15; border: 1px solid ${color}40">
            ${escapeHtml(seg.speaker_name)}
          </span>
          <span class="text-[11px] font-mono text-slate-500">${timeStr}</span>
        </div>
        <p class="text-xs text-slate-200 leading-relaxed">${escapeHtml(seg.text)}</p>
      </div>
    `;
  }).join('');

  feed.scrollTop = feed.scrollHeight;
}

function updateLiveSpeakers(segments) {
  const counts = {};
  segments.forEach(s => {
    counts[s.speaker_name] = (counts[s.speaker_name] || 0) + (s.end - s.start);
  });
  const totalTime = Object.values(counts).reduce((a, b) => a + b, 0) || 1;

  const list = document.getElementById('live-speaker-list');
  list.innerHTML = Object.entries(counts).map(([name, dur]) => {
    const pct = Math.round((dur / totalTime) * 100);
    const color = SPEAKER_COLORS[Math.abs(hashString(name)) % SPEAKER_COLORS.length];
    return `
      <div class="p-2 rounded-lg bg-dark-950 border border-slate-800/80">
        <div class="flex justify-between items-center text-xs mb-1">
          <span class="font-medium text-slate-200">${escapeHtml(name)}</span>
          <span class="text-slate-400 font-mono">${Math.round(dur)}s (${pct}%)</span>
        </div>
        <div class="w-full h-1.5 bg-slate-900 rounded-full overflow-hidden">
          <div class="h-full rounded-full" style="width: ${pct}%; background-color: ${color}"></div>
        </div>
      </div>
    `;
  }).join('');
}

async function openLiveInStudio() {
  if (!lastRecordedLiveSession) return;
  await openArchiveInStudio(lastRecordedLiveSession);
}
