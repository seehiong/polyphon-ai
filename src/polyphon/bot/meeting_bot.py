"""Polyphon Meeting Bot: Headless virtual attendee for automated meeting transcription."""

from __future__ import annotations

import asyncio
import json
import logging
import struct
import tempfile
import time
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger("polyphon.bot")


def get_or_create_silent_wav() -> str:
    """Generate a silent 16kHz WAV file to feed Chromium's fake audio capture device."""
    silent_path = Path(tempfile.gettempdir()) / "polyphon_silent_capture.wav"
    if not silent_path.exists() or silent_path.stat().st_size == 0:
        num_samples = 16000
        with wave.open(str(silent_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(struct.pack(f"<{num_samples}h", *([0] * num_samples)))
    return str(silent_path)


# Audio interceptor script injected into meeting pages to tap WebRTC audio and stream to Polyphon
AUDIO_INTERCEPTOR_SCRIPT = """
(function() {
  if (window.__polyphon_audio_hook_installed) return;
  window.__polyphon_audio_hook_installed = true;

  console.log("[Polyphon Bot] Installing WebRTC & Web Audio interceptor...");

  const wsUrl = window.__polyphon_ws_url || "ws://localhost:7860/api/stream/ws";
  let ws = null;
  let audioContext = null;
  let processorNode = null;
  let mixer = null;
  let isInternalConnecting = false;
  const connectedStreams = new Set();
  const connectedElements = new Set();
  const localMedia = new Set();
  let chunksSent = 0;
  let lastAmpLogTime = 0;

  window.__polyphon_meeting_active = false;
  window.__polyphon_session_started = false;
  window.__polyphon_explicit_stop = false;

  window.__polyphon_start_recording = function() {
    if (window.__polyphon_session_started) return;
    window.__polyphon_meeting_active = true;
    window.__polyphon_session_started = true;
    if (ws && ws.readyState === WebSocket.OPEN) {
      const startMsg = {
        action: "start",
        diarizer: "sortformer",
        identify: true,
        language: "en",
        model: "base"
      };
      if (window.__polyphon_session_name) {
        startMsg.name = window.__polyphon_session_name;
      }
      if (window.__polyphon_session_title) {
        startMsg.title = window.__polyphon_session_title;
      }
      ws.send(JSON.stringify(startMsg));
      console.log("[Polyphon Bot] Meeting is active: recording session started.");
    }
  };

  window.__polyphon_stop_recording = function() {
    window.__polyphon_meeting_active = false;
    window.__polyphon_explicit_stop = true;
    if (ws && ws.readyState === WebSocket.OPEN) {
      try {
        ws.send(JSON.stringify({ action: "stop" }));
      } catch(e) {}
    }
  };

  // Track the bot's own local microphone to never record or stream it
  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    const origGUM = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    navigator.mediaDevices.getUserMedia = async function(...args) {
      const stream = await origGUM(...args);
      if (stream) {
        localMedia.add(stream);
        try {
          stream.getTracks().forEach(t => localMedia.add(t));
        } catch(e) {}
      }
      return stream;
    };
  }

  function initAudioContext() {
    if (audioContext) {
      if (audioContext.state === 'suspended') {
        audioContext.resume().catch(()=>{});
      }
      return;
    }
    try {
      audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      if (audioContext.state === 'suspended') {
        audioContext.resume().catch(()=>{});
      }
      window.__polyphon_audio_context = audioContext;

      mixer = audioContext.createGain();
      mixer.gain.value = 1.0;

      processorNode = audioContext.createScriptProcessor(4096, 1, 1);
      mixer.connect(processorNode);

      // Silent sink to keep Web Audio graph active without acoustic feedback
      const silentSink = audioContext.createGain();
      silentSink.gain.value = 0.0;
      processorNode.connect(silentSink);
      silentSink.connect(audioContext.destination);

      processorNode.onaudioprocess = (e) => {
        if (!window.__polyphon_meeting_active) return;
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        const inputData = e.inputBuffer.getChannelData(0);
        
        // Convert Float32Array to 16-bit PCM ArrayBuffer
        const pcmBuffer = new ArrayBuffer(inputData.length * 2);
        const pcmView = new DataView(pcmBuffer);
        let maxAmp = 0.0;
        for (let i = 0; i < inputData.length; i++) {
          let s = Math.max(-1, Math.min(1, inputData[i]));
          let abs = Math.abs(s);
          if (abs > maxAmp) maxAmp = abs;
          pcmView.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        }
        ws.send(pcmBuffer);
        chunksSent++;
        if (chunksSent === 15) {
          console.log("[Polyphon Bot] Audio streaming pipeline active: audio buffers transmitting to server.");
        }
        if (maxAmp > 0.02) {
          const now = Date.now();
          if (now - lastAmpLogTime > 5000) {
            lastAmpLogTime = now;
            console.log(`[Polyphon Bot] 🎙️ Live participant speech detected! (peak: ${(maxAmp * 100).toFixed(1)}%)`);
          }
        }
      };

      console.log("[Polyphon Bot] Web Audio processing graph initialized at 16kHz.");
    } catch (err) {
      console.error("[Polyphon Bot] Failed to initialize AudioContext:", err);
    }
  }

  function connectMediaStream(stream) {
    if (!stream || connectedStreams.has(stream)) return;
    // Discard the bot's own microphone track (notetakers only record external participant speech)
    if (localMedia.has(stream)) {
      console.log("[Polyphon Bot] Ignoring local microphone stream from recording mixer.");
      return;
    }
    if (stream.getAudioTracks) {
      const tracks = stream.getAudioTracks();
      if (tracks.length === 0) return;
      for (const t of tracks) {
        if (localMedia.has(t)) {
          console.log("[Polyphon Bot] Ignoring local microphone audio track from recording mixer.");
          return;
        }
      }
    }
    try {
      initAudioContext();
      connectedStreams.add(stream);
      isInternalConnecting = true;
      try {
        const streamSource = audioContext.createMediaStreamSource(stream);
        streamSource.connect(mixer);
      } finally {
        isInternalConnecting = false;
      }
      console.log("[Polyphon Bot] Connected participant MediaStream to mixer (tracks: " + (stream.getAudioTracks ? stream.getAudioTracks().length : 0) + ").");
    } catch (e) {
      console.warn("[Polyphon Bot] Could not connect MediaStream to mixer:", e.message);
    }
  }

  function connectMediaElement(element) {
    if (!element || connectedElements.has(element)) return;
    try {
      // Ignore UI sound effect chimes in meeting applications
      if (element.src && (element.src.includes('/sounds/') || element.src.endsWith('.mp3') || element.src.endsWith('.ogg'))) {
        return;
      }
      initAudioContext();
      if (element.srcObject && (element.srcObject instanceof MediaStream || 'getAudioTracks' in element.srcObject)) {
        connectMediaStream(element.srcObject);
        connectedElements.add(element);
        return;
      }
      if (element.src) {
        const source = audioContext.createMediaElementSource(element);
        source.connect(mixer);
        source.connect(audioContext.destination);
        connectedElements.add(element);
        console.log("[Polyphon Bot] Connected media element to audio mixer:", element.tagName);
      }
    } catch (e) {
      console.warn("[Polyphon Bot] Could not connect media element:", e.message);
    }
  }

  // Intercept Web Audio createMediaStreamSource (used by Jitsi Meet, Discord, etc.)
  function hookAudioContextProto(proto) {
    if (!proto || !proto.createMediaStreamSource) return;
    const orig = proto.createMediaStreamSource;
    proto.createMediaStreamSource = function(stream) {
      if (!isInternalConnecting && stream) {
        console.log("[Polyphon Bot] Intercepted AudioContext.createMediaStreamSource from app.");
        connectMediaStream(stream);
      }
      return orig.apply(this, arguments);
    };
  }

  if (window.AudioContext && window.AudioContext.prototype) {
    hookAudioContextProto(window.AudioContext.prototype);
  }
  if (window.webkitAudioContext && window.webkitAudioContext.prototype) {
    hookAudioContextProto(window.webkitAudioContext.prototype);
  }

  // Intercept WebRTC RTCPeerConnection to tap incoming audio tracks directly
  const origPeerConnection = window.RTCPeerConnection || window.webkitRTCPeerConnection;
  if (origPeerConnection) {
    const PeerConnectionProxy = function(...args) {
      const pc = new origPeerConnection(...args);
      pc.addEventListener("track", (event) => {
        if (event && event.track && event.track.kind === "audio") {
          console.log("[Polyphon Bot] Remote participant audio track received from WebRTC.");
          if (event.streams && event.streams.length > 0) {
            connectMediaStream(event.streams[0]);
          } else {
            try {
              const ms = new MediaStream([event.track]);
              connectMediaStream(ms);
            } catch (err) {}
          }
        }
      });
      return pc;
    };
    PeerConnectionProxy.prototype = origPeerConnection.prototype;
    window.RTCPeerConnection = PeerConnectionProxy;
    if (window.webkitRTCPeerConnection) {
      window.webkitRTCPeerConnection = PeerConnectionProxy;
    }
  }

  // Hook srcObject property on HTMLMediaElement to capture streams set dynamically
  const origSrcObjectDesc = Object.getOwnPropertyDescriptor(HTMLMediaElement.prototype, 'srcObject');
  if (origSrcObjectDesc && origSrcObjectDesc.set) {
    const origSet = origSrcObjectDesc.set;
    Object.defineProperty(HTMLMediaElement.prototype, 'srcObject', {
      set: function(stream) {
        origSet.call(this, stream);
        if (stream) connectMediaStream(stream);
      },
      get: origSrcObjectDesc.get,
      configurable: true
    });
  }

  // Safely observe dynamically added <audio> and <video> elements once DOM is ready
  function startObserver() {
    const target = document.body || document.documentElement;
    if (!target) {
      window.addEventListener("DOMContentLoaded", startObserver);
      return;
    }
    try {
      const observer = new MutationObserver(() => {
        document.querySelectorAll("audio, video").forEach(connectMediaElement);
      });
      observer.observe(target, { childList: true, subtree: true });
    } catch (e) {}
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", startObserver);
  } else {
    startObserver();
  }

  // Hook into HTMLMediaElement.prototype.play
  const origPlay = HTMLMediaElement.prototype.play;
  HTMLMediaElement.prototype.play = function() {
    connectMediaElement(this);
    if (audioContext && audioContext.state === 'suspended') {
      audioContext.resume().catch(()=>{});
    }
    return origPlay.apply(this, arguments);
  };

  // Connect to Polyphon Streaming WebSocket
  function connectWebSocket() {
    try {
      ws = new WebSocket(wsUrl);
      window.__polyphon_ws = ws;
      ws.binaryType = "arraybuffer";
      ws.onopen = () => {
        console.log("[Polyphon Bot] Connected to Polyphon Streaming WebSocket at:", wsUrl);
        if (window.__polyphon_meeting_active && !window.__polyphon_session_started) {
          window.__polyphon_start_recording();
        }
      };
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          window.__polyphon_latest_event = data;
          if (data.segments && data.segments.length > 0) {
            const latest = data.segments[data.segments.length - 1];
            console.log(`[Polyphon Bot Transcript] ${latest.speaker_name}: ${latest.text}`);
          }
        } catch(e) {}
      };
      ws.onerror = (err) => {
        console.warn("[Polyphon Bot] WebSocket error:", err);
      };
      ws.onclose = () => {
        if (window.__polyphon_explicit_stop) return;
        console.log("[Polyphon Bot] WebSocket closed, retrying in 3s...");
        setTimeout(connectWebSocket, 3000);
      };
    } catch (err) {
      console.error("[Polyphon Bot] WebSocket connection exception:", err);
    }
  }

  connectWebSocket();
})();
"""


class MeetingBot:
    """Headless virtual attendee capable of joining video meetings and streaming audio."""

    def __init__(
        self,
        meeting_url: str,
        bot_name: str = "Polyphon Notetaker",
        server_ws_url: str = "ws://localhost:7860/api/stream/ws",
        session_title: str | None = None,
        session_name: str | None = None,
        headless: bool = True,
        duration_mins: int | None = None,
        on_transcript: Callable[[dict[str, Any]], None] | None = None,
        on_status: Callable[[str], None] | None = None,
    ):
        self.meeting_url = meeting_url.strip()
        self.bot_name = bot_name.strip()
        self.server_ws_url = server_ws_url.strip()
        self.session_title = session_title.strip() if session_title else None
        self.session_name = session_name.strip() if session_name else None
        self.headless = headless
        self.duration_mins = duration_mins
        self.on_transcript = on_transcript
        self.on_status = on_status
        self.is_running = False
        self._browser: Any = None
        self._page: Any = None
        self._playwright: Any = None

    async def start(self) -> None:
        """Launch the headless browser and join the target meeting."""
        try:
            from playwright.async_api import async_playwright
        except ImportError as err:
            raise RuntimeError(
                'Playwright is required for the meeting bot. Run `pip install "polyphon-ai[bot]"` and `playwright install`.'
            ) from err

        logger.info("Launching MeetingBot for URL: %s", self.meeting_url)
        if self.on_status:
            self.on_status(f"[cyan]🚀 Launching MeetingBot browser for:[/cyan] {self.meeting_url}")
        self.is_running = True

        silent_wav = get_or_create_silent_wav()
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
            args=[
                "--use-fake-ui-for-media-stream",
                "--use-fake-device-for-media-stream",
                f"--use-file-for-fake-audio-capture={silent_wav}",
                "--autoplay-policy=no-user-gesture-required",
                "--ignore-certificate-errors",
                "--allow-insecure-localhost",
                "--disable-web-security",
                "--allow-running-insecure-content",
                "--disable-blink-features=AutomationControlled",
            ],
        )

        context = await self._browser.new_context(
            ignore_https_errors=True,
            permissions=["microphone", "camera"],
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
            ),
        )

        # Inject WS URL configuration and audio hook before page scripts execute
        await context.add_init_script(f"window.__polyphon_ws_url = '{self.server_ws_url}';")
        if self.session_name:
            await context.add_init_script(f"window.__polyphon_session_name = {json.dumps(self.session_name)};")
        if self.session_title:
            await context.add_init_script(f"window.__polyphon_session_title = {json.dumps(self.session_title)};")
        await context.add_init_script(AUDIO_INTERCEPTOR_SCRIPT)

        self._page = await context.new_page()

        def _handle_console(msg: Any) -> None:
            text = msg.text
            if "[Polyphon Bot" in text:
                logger.info("%s", text)
                if self.on_status:
                    self.on_status(f"[dim]{text}[/dim]")
            else:
                logger.debug("[Browser %s] %s", msg.type, text)

        self._page.on("console", _handle_console)
        self._page.on("pageerror", lambda err: logger.warning("[Browser Error] %s", err))

        # Route navigation based on meeting platform
        await self._navigate_and_join()

        # Run listener loop for transcript events or duration limit
        start_time = time.time()
        max_seconds = (self.duration_mins * 60) if self.duration_mins else float("inf")
        has_seen_participants = False
        consecutive_empty_seconds = 0
        waiting_status_shown = False
        meeting_started_announced = False

        try:
            while self.is_running and (time.time() - start_time < max_seconds):
                await asyncio.sleep(1.0)

                # Check if meeting was kicked or ended in platform
                try:
                    is_ended = await self._page.evaluate(
                        "() => Boolean(window.__polyphon_kicked || window.__polyphon_meeting_ended)"
                    )
                    if is_ended:
                        logger.info("Meeting ended or bot was removed.")
                        if self.on_status:
                            self.on_status(
                                "[bold yellow]👋 Meeting session ended or bot was removed. Finalizing session...[/bold yellow]"
                            )
                        break
                except Exception:
                    pass

                # Check for latest transcript event from page
                try:
                    event = await self._page.evaluate("() => window.__polyphon_latest_event || null")
                    if event and self.on_transcript:
                        self.on_transcript(event)
                except Exception:
                    pass

                # Check participant count to detect when meeting starts and when participants leave
                try:
                    participant_info = await self._page.evaluate("""() => {
                        // Jitsi Meet
                        if (window.APP?.conference?.listMembers) {
                            try {
                                const members = window.APP.conference.listMembers();
                                return { platform: 'jitsi', count: members.length };
                            } catch(e) {}
                        }
                        // Google Meet
                        const gmeet = document.querySelectorAll('[data-participant-id], [data-requested-participant-id]');
                        if (gmeet.length > 0) {
                            return { platform: 'google_meet', count: Math.max(0, gmeet.length - 1) };
                        }
                        // Generic remote video tiles
                        const remoteTiles = document.querySelectorAll('#filmstripRemoteVideos .videocontainer, [id^="participant_"], .remote-video');
                        if (remoteTiles.length > 0) {
                            return { platform: 'generic', count: remoteTiles.length };
                        }
                        return { platform: 'none', count: 0 };
                    }""")
                    if participant_info and isinstance(participant_info, dict):
                        count = participant_info.get("count", 0)
                        platform = participant_info.get("platform", "none")

                        # If other participants are present OR fallback if platform doesn't report counts and >20s elapsed
                        if count > 0 or (platform == "none" and (time.time() - start_time > 20)):
                            if not has_seen_participants:
                                has_seen_participants = True
                                if not meeting_started_announced:
                                    meeting_started_announced = True
                                    if self.on_status:
                                        msg = (
                                            f"[bold green]🎉 Meeting has started! ({count} participant(s) present). Starting notetaker...[/bold green]"
                                            if count > 0
                                            else "[bold green]🎉 Starting notetaker audio recording...[/bold green]"
                                        )
                                        self.on_status(msg)
                                # Activate audio streaming pipeline now that meeting has started
                                await self._page.evaluate(
                                    "() => { if (window.__polyphon_start_recording) window.__polyphon_start_recording(); }"
                                )
                            consecutive_empty_seconds = 0
                        elif count == 0:
                            if not has_seen_participants:
                                if not waiting_status_shown:
                                    waiting_status_shown = True
                                    if self.on_status:
                                        self.on_status(
                                            "[yellow]⏳ Meeting room joined, but no other participants are present yet. Waiting for meeting to start...[/yellow]"
                                        )
                            else:
                                consecutive_empty_seconds += 1
                                if consecutive_empty_seconds >= 2:
                                    # Mute/stop sending immediately to prevent silence artifacts
                                    try:
                                        await self._page.evaluate("() => { window.__polyphon_meeting_active = false; }")
                                    except Exception:
                                        pass
                                    logger.info("All participants have left the meeting. Ending session.")
                                    if self.on_status:
                                        self.on_status(
                                            "[bold yellow]👋 All participants have left the meeting. Finalizing notetaker session...[/bold yellow]"
                                        )
                                    break
                except Exception:
                    pass

                # If still on a waiting/prejoin screen and join button becomes enabled (e.g. host arrives), click it
                try:
                    join_btn = self._page.locator(
                        "button:has-text('Join meeting'), div[role='button']:has-text('Join meeting')"
                    ).first
                    if await join_btn.count() > 0 and await join_btn.is_visible() and await join_btn.is_enabled():
                        await join_btn.click(timeout=2000)
                except Exception:
                    pass
        except (asyncio.CancelledError, KeyboardInterrupt):
            logger.info("MeetingBot session cancelled.")
        finally:
            await self.stop()

    async def _navigate_and_join(self) -> None:
        """Platform-specific logic to navigate to meeting and handle join flow."""
        page = self._page
        raw_url = self.meeting_url
        url_lower = raw_url.lower()

        # Append mute hash parameters for Jitsi Meet so camera and mic start disabled
        target_url = raw_url
        if ("meet.jit.si" in url_lower or "jitsi" in url_lower) and "#" not in target_url:
            target_url = f"{target_url}#config.startWithAudioMuted=true&config.startWithVideoMuted=true"

        logger.info("Navigating to %s ...", target_url)
        if self.on_status:
            self.on_status(f"[cyan]🌐 Navigating to:[/cyan] {target_url}")
        await page.goto(target_url, wait_until="networkidle", timeout=30000)
        await asyncio.sleep(2.0)

        if "meet.jit.si" in url_lower or "jitsi" in url_lower:
            await self._handle_jitsi_join()
        elif "meet.google.com" in url_lower:
            await self._handle_google_meet_join()
        else:
            await self._handle_generic_join()

    async def _handle_jitsi_join(self) -> None:
        """Handle joining a Jitsi Meet room."""
        page = self._page
        logger.info("[Jitsi Meet] Setting display name to '%s'...", self.bot_name)
        if self.on_status:
            self.on_status(f"[cyan][Jitsi Meet][/cyan] Setting display name to '{self.bot_name}'...")

        # Find display name input on pre-join screen
        name_input_selectors = [
            "input[placeholder*='name' i]",
            "input[data-testid='prejoin.screen-name']",
            "#premeeting-name-input",
        ]
        name_input_el = None
        for sel in name_input_selectors:
            input_el = page.locator(sel)
            if await input_el.count() > 0 and await input_el.first.is_visible():
                name_input_el = input_el.first
                await name_input_el.fill(self.bot_name)
                break

        await asyncio.sleep(1.0)

        # Press Enter on name input to enter meeting directly
        if name_input_el and hasattr(name_input_el, "press"):
            try:
                res = name_input_el.press("Enter")
                if asyncio.iscoroutine(res):
                    await res
                await asyncio.sleep(1.0)
            except Exception:
                pass

        # Click Join button
        join_btn_selectors = [
            "button:has-text('Join meeting')",
            "button:has-text('Join')",
            "div[role='button']:has-text('Join meeting')",
            "button[data-testid='prejoin.joinMeeting']",
            ".action-btn.primary",
        ]
        for sel in join_btn_selectors:
            btn = page.locator(sel)
            if await btn.count() > 0 and await btn.first.is_visible():
                logger.info("[Jitsi Meet] Clicking join button: %s", sel)
                try:
                    await btn.first.click(timeout=3000)
                    break
                except Exception as e:
                    logger.debug("[Jitsi Meet] Direct click deferred (waiting for host): %s", e)
                    try:
                        await btn.first.evaluate("el => el.click()")
                    except Exception:
                        pass
                    break

        # Explicitly ensure camera and microphone are muted in conference, and hook leave/kicked events
        if hasattr(page, "evaluate"):
            try:
                res = page.evaluate("""() => {
                    if (window.APP?.conference) {
                        try { window.APP.conference.muteAudio(true); } catch(e) {}
                        try { window.APP.conference.muteVideo(true); } catch(e) {}
                        try {
                            window.APP.conference.on("conference.kicked", () => {
                                console.log("[Polyphon Bot] Kicked from conference.");
                                window.__polyphon_kicked = true;
                            });
                            window.APP.conference.on("conference.left", () => {
                                console.log("[Polyphon Bot] Left conference room.");
                                window.__polyphon_meeting_ended = true;
                            });
                        } catch(e) {}
                    }
                }""")
                if asyncio.iscoroutine(res):
                    await res
            except Exception:
                pass

        logger.info("[Jitsi Meet] Successfully connected to meeting room or waiting lobby.")
        if self.on_status:
            self.on_status(
                "[bold green][Jitsi Meet] Connected to meeting room![/bold green] Listening for participant audio..."
            )

    async def _handle_google_meet_join(self) -> None:
        """Handle joining a Google Meet room."""
        page = self._page
        logger.info("[Google Meet] Preparing to join as '%s'...", self.bot_name)
        if self.on_status:
            self.on_status(f"[cyan][Google Meet][/cyan] Preparing to join as '{self.bot_name}'...")

        # Mute camera & microphone buttons before joining
        mute_mic_btn = page.locator("button[aria-label*='turn off microphone' i], button[aria-label*='mic' i]")
        if await mute_mic_btn.count() > 0:
            try:
                await mute_mic_btn.first.click(timeout=3000)
                logger.info("[Google Meet] Muted microphone.")
            except Exception:
                pass

        mute_cam_btn = page.locator("button[aria-label*='turn off camera' i], button[aria-label*='camera' i]")
        if await mute_cam_btn.count() > 0:
            try:
                await mute_cam_btn.first.click(timeout=3000)
                logger.info("[Google Meet] Muted camera.")
            except Exception:
                pass

        # Enter guest name
        name_input = page.locator("input[placeholder*='Your name' i], input[aria-label*='name' i]")
        if await name_input.count() > 0 and await name_input.first.is_visible():
            await name_input.first.fill(self.bot_name)
            logger.info("[Google Meet] Entered name: %s", self.bot_name)

        await asyncio.sleep(1.0)

        # Click "Ask to join" or "Join now"
        join_btn = page.locator("button:has-text('Ask to join'), button:has-text('Join now')")
        if await join_btn.count() > 0:
            await join_btn.first.click()
            logger.info("[Google Meet] Clicked join button. Waiting for host admission...")
            if self.on_status:
                self.on_status("[cyan][Google Meet][/cyan] Requested to join. Waiting for host admission...")

    async def _handle_generic_join(self) -> None:
        """Fallback for generic WebRTC meeting platforms."""
        page = self._page
        logger.info("[Generic WebRTC] Attempting automated join...")
        if self.on_status:
            self.on_status("[cyan][Generic WebRTC][/cyan] Attempting automated join...")
        name_input = page.locator("input[type='text'], input[placeholder*='name' i]")
        if await name_input.count() > 0 and await name_input.first.is_visible():
            await name_input.first.fill(self.bot_name)

        join_btn = page.locator("button:has-text('Join'), button:has-text('Enter')")
        if await join_btn.count() > 0 and await join_btn.first.is_visible():
            await join_btn.first.click()

    async def stop(self) -> None:
        """Stop the meeting bot and terminate browser."""
        if not self.is_running and self._page is None and self._browser is None and self._playwright is None:
            return
        self.is_running = False
        if self._page:
            try:
                # Gracefully notify Polyphon server to finalize session & write all reports
                await self._page.evaluate("""() => {
                    if (window.__polyphon_stop_recording) {
                        window.__polyphon_stop_recording();
                    } else if (window.__polyphon_ws && window.__polyphon_ws.readyState === WebSocket.OPEN) {
                        window.__polyphon_ws.send(JSON.stringify({ action: "stop" }));
                    }
                }""")
                await asyncio.sleep(0.5)
            except Exception:
                pass
            self._page = None

        if self._browser:
            try:
                await self._browser.close()
            except Exception:
                pass
            self._browser = None

        if self._playwright:
            try:
                await self._playwright.stop()
            except Exception:
                pass
            self._playwright = None

        logger.info("MeetingBot session terminated.")
