import { useState, useRef, useEffect } from 'react';
import './App.css';
import './ui-polish.css';

const BACKEND_URL = `${import.meta.env.VITE_BACKEND_URL}/webrtc/offer`;

const VAD_STATES = ['idle', 'listening', 'speaking', 'processing'];

// Project target from the Auralis plan: sub-800 ms responses
const LATENCY_TARGET_MS = 800;
const LATENCY_WARN_MS = 1500;

// Fallback frame length if the backend does not send samples/sample_rate
const DEFAULT_FRAME_MS = 20;

// Stop waiting for ICE gathering after this long.
const ICE_GATHER_TIMEOUT_MS = 3000;

// Mic meter zones (level is 0..1, mapped from -60 dB..0 dB)
const METER_GOOD_LEVEL = 0.25; // ~ -45 dB
const METER_LOUD_LEVEL = 0.85; // ~ -9 dB

// Fix #2: frames arrive ~50/sec. Only update the Frames counter every
// N frames so the whole dashboard does not re-render 50 times a second.
const FRAME_UI_EVERY = 10;

// Fix #3: a finished segment that gets no transcript within this time is
// treated as empty, so it cannot steal the next sentence's transcript.
const TRANSCRIPT_TIMEOUT_MS = 15000;

// Fix #4: ICE servers come from .env instead of being hardcoded.
// If the TURN variables are missing, only STUN is used.
const buildIceServers = () => {
  const servers = [{ urls: 'stun:stun.l.google.com:19302' }];

  const turnUrl = import.meta.env.VITE_TURN_URL;
  const turnUsername = import.meta.env.VITE_TURN_USERNAME;
  const turnCredential = import.meta.env.VITE_TURN_CREDENTIAL;

  if (turnUrl && turnUsername && turnCredential) {
    servers.push({
      urls: turnUrl,
      username: turnUsername,
      credential: turnCredential,
    });
  }

  return servers;
};

const getLatencyClass = (ms) => {
  if (ms === null || ms === undefined) return '';
  if (ms < LATENCY_TARGET_MS) return 'latency-good';
  if (ms < LATENCY_WARN_MS) return 'latency-warn';
  return 'latency-bad';
};

// Week 3 Day 1: colour for the "AI audio" status tile
// (reuses the existing green / yellow / red classes)
const getAiAudioClass = (state) => {
  if (state === 'receiving') return 'latency-good';
  if (state === 'track received' || state === 'stalled') return 'latency-warn';
  if (state === 'blocked') return 'latency-bad';
  return '';
};

const getMeterClass = (level) => {
  if (level < METER_GOOD_LEVEL) return 'meter-quiet';
  if (level < METER_LOUD_LEVEL) return 'meter-good';
  return 'meter-loud';
};

const getMeterText = (level) => {
  if (level < METER_GOOD_LEVEL) return 'quiet';
  if (level < METER_LOUD_LEVEL) return 'good';
  return 'loud';
};

const formatMs = (ms) => (ms === null || ms === undefined ? '—' : `${ms} ms`);

const average = (values) =>
  values.length > 0
    ? Math.round(values.reduce((sum, v) => sum + v, 0) / values.length)
    : null;

function App() {
  const [status, setStatus] = useState('idle');
  const [error, setError] = useState(null);
  const [isConnecting, setIsConnecting] = useState(false);
  const [iceState, setIceState] = useState('new');
  const [vadState, setVadState] = useState('idle');
  // Week 2: one "turn" = what the user said + Auralis's streamed reply
  // { id, time, userText, sttMs, aiText, aiState: 'waiting'|'streaming'|'done'|'error',
  //   ttftMs, totalMs, error }
  const [turns, setTurns] = useState([]);
  const [events, setEvents] = useState([]);
  const [frameCount, setFrameCount] = useState(0);
  const [latencies, setLatencies] = useState([]);
  const [ttfts, setTtfts] = useState([]);
  const [totals, setTotals] = useState([]);
  const [segments, setSegments] = useState([]);

  // Week 3 Day 1: AI audio (backend -> browser) status
  const [aiAudio, setAiAudio] = useState('waiting');
  const [aiPackets, setAiPackets] = useState(0);

  const streamRef = useRef(null);
  const pcRef = useRef(null);
  const audioRef = useRef(null);
  const dataChannelRef = useRef(null);
  const transcriptEndRef = useRef(null);

  // Week 2: id of the turn that the next llm_* messages belong to.
  // The backend handles turns one at a time, in order.
  const currentTurnIdRef = useRef(null);

  // Day 4: arrival time of the most recent frame that contained speech
  const lastSpeechAtRef = useRef(null);

  // Day 5: frame-based segment tracking (refs, because frames arrive ~50/sec)
  const frameMsRef = useRef(DEFAULT_FRAME_MS);
  const segmentRef = useRef({
    startFrame: null,
    lastSpeechFrame: null,
    prevFinishFrame: null,
  });

  // Fix #3: queue of finished segments still waiting for a transcript
  // (oldest first). Each entry: { id, finishedAt }
  const pendingSegmentsRef = useRef([]);

  // Day 6: connection timing + ICE candidate diagnostics
  const connectStartedAtRef = useRef(null);
  const iceCandidateCountsRef = useRef({});
  const relayReadyRef = useRef(false);

  // Day 6: mic level meter (updated directly on the DOM, not via state)
  const audioCtxRef = useRef(null);
  const meterRafRef = useRef(null);
  const meterFillRef = useRef(null);
  const meterLabelRef = useRef(null);

  // Week 3 Day 1: timer that reads WebRTC stats for the incoming AI audio
  const statsTimerRef = useRef(null);

  // Auto-scroll transcript to the newest line
  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [turns]);

  // Clean up the mic meter and stats timer if the component unmounts
  useEffect(() => {
    return () => {
      if (meterRafRef.current) cancelAnimationFrame(meterRafRef.current);
      audioCtxRef.current?.close().catch(() => {});
      if (statsTimerRef.current) clearInterval(statsTimerRef.current);
    };
  }, []);

  // ---------------------------------------------------------
  // STATS
  // ---------------------------------------------------------

  const lastLatency = latencies.length > 0 ? latencies[latencies.length - 1] : null;
  const avgLatency = average(latencies);

  const lastTtft = ttfts.length > 0 ? ttfts[ttfts.length - 1] : null;
  const avgTtft = average(ttfts);
  const lastTotal = totals.length > 0 ? totals[totals.length - 1] : null;

  const transcribedCount = segments.filter((s) => s.transcribed).length;
  const avgSilenceWait = average(segments.map((s) => s.silenceWaitMs));
  const avgSpeech = average(segments.map((s) => s.speechMs));

  // ---------------------------------------------------------
  // EVENT LOGGER
  // ---------------------------------------------------------

  const addEvent = (message) => {
    const time = new Date().toLocaleTimeString();
    setEvents((previous) =>
      [{ id: `${Date.now()}-${Math.random()}`, time, message }, ...previous].slice(0, 30)
    );
  };

  // ---------------------------------------------------------
  // TRANSCRIPT HISTORY
  // ---------------------------------------------------------

  const addTurn = (userText, sttMs) => {
    const id = `${Date.now()}-${Math.random()}`;
    const time = new Date().toLocaleTimeString();

    currentTurnIdRef.current = id;

    setTurns((previous) =>
      [
        ...previous,
        {
          id,
          time,
          userText,
          sttMs,
          aiText: '',
          aiState: 'waiting',
          ttftMs: null,
          totalMs: null,
          error: null,
        },
      ].slice(-30)
    );
  };

  // Update the turn the LLM is currently answering
  const updateCurrentTurn = (update) => {
    const id = currentTurnIdRef.current;
    if (!id) return;

    setTurns((previous) =>
      previous.map((t) => (t.id === id ? { ...t, ...update(t) } : t))
    );
  };

  // ---------------------------------------------------------
  // AI AUDIO STATS (Week 3 Day 1)
  // ---------------------------------------------------------

  const stopStatsPolling = () => {
    if (statsTimerRef.current) {
      clearInterval(statsTimerRef.current);
      statsTimerRef.current = null;
    }
  };

  // Reads the browser's WebRTC statistics once per second and shows how
  // many audio packets have arrived from the backend. This proves audio is
  // reaching the browser even if the speakers are muted.
  const startStatsPolling = (pc) => {
    stopStatsPolling();

    statsTimerRef.current = setInterval(async () => {
      try {
        const stats = await pc.getStats();

        stats.forEach((report) => {
          const kind = report.kind || report.mediaType;

          if (report.type === 'inbound-rtp' && kind === 'audio') {
            setAiPackets(report.packetsReceived ?? 0);
          }
        });
      } catch (err) {
        // The connection may be closing; ignore
      }
    }, 1000);
  };

  // Retry playback after the browser blocked autoplay
  const enableAiAudio = () => {
    if (!audioRef.current) return;

    audioRef.current
      .play()
      .then(() => {
        setAiAudio('receiving');
        setError(null);
        addEvent('AI audio enabled');
      })
      .catch((err) => {
        console.error('[Audio] Retry failed:', err);
        setError('Could not start AI audio. Check the browser volume and output device.');
      });
  };

  // ---------------------------------------------------------
  // MIC LEVEL METER (Day 6)
  // ---------------------------------------------------------

  const startMeter = (stream) => {
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      const ctx = new AudioCtx();
      const source = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();

      analyser.fftSize = 1024;
      source.connect(analyser);

      audioCtxRef.current = ctx;

      const buffer = new Float32Array(analyser.fftSize);

      const tick = () => {
        analyser.getFloatTimeDomainData(buffer);

        let sum = 0;
        for (let i = 0; i < buffer.length; i++) {
          sum += buffer[i] * buffer[i];
        }

        const rms = Math.sqrt(sum / buffer.length);
        const db = rms > 0 ? 20 * Math.log10(rms) : -100;
        const level = Math.min(1, Math.max(0, (db + 60) / 60));

        if (meterFillRef.current) {
          meterFillRef.current.style.width = `${Math.round(level * 100)}%`;
          meterFillRef.current.className = `mic-meter-fill ${getMeterClass(level)}`;
        }

        if (meterLabelRef.current) {
          const dbText = db > -100 ? `${db.toFixed(0)} dB` : '−∞ dB';
          meterLabelRef.current.textContent = `${dbText} · ${getMeterText(level)}`;
        }

        meterRafRef.current = requestAnimationFrame(tick);
      };

      tick();
    } catch (err) {
      console.warn('[METER] Could not start mic level meter:', err);
      addEvent('Mic level meter unavailable');
    }
  };

  const stopMeter = () => {
    if (meterRafRef.current) {
      cancelAnimationFrame(meterRafRef.current);
      meterRafRef.current = null;
    }

    if (audioCtxRef.current) {
      audioCtxRef.current.close().catch(() => {});
      audioCtxRef.current = null;
    }

    if (meterFillRef.current) {
      meterFillRef.current.style.width = '0%';
      meterFillRef.current.className = 'mic-meter-fill';
    }

    if (meterLabelRef.current) {
      meterLabelRef.current.textContent = 'mic off';
    }
  };

  // ---------------------------------------------------------
  // SEGMENT TRACKING (Day 5)
  // ---------------------------------------------------------

  const resetSegmentTracking = () => {
    segmentRef.current = {
      startFrame: null,
      lastSpeechFrame: null,
      prevFinishFrame: null,
    };
    pendingSegmentsRef.current = [];
  };

  const trackSegment = (data) => {
    const seg = segmentRef.current;

    if (data.samples && data.sample_rate) {
      frameMsRef.current = (data.samples / data.sample_rate) * 1000;
    }

    const frameMs = frameMsRef.current;

    if (data.segment_started) {
      seg.startFrame = data.frame;
      seg.lastSpeechFrame = data.frame;
    }

    if (data.is_speech && seg.startFrame !== null) {
      seg.lastSpeechFrame = data.frame;
    }

    if (data.segment_finished && seg.startFrame !== null) {
      const record = {
        id: `${Date.now()}-${Math.random()}`,
        time: new Date().toLocaleTimeString(),
        speechMs: Math.round((seg.lastSpeechFrame - seg.startFrame + 1) * frameMs),
        silenceWaitMs: Math.round((data.frame - seg.lastSpeechFrame) * frameMs),
        gapBeforeMs:
          seg.prevFinishFrame !== null
            ? Math.round((seg.startFrame - seg.prevFinishFrame) * frameMs)
            : null,
        transcribed: false,
        text: '',
      };

      // Fix #3: add to the queue instead of overwriting a single pending id
      pendingSegmentsRef.current.push({
        id: record.id,
        finishedAt: performance.now(),
      });

      setSegments((previous) => [...previous, record].slice(-20));

      seg.prevFinishFrame = data.frame;
      seg.startFrame = null;
      seg.lastSpeechFrame = null;
    }
  };

  // Fix #3: match the transcript to the oldest segment still waiting.
  // Segments that waited longer than TRANSCRIPT_TIMEOUT_MS are dropped from
  // the queue first (they stay "✗ empty" in the audit table).
  const markSegmentTranscribed = (text) => {
    const now = performance.now();
    const queue = pendingSegmentsRef.current;

    while (queue.length > 0 && now - queue[0].finishedAt > TRANSCRIPT_TIMEOUT_MS) {
      queue.shift();
    }

    const next = queue.shift();
    if (!next) return;

    setSegments((previous) =>
      previous.map((s) => (s.id === next.id ? { ...s, transcribed: true, text } : s))
    );
  };

  // ---------------------------------------------------------
  // MICROPHONE
  // ---------------------------------------------------------

  const startMic = async () => {
    setError(null);

    // Guard: do not create a second mic stream while one is running.
    if (streamRef.current) {
      addEvent('Microphone already active');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          // Browser processing ON: boosts quiet mics (stronger signal for
          // the VAD and Whisper), removes background noise, and prevents
          // echo once TTS playback arrives in Week 3.
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
        video: false,
      });

      streamRef.current = stream;
      const track = stream.getAudioTracks()[0];

      console.log('[MIC] Settings:', track.getSettings());
      addEvent(`Microphone started: ${track.label}`);
      setStatus('mic active');

      startMeter(stream);
    } catch (err) {
      console.error('[MIC] Microphone error:', err);

      if (err.name === 'NotAllowedError') {
        setError('Microphone permission denied. Allow microphone access in the browser.');
      } else if (err.name === 'NotFoundError') {
        setError('No microphone found. Connect a microphone and try again.');
      } else {
        setError(`Could not access microphone: ${err.message}`);
      }

      setStatus('mic unavailable');
    }
  };

  // ---------------------------------------------------------
  // DATACHANNEL MESSAGE HANDLER
  // Matches the backend's audio_track.py message types
  // ---------------------------------------------------------

  const handleDataChannelMessage = (rawMessage) => {
    let data;

    try {
      data = typeof rawMessage === 'string' ? JSON.parse(rawMessage) : rawMessage;
    } catch (err) {
      addEvent(`Raw message: ${rawMessage}`);
      return;
    }

    switch (data.type) {
      // ~50 per second: update counters, never log these
      case 'audio_frame': {
        // Fix #2: throttle the Frames counter
        const frame = data.frame ?? 0;
        if (frame % FRAME_UI_EVERY === 0) {
          setFrameCount(frame);
        }

        if (data.is_speech) {
          lastSpeechAtRef.current = performance.now();
        }
        trackSegment(data);
        return;
      }

      case 'vad':
        if (VAD_STATES.includes(data.vad_state)) {
          setVadState(data.vad_state);
          addEvent(`VAD: ${data.vad_state}`);
          if (data.vad_state === 'processing') {
            setStatus('transcribing...');
          }
        }
        return;

      case 'transcript': {
        if (typeof data.text !== 'string' || !data.text.trim()) return;

        const text = data.text.trim();

        // End of speech -> transcript ready.
        // Prefer the backend's measurement (stt_ms, timed from the moment
        // the VAD detected end of speech). Fall back to the browser clock.
        let latencyMs = null;
        if (typeof data.stt_ms === 'number') {
          latencyMs = Math.round(data.stt_ms);
        } else if (lastSpeechAtRef.current !== null) {
          latencyMs = Math.round(performance.now() - lastSpeechAtRef.current);
        }

        lastSpeechAtRef.current = null;

        addTurn(text, latencyMs);
        markSegmentTranscribed(text);

        if (latencyMs !== null) {
          setLatencies((previous) => [...previous, latencyMs].slice(-20));
          addEvent(`Transcript (${latencyMs} ms): ${text}`);
        } else {
          addEvent(`Transcript: ${text}`);
        }

        setStatus('transcript received');
        return;
      }

      // -------- Week 2: LLM reply (Llama 3, streamed) --------

      case 'llm_start':
        setStatus('AI is replying...');
        updateCurrentTurn(() => ({ aiState: 'streaming' }));
        return;

      // Many per reply: update the bubble, never log these
      case 'llm_token': {
        const token = typeof data.text === 'string' ? data.text : '';

        if (data.first && typeof data.ttft_ms === 'number') {
          const ttft = Math.round(data.ttft_ms);
          setTtfts((previous) => [...previous, ttft].slice(-20));
          updateCurrentTurn((t) => ({
            aiState: 'streaming',
            ttftMs: ttft,
            aiText: t.aiText + token,
          }));
        } else {
          updateCurrentTurn((t) => ({ aiState: 'streaming', aiText: t.aiText + token }));
        }
        return;
      }

      case 'llm_done': {
        const total = typeof data.total_ms === 'number' ? Math.round(data.total_ms) : null;
        const ttft = typeof data.ttft_ms === 'number' ? Math.round(data.ttft_ms) : null;

        if (total !== null) {
          setTotals((previous) => [...previous, total].slice(-20));
        }

        updateCurrentTurn((t) => ({
          aiState: 'done',
          // Use the final text from the backend if it sent one
          aiText: typeof data.text === 'string' && data.text.trim() ? data.text.trim() : t.aiText,
          ttftMs: t.ttftMs ?? ttft,
          totalMs: total,
        }));

        addEvent(
          `AI reply done (TTFT ${formatMs(ttft)}, total ${formatMs(total)}, ${data.tokens ?? '?'} tokens)`
        );
        setStatus('reply received');
        return;
      }

      case 'llm_error':
        console.error('[Backend] LLM error:', data);
        updateCurrentTurn(() => ({ aiState: 'error', error: data.error || 'LLM error' }));
        addEvent(`LLM error: ${data.error}`);
        setStatus('LLM error');
        return;

      case 'audio_error':
        console.error('[Backend] Audio error:', data);
        addEvent(`Backend audio error (frame ${data.frame}): ${data.error}`);
        return;

      default:
        if (typeof data.status === 'string') {
          setStatus(data.status);
          addEvent(`Backend status: ${data.status}`);
        } else {
          addEvent(`Unknown message: ${rawMessage}`);
        }
    }
  };

  // ---------------------------------------------------------
  // SETUP DATA CHANNEL
  // ---------------------------------------------------------

  const setupDataChannel = (channel, source = 'DataChannel') => {
    if (!channel) return;

    dataChannelRef.current = channel;

    channel.onopen = () => addEvent(`${source} opened: ${channel.label}`);
    channel.onmessage = (event) => handleDataChannelMessage(event.data);
    channel.onerror = (event) => {
      console.error(`${source} error:`, event);
      addEvent(`${source} error`);
    };
    channel.onclose = () => addEvent(`${source} closed`);
  };

  // ---------------------------------------------------------
  // WEBRTC CONNECTION
  // ---------------------------------------------------------

  const connectWebRTC = async () => {
    setError(null);

    if (!streamRef.current) {
      setError('Start the microphone before connecting.');
      return;
    }

    if (!import.meta.env.VITE_BACKEND_URL) {
      setError('Backend URL is not configured. Check your .env file.');
      return;
    }

    if (
      pcRef.current &&
      (pcRef.current.connectionState === 'connected' ||
        pcRef.current.connectionState === 'connecting')
    ) {
      addEvent('WebRTC connection already active');
      return;
    }

    setIsConnecting(true);
    setVadState('idle');
    setFrameCount(0);
    setAiAudio('waiting');
    setAiPackets(0);
    lastSpeechAtRef.current = null;
    resetSegmentTracking();

    connectStartedAtRef.current = performance.now();
    iceCandidateCountsRef.current = {};
    relayReadyRef.current = false;

    // Fix #4: ICE servers from .env
    const iceServers = buildIceServers();
    if (iceServers.length === 1) {
      addEvent('No TURN server in .env — using STUN only');
    }

    const pc = new RTCPeerConnection({ iceServers });

    pcRef.current = pc;
    addEvent('PeerConnection created');

    // Count candidate types (host / srflx / relay) for diagnostics
    pc.onicecandidate = (event) => {
      if (!event.candidate) return;

      const type = event.candidate.type || 'unknown';
      const counts = iceCandidateCountsRef.current;
      counts[type] = (counts[type] || 0) + 1;

      if (type === 'relay') {
        relayReadyRef.current = true;
      }
    };

    pc.oniceconnectionstatechange = () => {
      const state = pc.iceConnectionState;
      setIceState(state);
      addEvent(`ICE: ${state}`);
      if (state === 'failed') {
        setError('ICE connection failed. The backend or TURN connection may need attention.');
      }
    };

    const dataChannel = pc.createDataChannel('vad-state');
    setupDataChannel(dataChannel, 'Frontend DataChannel');

    pc.ondatachannel = (event) => setupDataChannel(event.channel, 'Backend DataChannel');

    const audioTracks = streamRef.current.getAudioTracks();

    if (audioTracks.length === 0) {
      setError('No audio track found in microphone stream.');
      setIsConnecting(false);
      pc.close();
      return;
    }

    audioTracks.forEach((track) => pc.addTrack(track, streamRef.current));
    addEvent('Microphone audio track added');

    pc.onconnectionstatechange = () => {
      const state = pc.connectionState;
      setStatus(`webrtc: ${state}`);
      addEvent(`Connection: ${state}`);

      if (state === 'connected') {
        setIsConnecting(false);
        startStatsPolling(pc);

        const seconds =
          connectStartedAtRef.current !== null
            ? ((performance.now() - connectStartedAtRef.current) / 1000).toFixed(1)
            : null;

        addEvent(
          seconds !== null
            ? `WebRTC connection established in ${seconds} s`
            : 'WebRTC connection established'
        );
      }

      if (state === 'failed') {
        setIsConnecting(false);
        stopStatsPolling();
        setError('Media connection failed. Check the backend WebRTC connection and TURN configuration.');
      }

      if (state === 'disconnected' || state === 'closed') {
        setIsConnecting(false);
        stopStatsPolling();
      }
    };

    // Week 3 Day 1: the backend sends AI speech as an audio track.
    // The browser buffers and plays it; we show its status.
    pc.ontrack = (event) => {
      addEvent(`Remote ${event.track.kind} track received`);

      if (event.track.kind !== 'audio') return;

      setAiAudio('track received');

      // A remote track starts muted and unmutes once packets arrive
      event.track.onunmute = () => {
        setAiAudio((previous) => (previous === 'blocked' ? previous : 'receiving'));
        addEvent('AI audio is flowing');
      };

      event.track.onmute = () => {
        setAiAudio((previous) => (previous === 'blocked' ? previous : 'stalled'));
        addEvent('AI audio stopped arriving');
      };

      if (!event.track.muted) {
        setAiAudio('receiving');
      }

      const stream =
        event.streams && event.streams[0]
          ? event.streams[0]
          : new MediaStream([event.track]);

      if (audioRef.current) {
        audioRef.current.srcObject = stream;

        audioRef.current.play().catch((err) => {
          console.error('[Audio] Playback failed:', err);
          setAiAudio('blocked');
          setError('Browser blocked AI audio. Click ENABLE AI AUDIO.');
        });
      }
    };

    try {
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      addEvent('SDP offer created');

      // Wait for ICE gathering, but stop early when possible:
      // - gathering completes, or
      // - a relay (TURN) candidate arrives, or
      // - ICE_GATHER_TIMEOUT_MS passes
      await new Promise((resolve) => {
        if (pc.iceGatheringState === 'complete') {
          addEvent('ICE gathering done (complete)');
          resolve();
          return;
        }

        let settled = false;

        const finish = (reason) => {
          if (settled) return;
          settled = true;
          clearTimeout(timeout);
          clearInterval(relayCheck);
          pc.removeEventListener('icegatheringstatechange', onGatheringChange);
          addEvent(`ICE gathering done (${reason})`);
          resolve();
        };

        const onGatheringChange = () => {
          if (pc.iceGatheringState === 'complete') finish('complete');
        };

        // Poll the relay flag set by onicecandidate
        const relayCheck = setInterval(() => {
          if (relayReadyRef.current) finish('relay ready');
        }, 50);

        const timeout = setTimeout(() => {
          finish(`timeout after ${ICE_GATHER_TIMEOUT_MS / 1000} s`);
        }, ICE_GATHER_TIMEOUT_MS);

        pc.addEventListener('icegatheringstatechange', onGatheringChange);
      });

      const counts = iceCandidateCountsRef.current;
      addEvent(
        `ICE candidates: host=${counts.host || 0}, ` +
        `srflx=${counts.srflx || 0}, relay=${counts.relay || 0}`
      );

      if (!pc.localDescription) {
        throw new Error('Local SDP description was not created.');
      }

      setStatus('sending offer to backend...');

      const response = await fetch(BACKEND_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sdp: pc.localDescription.sdp,
          type: pc.localDescription.type,
        }),
      });

      if (!response.ok) {
        if (response.status === 404) {
          throw new Error('Signaling endpoint not found (404). Check the backend URL and /webrtc/offer route.');
        }
        if (response.status === 403) {
          throw new Error('Request blocked (403). The backend may not allow this origin.');
        }
        throw new Error(`Backend responded with ${response.status}`);
      }

      const answer = await response.json();
      addEvent('SDP answer received');

      if (!answer || !answer.sdp || !answer.type) {
        throw new Error('Backend returned an invalid SDP answer.');
      }

      await pc.setRemoteDescription(answer);
      addEvent('Remote SDP applied');
      setStatus('handshake complete — connecting');
    } catch (err) {
      console.error('[WebRTC] Handshake failed:', err);
      setIsConnecting(false);

      if (err.message === 'Failed to fetch') {
        setError('Could not reach the backend. Check that the backend and ngrok are running.');
      } else {
        setError(err.message || 'WebRTC handshake failed.');
      }

      setStatus('connection failed');
    }
  };

  // ---------------------------------------------------------
  // STOP EVERYTHING
  // ---------------------------------------------------------

  const stopMic = () => {
    stopMeter();
    stopStatsPolling();

    if (dataChannelRef.current) {
      try { dataChannelRef.current.close(); } catch (err) { console.warn(err); }
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
    }

    if (pcRef.current) {
      try { pcRef.current.close(); } catch (err) { console.warn(err); }
    }

    if (audioRef.current) {
      audioRef.current.srcObject = null;
    }

    streamRef.current = null;
    pcRef.current = null;
    dataChannelRef.current = null;
    lastSpeechAtRef.current = null;
    connectStartedAtRef.current = null;
    currentTurnIdRef.current = null;
    resetSegmentTracking();

    setError(null);
    setIsConnecting(false);
    setVadState('idle');
    setStatus('idle');
    setIceState('new');
    setFrameCount(0);
    setAiAudio('waiting');
    setAiPackets(0);
    setEvents([]);
    addEvent('Connection stopped');
    // Transcripts, latency and audit history are kept for review after stopping
  };

  const clearHistory = () => {
    setTurns([]);
    setLatencies([]);
    setTtfts([]);
    setTotals([]);
    setSegments([]);
    currentTurnIdRef.current = null;
  };

  // ---------------------------------------------------------
  // VAD DISPLAY
  // ---------------------------------------------------------

  const getVadIcon = () =>
    ({ speaking: '🗣️', listening: '🎧', processing: '⚙️' }[vadState] || '◉');

  const getVadLabel = () =>
    ({ speaking: 'SPEAKING', listening: 'LISTENING', processing: 'PROCESSING' }[vadState] || 'IDLE');

  // ---------------------------------------------------------
  // UI
  // ---------------------------------------------------------

  return (
    <div className="auralis-app">
      <div className="auralis-container">

        <header className="auralis-header">
          <div className="brand">
            <div className="brand-logo">A</div>
            <div>
              <h1>AURALIS</h1>
              <p>REAL-TIME VOICE INTELLIGENCE ENGINE</p>
            </div>
          </div>
          <div className="system-status">
            <span className="status-dot"></span>
            SYSTEM ONLINE
          </div>
        </header>

        <div className="dashboard-grid">

          {/* CONNECTION */}

          <section className="card">
            <div className="card-title">
              <h2>Connection</h2>
              <span>WEBRTC</span>
            </div>

            <div className="connection-grid">
              <div className="metric">
                <div className="metric-label">Status</div>
                <div className="metric-value">{status}</div>
              </div>
              <div className="metric">
                <div className="metric-label">ICE State</div>
                <div className="metric-value">{iceState}</div>
              </div>
              <div className="metric">
                <div className="metric-label">Frames</div>
                <div className="metric-value">{frameCount}</div>
              </div>
              <div className="metric">
                <div className="metric-label">Utterances</div>
                {/* Fix #1: count transcripts, not latency values */}
                <div className="metric-value">{turns.length}</div>
              </div>
              <div className="metric">
                <div className="metric-label">AI audio</div>
                <div className={`metric-value ${getAiAudioClass(aiAudio)}`}>{aiAudio}</div>
              </div>
              <div className="metric">
                <div className="metric-label">AI packets</div>
                <div className="metric-value">{aiPackets}</div>
              </div>
            </div>

            {/* MIC LEVEL METER (Day 6) */}

            <div className="mic-meter">
              <div className="mic-meter-header">
                <span className="metric-label">Mic level</span>
                <span className="mic-meter-label" ref={meterLabelRef}>mic off</span>
              </div>
              <div className="mic-meter-track">
                <div className="mic-meter-fill" ref={meterFillRef} />
              </div>
            </div>

            <div className="controls">
              <button className="btn" onClick={startMic}>🎙 START MIC</button>
              <button className="btn btn-primary" onClick={connectWebRTC} disabled={isConnecting}>
                {isConnecting ? 'CONNECTING...' : 'CONNECT'}
              </button>
              <button className="btn btn-danger" onClick={stopMic}>STOP</button>
              {aiAudio === 'blocked' && (
                <button className="btn" onClick={enableAiAudio}>🔊 ENABLE AI AUDIO</button>
              )}
            </div>

            {error && <div className="error-box">⚠ {error}</div>}
          </section>

          {/* VOICE ACTIVITY + LATENCY */}

          <section className="card">
            <div className="card-title">
              <h2>Voice Activity</h2>
              <span>VAD ENGINE</span>
            </div>

            <div className="vad-box">
              <div className="vad-state">
                <div className="vad-icon">{getVadIcon()}</div>
                <div className="vad-name">{getVadLabel()}</div>
              </div>
            </div>

            <div className="connection-grid latency-grid">
              <div className="metric">
                <div className="metric-label">Speech → Text (last)</div>
                <div className={`metric-value ${getLatencyClass(lastLatency)}`}>
                  {formatMs(lastLatency)}
                </div>
                <div className="metric-sub">avg {formatMs(avgLatency)}</div>
              </div>
              <div className="metric metric-highlight">
                <div className="metric-label">TTFT (last)</div>
                <div className={`metric-value ${getLatencyClass(lastTtft)}`}>
                  {formatMs(lastTtft)}
                </div>
                <div className="metric-sub">speech end → first AI word</div>
              </div>
              <div className="metric">
                <div className="metric-label">Avg TTFT</div>
                <div className={`metric-value ${getLatencyClass(avgTtft)}`}>
                  {formatMs(avgTtft)}
                </div>
                <div className="metric-sub">{ttfts.length} {ttfts.length === 1 ? 'reply' : 'replies'}</div>
              </div>
              <div className="metric">
                <div className="metric-label">Full reply (last)</div>
                <div className="metric-value">{formatMs(lastTotal)}</div>
                <div className="metric-sub">speech end → last AI word</div>
              </div>
            </div>
          </section>

          {/* CONVERSATION (Week 2: transcript + streamed LLM reply) */}

          <section className="card full">
            <div className="card-title">
              <h2>Conversation</h2>
              <span>
                WHISPER → LLAMA 3
                {turns.length > 0 && (
                  <button className="btn-link" onClick={clearHistory}>&nbsp;· CLEAR</button>
                )}
              </span>
            </div>

            <div className="conversation-box">
              {turns.length === 0 ? (
                <span className="transcript-placeholder">
                  Connect and say something. Your words and Auralis's reply appear here.
                </span>
              ) : (
                turns.map((turn) => (
                  <div className="turn" key={turn.id}>
                    <div className="bubble bubble-user">
                      <div className="bubble-meta">
                        <span className="bubble-who">You</span>
                        <span className="bubble-time">{turn.time}</span>
                        {turn.sttMs !== null && turn.sttMs !== undefined && (
                          <span className={`bubble-badge ${getLatencyClass(turn.sttMs)}`}>
                            STT {turn.sttMs} ms
                          </span>
                        )}
                      </div>
                      <div className="bubble-text">{turn.userText}</div>
                    </div>

                    <div className={`bubble bubble-ai ai-${turn.aiState}`}>
                      <div className="bubble-meta">
                        <span className="bubble-who">Auralis</span>
                        {turn.ttftMs !== null && (
                          <span className={`bubble-badge ${getLatencyClass(turn.ttftMs)}`}>
                            TTFT {turn.ttftMs} ms
                          </span>
                        )}
                        {turn.totalMs !== null && (
                          <span className="bubble-badge">total {turn.totalMs} ms</span>
                        )}
                      </div>
                      <div className="bubble-text">
                        {turn.aiState === 'error' && `⚠ ${turn.error}`}
                        {turn.aiState === 'waiting' && (
                          <span className="typing">
                            <span></span><span></span><span></span>
                          </span>
                        )}
                        {(turn.aiState === 'streaming' || turn.aiState === 'done') && turn.aiText}
                        {turn.aiState === 'streaming' && <span className="cursor">▍</span>}
                      </div>
                    </div>
                  </div>
                ))
              )}
              <div ref={transcriptEndRef} />
            </div>
          </section>

          {/* TRANSCRIPTION AUDIT */}

          <section className="card full">
            <div className="card-title">
              <h2>Transcription Audit</h2>
              <span>VAD SEGMENTS · SILENCE THRESHOLDS</span>
            </div>

            <div className="audit-summary">
              <div className="metric">
                <div className="metric-label">Segments detected</div>
                <div className="metric-value">{segments.length}</div>
              </div>
              <div className="metric">
                <div className="metric-label">Transcribed</div>
                <div className="metric-value">
                  {segments.length > 0 ? `${transcribedCount} / ${segments.length}` : '—'}
                </div>
              </div>
              <div className="metric">
                <div className="metric-label">Avg silence wait</div>
                <div className="metric-value">{formatMs(avgSilenceWait)}</div>
              </div>
              <div className="metric">
                <div className="metric-label">Avg speech</div>
                <div className="metric-value">{formatMs(avgSpeech)}</div>
              </div>
            </div>

            <div className="audit-table-wrap">
              {segments.length === 0 ? (
                <div className="transcript-placeholder">
                  No speech segments yet. Speak a sentence, then pause.
                </div>
              ) : (
                <table className="audit-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Time</th>
                      <th>Speech</th>
                      <th>Silence wait</th>
                      <th>Gap before</th>
                      <th>Result</th>
                    </tr>
                  </thead>
                  <tbody>
                    {segments.map((s, index) => (
                      <tr key={s.id}>
                        <td>{index + 1}</td>
                        <td>{s.time}</td>
                        <td>{formatMs(s.speechMs)}</td>
                        <td>{formatMs(s.silenceWaitMs)}</td>
                        <td>{formatMs(s.gapBeforeMs)}</td>
                        <td className={s.transcribed ? 'audit-ok' : 'audit-miss'}>
                          {s.transcribed ? `✓ ${s.text}` : '✗ empty'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </section>

          {/* SYSTEM EVENTS */}

          <section className="card full">
            <div className="card-title">
              <h2>System Events</h2>
              <span>LIVE LOG</span>
            </div>
            <div className="events">
              {events.length === 0 ? (
                <div className="transcript-placeholder">No events yet.</div>
              ) : (
                events.map((event) => (
                  <div className="event" key={event.id}>
                    <span className="event-time">{event.time}</span>
                    <span className="event-message">{event.message}</span>
                  </div>
                ))
              )}
            </div>
          </section>

        </div>

        <div className="footer">AURALIS // REAL-TIME MULTI-MODEL VOICE SYSTEM</div>

        <audio ref={audioRef} autoPlay playsInline />
      </div>
    </div>
  );
}

export default App;