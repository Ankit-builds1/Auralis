import { useState, useRef, useEffect } from 'react';
import './App.css';
import './ui-polish.css';

const BACKEND_URL = `${import.meta.env.VITE_BACKEND_URL}/webrtc/offer`;

// Week 3 Day 3: 'ai_speaking' = Auralis is talking (Piper TTS)
const VAD_STATES = ['idle', 'listening', 'speaking', 'processing', 'ai_speaking'];

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

// Week 3 Day 2 fix: emotion guesses below this confidence are shown as
// "neutral" (the raw guess is still shown in brackets for debugging).
const EMOTION_SHOW_MIN = 0.6;

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

// Week 3 Day 2: emoji for each emotion label sent by the backend
const EMOTION_EMOJI = {
  angry: '😠',
  disgust: '🤢',
  fear: '😨',
  happy: '😊',
  neutral: '😐',
  sad: '😢',
};

// Week 3 Day 2 fix: decide what to show for one emotion result.
// Confident  -> real label + group colour.
// Not sure   -> "neutral" in the okay colour, raw guess kept for debugging.
const getEmotionView = (emotion) => {
  if (!emotion) return null;

  const pct = Math.round(emotion.confidence * 100);
  const sure = emotion.confident && emotion.confidence >= EMOTION_SHOW_MIN;

  if (sure) {
    return {
      sure: true,
      emoji: EMOTION_EMOJI[emotion.label] ?? '',
      text: emotion.label,
      group: emotion.group,
      pct,
      raw: emotion.label,
    };
  }

  return {
    sure: false,
    emoji: EMOTION_EMOJI.neutral,
    text: 'neutral',
    group: 'okay',
    pct,
    raw: emotion.label,
  };
};

// Week 3 Day 3: voice style Piper used for the reply (from the emotion group)
const VOICE_STYLE_ICON = {
  normal: '🙂',
  calm: '🫶',
  soft: '🤍',
};

// Week 3 Day 4: latency breakdown (speech end -> Auralis starts talking)
// Each turn's time is split into the stages that add up to "first audio".
const LATENCY_STAGES = [
  { key: 'stt', label: 'Speech → text', cls: 'stage-stt' },
  { key: 'llm', label: 'Llama thinking', cls: 'stage-llm' },
  { key: 'piece', label: 'First phrase written', cls: 'stage-piece' },
  { key: 'tts', label: 'Voice synthesis', cls: 'stage-tts' },
];
const BREAKDOWN_TURNS = 8;

// Split one turn's metrics into stage durations (ms), never negative
const getStages = (m) => {
  const stt = m.stt_ms ?? 0;
  const ttft = m.ttft_ms ?? stt;
  const piece = m.first_piece_ms ?? ttft;
  const ttfa = m.ttfa_ms ?? piece;
  return {
    stt: Math.max(0, stt),
    llm: Math.max(0, ttft - stt),
    piece: Math.max(0, piece - ttft),
    tts: Math.max(0, ttfa - piece),
  };
};

// p-th percentile (0..100) of a list of numbers, nearest-rank method
const percentile = (values, p) => {
  const list = values.filter((v) => typeof v === 'number').sort((a, b) => a - b);
  if (list.length === 0) return null;
  const rank = Math.ceil((p / 100) * list.length) - 1;
  return list[Math.min(list.length - 1, Math.max(0, rank))];
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
  //   ttftMs, totalMs, error, emotion }
  const [turns, setTurns] = useState([]);
  const [events, setEvents] = useState([]);
  const [frameCount, setFrameCount] = useState(0);
  const [latencies, setLatencies] = useState([]);
  const [ttfts, setTtfts] = useState([]);
  const [totals, setTotals] = useState([]);
  const [segments, setSegments] = useState([]);

  // Week 3 Day 2: voice emotion per turn (WavLM-Large on the backend)
  // { label, confidence, group: 'upset'|'low'|'okay', confident, ms }
  const [emotions, setEmotions] = useState([]);

  // Week 3 Day 3: time to first audio (speech end -> Auralis starts talking)
  const [ttfas, setTtfas] = useState([]);
  const [echoCount, setEchoCount] = useState(0);

  // Week 3 Day 4: one latency breakdown per turn (from 'turn_metrics')
  const [turnMetrics, setTurnMetrics] = useState([]);

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

  const lastEmotion = emotions.length > 0 ? emotions[emotions.length - 1] : null;
  const lastEmotionView = getEmotionView(lastEmotion);

  const lastTtfa = ttfas.length > 0 ? ttfas[ttfas.length - 1] : null;
  const avgTtfa = average(ttfas);

  // Week 3 Day 4: latency breakdown stats
  const recentMetrics = turnMetrics.slice(-BREAKDOWN_TURNS);
  const breakdownScale = Math.max(
    LATENCY_TARGET_MS,
    ...recentMetrics.map((m) => m.ttfa_ms ?? m.ttft_ms ?? m.stt_ms ?? 0)
  );
  const ttfaValues = turnMetrics.map((m) => m.ttfa_ms);
  const p50Ttfa = percentile(ttfaValues, 50);
  const p95Ttfa = percentile(ttfaValues, 95);
  const avgStages = LATENCY_STAGES.map((stage) => ({
    ...stage,
    ms: average(turnMetrics.map((m) => getStages(m)[stage.key])),
  }));
  const slowestStage = avgStages.reduce(
    (worst, stage) => ((stage.ms ?? 0) > (worst?.ms ?? -1) ? stage : worst),
    null
  );

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
          emotion: null,
          // Week 3 Day 3: { style, ttfaMs, speaking, sentences, audioMs }
          voice: null,
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
          // Week 3 Day 2 fix:
          // echoCancellation ON  -> needed once the AI speaks (TTS).
          // noiseSuppression OFF -> keeps the natural voice tone for emotion.
          // autoGainControl OFF  -> keeps real loudness (main cue for anger).
          echoCancellation: true,
          noiseSuppression: false,
          autoGainControl: false,
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

        // Week 3 Day 3: speech heard while Auralis is talking is echo
        // (its own voice through the speakers). The backend ignores it,
        // so it is also kept out of the latency clock and the audit table.
        if (data.echo) return;

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
          if (data.vad_state === 'ai_speaking') {
            setStatus('Auralis is speaking...');
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

      // -------- Week 3 Day 2: voice emotion (WavLM-Large) --------
      // Sent right after the transcript, so it belongs to the newest turn

      case 'emotion': {
        if (typeof data.label !== 'string') return;

        const emotion = {
          label: data.label,
          confidence: typeof data.confidence === 'number' ? data.confidence : 0,
          group: data.group || 'okay',
          confident: data.confident !== false,
          ms: typeof data.ms === 'number' ? Math.round(data.ms) : null,
        };

        const view = getEmotionView(emotion);

        setEmotions((previous) => [...previous, emotion].slice(-20));
        updateCurrentTurn(() => ({ emotion }));
        addEvent(
          view.sure
            ? `Emotion: ${emotion.label} (${view.pct}%, ${emotion.group}, ${formatMs(emotion.ms)})`
            : `Emotion: neutral — low confidence, raw guess ${emotion.label} ${view.pct}% (${formatMs(emotion.ms)})`
        );
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

      // -------- Week 3 Day 3: Auralis speaks (Piper TTS) --------

      case 'tts_start': {
        const ttfa = typeof data.ttfa_ms === 'number' ? Math.round(data.ttfa_ms) : null;
        const style = data.style || 'normal';

        if (ttfa !== null) {
          setTtfas((previous) => [...previous, ttfa].slice(-20));
        }

        updateCurrentTurn(() => ({
          voice: { style, ttfaMs: ttfa, speaking: true, sentences: 0, audioMs: 0 },
        }));
        setStatus('Auralis is speaking...');
        addEvent(`AI voice started (TTFA ${formatMs(ttfa)}, ${style} voice)`);
        return;
      }

      case 'tts_sentence':
        updateCurrentTurn((t) => ({
          voice: {
            ...(t.voice || { style: 'normal', ttfaMs: null, speaking: true }),
            sentences: data.index ?? (t.voice?.sentences ?? 0) + 1,
            audioMs: (t.voice?.audioMs ?? 0) + (data.audio_ms ?? 0),
          },
        }));
        return;

      case 'tts_end':
        updateCurrentTurn((t) => ({
          voice: {
            ...(t.voice || { style: data.style || 'normal', ttfaMs: data.ttfa_ms ?? null }),
            speaking: false,
            sentences: data.sentences ?? t.voice?.sentences ?? 0,
            audioMs: data.audio_ms ?? t.voice?.audioMs ?? 0,
          },
        }));
        setStatus('reply spoken');
        addEvent(
          `AI voice done (${data.sentences ?? '?'} sentences, ` +
          `${formatMs(data.audio_ms)} audio, ${data.underruns ?? 0} gaps)`
        );
        return;

      // -------- Week 3 Day 4: latency breakdown for the finished turn --------

      case 'turn_metrics': {
        const metrics = {
          id: `${Date.now()}-${Math.random()}`,
          stt_ms: data.stt_ms ?? null,
          emotion_ms: data.emotion_ms ?? null,
          ttft_ms: data.ttft_ms ?? null,
          first_piece_ms: data.first_piece_ms ?? null,
          ttfa_ms: data.ttfa_ms ?? null,
          first_synth_ms: data.first_synth_ms ?? null,
          total_ms: data.total_ms ?? null,
          voice: data.voice ?? null,
        };

        setTurnMetrics((previous) => [...previous, metrics].slice(-50));
        updateCurrentTurn(() => ({ metrics }));
        addEvent(
          `Latency: STT ${formatMs(metrics.stt_ms)} → first word ${formatMs(metrics.ttft_ms)} ` +
          `→ first audio ${formatMs(metrics.ttfa_ms)}`
        );
        return;
      }

      case 'tts_error':
        console.error('[Backend] TTS error:', data);
        addEvent(`TTS error: ${data.error}`);
        return;

      case 'echo_ignored':
        setEchoCount((count) => count + 1);
        addEvent('Echo ignored (speech heard while Auralis was talking)');
        return;

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
    setEmotions([]);
    setTtfas([]);
    setEchoCount(0);
    setTurnMetrics([]);
    currentTurnIdRef.current = null;
  };

  // ---------------------------------------------------------
  // VAD DISPLAY
  // ---------------------------------------------------------

  const getVadIcon = () =>
    ({ speaking: '🗣️', listening: '🎧', processing: '⚙️', ai_speaking: '🔊' }[vadState] || '◉');

  const getVadLabel = () =>
    ({
      speaking: 'SPEAKING',
      listening: 'LISTENING',
      processing: 'PROCESSING',
      ai_speaking: 'AURALIS SPEAKING',
    }[vadState] || 'IDLE');

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
              <div className="metric">
                <div className="metric-label">AI voice</div>
                <div className={`metric-value ${vadState === 'ai_speaking' ? 'latency-good' : ''}`}>
                  {vadState === 'ai_speaking' ? 'speaking' : 'quiet'}
                </div>
              </div>
              <div className="metric">
                <div className="metric-label">Echo ignored</div>
                <div className="metric-value">{echoCount}</div>
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

            <div className={`vad-box vad-${vadState}`}>
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
              <div className="metric metric-highlight">
                <div className="metric-label">Time to first audio (last)</div>
                <div className={`metric-value ${getLatencyClass(lastTtfa)}`}>
                  {formatMs(lastTtfa)}
                </div>
                <div className="metric-sub">
                  {ttfas.length > 0
                    ? `avg ${formatMs(avgTtfa)} · speech end → AI voice`
                    : 'speech end → AI voice'}
                </div>
              </div>
              <div className="metric">
                <div className="metric-label">Emotion (last)</div>
                <div className={`metric-value emotion-text-${lastEmotionView?.group ?? 'none'}`}>
                  {lastEmotionView
                    ? `${lastEmotionView.emoji} ${lastEmotionView.text}`
                    : '—'}
                </div>
                <div className="metric-sub">
                  {lastEmotionView
                    ? lastEmotionView.sure
                      ? `${lastEmotionView.pct}% · ${lastEmotionView.group} · ${formatMs(lastEmotion.ms)}`
                      : `low confidence · guess ${lastEmotionView.raw} ${lastEmotionView.pct}% · ${formatMs(lastEmotion.ms)}`
                    : 'voice tone · WavLM-Large'}
                </div>
              </div>
            </div>
          </section>

          {/* CONVERSATION (Week 2: transcript + streamed LLM reply) */}

          <section className="card full">
            <div className="card-title">
              <h2>Conversation</h2>
              <span>
                WHISPER + WAVLM → LLAMA 3 → PIPER
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
                turns.map((turn) => {
                  const ev = getEmotionView(turn.emotion);

                  return (
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
                          {ev && (
                            <span
                              className={`bubble-badge emotion-${ev.group}`}
                              title={`model guess: ${ev.raw} ${ev.pct}%`}
                            >
                              {ev.emoji} {ev.text}{' '}
                              {ev.sure ? `${ev.pct}%` : `(guess ${ev.raw} ${ev.pct}%)`}
                            </span>
                          )}
                        </div>
                        <div className="bubble-text">{turn.userText}</div>
                      </div>

                      <div
                        className={`bubble bubble-ai ai-${turn.aiState}${
                          turn.voice?.speaking ? ' ai-voice-live' : ''
                        }`}
                      >
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
                          {turn.voice && (
                            <span
                              className={`bubble-badge voice-${turn.voice.style}${
                                turn.voice.speaking ? ' voice-live' : ''
                              }`}
                              title={`${turn.voice.sentences ?? 0} sentences · ${formatMs(turn.voice.audioMs)} audio`}
                            >
                              {turn.voice.speaking ? '🔊' : (VOICE_STYLE_ICON[turn.voice.style] ?? '🔈')}{' '}
                              {turn.voice.style} voice
                            </span>
                          )}
                          {turn.voice?.ttfaMs !== null && turn.voice?.ttfaMs !== undefined && (
                            <span className={`bubble-badge ${getLatencyClass(turn.voice.ttfaMs)}`}>
                              TTFA {turn.voice.ttfaMs} ms
                            </span>
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
                  );
                })
              )}
              <div ref={transcriptEndRef} />
            </div>
          </section>

          {/* LATENCY BREAKDOWN (Week 3 Day 4) */}

          <section className="card full">
            <div className="card-title">
              <h2>Latency Breakdown</h2>
              <span>SPEECH END → FIRST AI AUDIO · TARGET {LATENCY_TARGET_MS} MS</span>
            </div>

            <div className="audit-summary">
              <div className="metric">
                <div className="metric-label">First audio p50</div>
                <div className={`metric-value ${getLatencyClass(p50Ttfa)}`}>{formatMs(p50Ttfa)}</div>
                <div className="metric-sub">half of replies start faster</div>
              </div>
              <div className="metric">
                <div className="metric-label">First audio p95</div>
                <div className={`metric-value ${getLatencyClass(p95Ttfa)}`}>{formatMs(p95Ttfa)}</div>
                <div className="metric-sub">worst-case (95%)</div>
              </div>
              <div className="metric">
                <div className="metric-label">Slowest stage</div>
                <div className="metric-value">
                  {turnMetrics.length > 0 && slowestStage ? slowestStage.label : '—'}
                </div>
                <div className="metric-sub">
                  {turnMetrics.length > 0 && slowestStage
                    ? `avg ${formatMs(slowestStage.ms)}`
                    : 'needs one reply'}
                </div>
              </div>
              <div className="metric">
                <div className="metric-label">Turns measured</div>
                <div className="metric-value">{turnMetrics.length}</div>
                <div className="metric-sub">also saved in Backend/logs/latency.csv</div>
              </div>
            </div>

            <div className="latency-legend">
              {avgStages.map((stage) => (
                <span className="latency-legend-item" key={stage.key}>
                  <span className={`latency-swatch ${stage.cls}`} />
                  {stage.label}
                  {stage.ms !== null && <span className="latency-legend-ms"> · avg {stage.ms} ms</span>}
                </span>
              ))}
            </div>

            <div className="latency-bars">
              {recentMetrics.length === 0 ? (
                <div className="transcript-placeholder">
                  Say something. Each reply's time from end of speech to the first AI audio appears here.
                </div>
              ) : (
                recentMetrics.map((m, index) => {
                  const stages = getStages(m);
                  const end = m.ttfa_ms ?? m.ttft_ms ?? m.stt_ms ?? 0;

                  return (
                    <div className="latency-row" key={m.id}>
                      <span className="latency-row-label">
                        #{turnMetrics.length - recentMetrics.length + index + 1}
                      </span>
                      <div className="latency-track">
                        {LATENCY_STAGES.map((stage) =>
                          stages[stage.key] > 0 ? (
                            <div
                              key={stage.key}
                              className={`latency-segment ${stage.cls}`}
                              style={{ width: `${(stages[stage.key] / breakdownScale) * 100}%` }}
                              title={`${stage.label}: ${stages[stage.key]} ms`}
                            />
                          ) : null
                        )}
                        <div
                          className="latency-target"
                          style={{ left: `${(LATENCY_TARGET_MS / breakdownScale) * 100}%` }}
                          title={`target ${LATENCY_TARGET_MS} ms`}
                        />
                      </div>
                      <span className={`latency-row-total ${getLatencyClass(end)}`}>
                        {formatMs(end)}
                      </span>
                    </div>
                  );
                })
              )}
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