import { useState, useRef, useEffect } from 'react';
import './App.css';
import './ui-polish.css';
import { DEMO_SESSION } from './demoSession';

const BACKEND_URL = `${import.meta.env.VITE_BACKEND_URL}/webrtc/offer`;

// Week 3 Day 6: backend status + which models / devices it runs
const HEALTH_URL = `${import.meta.env.VITE_BACKEND_URL}/health`;

// Week 4 demo mode: links shown when the GPU backend is offline
const DEMO_VIDEO_URL = import.meta.env.VITE_DEMO_VIDEO_URL || '';
const REPO_URL = 'https://github.com/Ankit-builds1/Auralis';
const HEALTH_POLL_MS = 15000;

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

// ---------------------------------------------------------
// Week 4 Day 1: live waveform (user mic + Auralis voice)
// ---------------------------------------------------------

// Your waveform takes the colour of the emotion Auralis detected last
const WAVE_EMOTION_COLORS = {
  angry: '#f87171',
  disgust: '#4ade80',
  fear: '#c084fc',
  happy: '#facc15',
  neutral: '#2dd4bf',
  sad: '#60a5fa',
};

// Auralis's waveform takes the colour of the voice style it speaks in
const WAVE_VOICE_COLORS = {
  calm: '#5eead4',
  soft: '#93c5fd',
  normal: '#a78bfa',
};

// RMS above this = that side is "talking" (bright + glow), below = dimmed
const WAVE_ACTIVE_RMS = 0.015;
// Speech is quiet in raw samples; scale it up so the shape is visible
const WAVE_GAIN = 4;

// Draws one waveform on a canvas. Returns the RMS level (0 when no audio).
// Runs every animation frame, straight on the canvas (no React re-render).
const drawWave = (canvas, wave, color) => {
  if (!canvas) return 0;

  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (width === 0 || height === 0) return 0;

  if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
  }

  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const mid = height / 2;

  // faint centre line, always visible
  ctx.globalAlpha = 0.25;
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  ctx.shadowBlur = 0;
  ctx.beginPath();
  ctx.moveTo(0, mid);
  ctx.lineTo(width, mid);
  ctx.stroke();

  if (!wave) {
    ctx.globalAlpha = 1;
    return 0;
  }

  const { analyser, buffer } = wave;
  analyser.getFloatTimeDomainData(buffer);

  let sum = 0;
  for (let i = 0; i < buffer.length; i++) sum += buffer[i] * buffer[i];
  const rms = Math.sqrt(sum / buffer.length);
  const active = rms > WAVE_ACTIVE_RMS;

  ctx.globalAlpha = active ? 1 : 0.4;
  ctx.lineWidth = active ? 2.2 : 1.5;
  ctx.shadowColor = color;
  ctx.shadowBlur = active ? 14 : 0;
  ctx.beginPath();

  const step = width / (buffer.length - 1);
  for (let i = 0; i < buffer.length; i++) {
    const value = Math.max(-1, Math.min(1, buffer[i] * WAVE_GAIN));
    const y = mid - value * (mid - 4);
    if (i === 0) ctx.moveTo(0, y);
    else ctx.lineTo(i * step, y);
  }
  ctx.stroke();

  ctx.globalAlpha = 1;
  ctx.shadowBlur = 0;
  return rms;
};

// ---------------------------------------------------------
// Week 4 Day 3: premium "Experience" view (voice orb + clean chat)
// ---------------------------------------------------------

const VIEW_STORAGE_KEY = 'auralis_view_v1';

const loadView = () => {
  try {
    return localStorage.getItem(VIEW_STORAGE_KEY) === 'engineer' ? 'engineer' : 'experience';
  } catch {
    return 'experience';
  }
};

const hexToRgba = (hex, alpha) => {
  const h = hex.replace('#', '');
  const n = parseInt(h.length === 3 ? h.split('').map((c) => c + c).join('') : h, 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
};

// Reads one analyser and returns its RMS level (0 when there is no audio)
const readLevel = (wave) => {
  if (!wave) return 0;
  wave.analyser.getFloatTimeDomainData(wave.buffer);
  let sum = 0;
  for (let i = 0; i < wave.buffer.length; i++) sum += wave.buffer[i] * wave.buffer[i];
  return Math.sqrt(sum / wave.buffer.length);
};

// The voice orb: breathes when idle, swells and ripples with whoever is
// talking, coloured by the detected emotion / Auralis's voice style.
const drawOrb = (canvas, level, color, wave, time) => {
  if (!canvas) return;

  const dpr = window.devicePixelRatio || 1;
  const size = canvas.clientWidth;
  if (size === 0) return;
  if (canvas.width !== Math.round(size * dpr)) {
    canvas.width = Math.round(size * dpr);
    canvas.height = Math.round(size * dpr);
  }

  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, size, size);

  const c = size / 2;
  const breathe = Math.sin(time / 1400) * 0.025;
  const base = size * 0.22;
  const radius = base * (1 + breathe + Math.min(level * 8, 0.32));

  // soft outer glow, always fading to nothing before the canvas edge
  const glow = ctx.createRadialGradient(c, c, radius * 0.5, c, c, c * 0.98);
  glow.addColorStop(0, hexToRgba(color, 0.30 + Math.min(level * 4, 0.22)));
  glow.addColorStop(0.55, hexToRgba(color, 0.08));
  glow.addColorStop(1, hexToRgba(color, 0));
  ctx.fillStyle = glow;
  ctx.beginPath();
  ctx.arc(c, c, c, 0, Math.PI * 2);
  ctx.fill();

  // rippled edge from the live voice: loudness of small slices of the
  // waveform, smoothed around the circle (a soft blob, never spiky)
  const points = 72;
  const bumps = new Array(points).fill(0);
  if (wave) {
    const slice = Math.floor(wave.buffer.length / points);
    for (let i = 0; i < points; i++) {
      let sum = 0;
      for (let j = 0; j < slice; j++) sum += Math.abs(wave.buffer[i * slice + j] || 0);
      bumps[i] = Math.min(1, (sum / slice) * 6);
    }
    for (let pass = 0; pass < 3; pass++) {
      const copy = bumps.slice();
      for (let i = 0; i < points; i++) {
        bumps[i] = (copy[(i - 1 + points) % points] + copy[i] * 2 + copy[(i + 1) % points]) / 4;
      }
    }
  }

  ctx.beginPath();
  for (let i = 0; i <= points; i++) {
    const angle = (i / points) * Math.PI * 2;
    const wobble = Math.sin(angle * 3 + time / 900) * 1.5 + bumps[i % points] * radius * 0.16;
    const r = radius + wobble;
    const x = c + Math.cos(angle) * r;
    const y = c + Math.sin(angle) * r;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.closePath();

  const body = ctx.createRadialGradient(
    c - radius * 0.35, c - radius * 0.4, radius * 0.1, c, c, radius * 1.05
  );
  body.addColorStop(0, 'rgba(255, 255, 255, 0.92)');
  body.addColorStop(0.28, hexToRgba(color, 0.95));
  body.addColorStop(1, hexToRgba(color, 0.18));
  ctx.fillStyle = body;
  ctx.shadowColor = color;
  ctx.shadowBlur = 40;
  ctx.fill();
  ctx.shadowBlur = 0;

  // thin highlight ring
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.18)';
  ctx.lineWidth = 1;
  ctx.stroke();
};

// Minimal line icons (premium look: no emoji in the Experience view)
const Icon = ({ name, size = 18 }) => {
  const paths = {
    mic: <><rect x="9" y="3" width="6" height="11" rx="3" /><path d="M5 11a7 7 0 0 0 14 0M12 18v3" /></>,
    stop: <rect x="6" y="6" width="12" height="12" rx="2.5" />,
    hand: <path d="M8 13V6.5a1.5 1.5 0 0 1 3 0V12m0-7a1.5 1.5 0 0 1 3 0v7m0-5.5a1.5 1.5 0 0 1 3 0V14a6 6 0 0 1-6 6h-.5a6 6 0 0 1-4.6-2.2L4.5 15a1.6 1.6 0 0 1 2.4-2.1L8 14" />,
    sliders: <path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0M14 4v4M8 10v4M16 16v4" />,
    sparkle: <path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z" />,
  };
  return (
    <svg
      width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"
    >
      {paths[name]}
    </svg>
  );
};

// ---------------------------------------------------------
// Week 4 feature: emotion map (arousal / valence) + training mode
// ---------------------------------------------------------

// Where each emotion sits on the circumplex: x = valence (unpleasant ->
// pleasant), y = arousal (calm -> excited). PROJECT.pdf: "classifying the
// speaker's emotional state (Arousal/Valence)".
const EMOTION_VA = {
  happy: [0.75, 0.45],
  neutral: [0.05, -0.05],
  sad: [-0.65, -0.55],
  angry: [-0.6, 0.8],
  fear: [-0.7, 0.6],
  disgust: [-0.7, 0.25],
};

// Week 4 feature: the 12 training scenarios (same ids as the backend's
// negotiation.SCENARIOS; the backend sends the full list once training starts)
const SCENARIO_LIST = [
  { id: 'office_lockin', name: 'Arjun', pronoun: 'him', title: 'Locked-in office worker', level: 'Hard', start: 85 },
  { id: 'flight_cancelled', name: 'Meera', pronoun: 'her', title: 'Cancelled flight at the airport', level: 'Medium', start: 75 },
  { id: 'exam_failed', name: 'Rohan', pronoun: 'him', title: 'Student who failed an exam', level: 'Easy', start: 70 },
  { id: 'noisy_neighbour', name: 'Mr. Sharma', pronoun: 'him', title: 'Angry neighbour at midnight', level: 'Easy', start: 70 },
  { id: 'hospital_wait', name: 'Kavya', pronoun: 'her', title: 'Worried daughter at the hospital', level: 'Medium', start: 80 },
  { id: 'loan_rejected', name: 'Vikram', pronoun: 'him', title: 'Loan rejected at the bank', level: 'Medium', start: 80 },
  { id: 'fake_complaint', name: 'Sameer', pronoun: 'him', title: 'Delivery partner and a fake complaint', level: 'Easy', start: 70 },
  { id: 'unpaid_salary', name: 'Raju', pronoun: 'him', title: 'Unpaid factory worker at the gate', level: 'Hard', start: 85 },
  { id: 'child_late', name: 'Neha', pronoun: 'her', title: 'Parent whose child is late from school', level: 'Hard', start: 90 },
  { id: 'road_accident', name: 'Ajay', pronoun: 'him', title: 'Driver after a minor road accident', level: 'Medium', start: 80 },
  { id: 'online_fraud', name: 'Mrs. Iyer', pronoun: 'her', title: 'Senior citizen after an online fraud', level: 'Medium', start: 75 },
  { id: 'eviction', name: 'Karan', pronoun: 'him', title: 'Tenant told to leave suddenly', level: 'Medium', start: 75 },
];
const findScenario = (id) => SCENARIO_LIST.find((x) => x.id === id) || SCENARIO_LIST[0];

// Training caller: agitation 100 -> upset & tense, 0 -> calm & settled
const agitationToVA = (a) => [0.55 - (a / 100) * 1.35, (a / 100) * 1.5 - 0.7];

const agitationColor = (a) =>
  a >= 70 ? '#f87171' : a >= 45 ? '#fbbf24' : a > 30 ? '#a3e635' : '#34d399';

const EmotionMap = ({ you, caller, callerName = 'Caller' }) => {
  const size = 260;
  const pad = 22;
  const toXY = ([v, a]) => [
    pad + ((v + 1) / 2) * (size - 2 * pad),
    pad + ((1 - a) / 2) * (size - 2 * pad),
  ];
  const trail = (pts, color, label) => {
    if (!pts.length) return null;
    const xy = pts.map(toXY);
    return (
      <g>
        <polyline
          points={xy.map(([x, y]) => `${x},${y}`).join(' ')}
          fill="none" stroke={color} strokeOpacity="0.45" strokeWidth="1.5"
        />
        {xy.map(([x, y], i) => (
          <circle
            key={i} cx={x} cy={y}
            r={i === xy.length - 1 ? 7 : 3.5}
            fill={color}
            fillOpacity={i === xy.length - 1 ? 1 : 0.25 + (0.6 * (i + 1)) / xy.length}
          />
        ))}
        <text
          x={xy[xy.length - 1][0] + 10} y={xy[xy.length - 1][1] + 4}
          fill={color} fontSize="11" fontWeight="600"
        >
          {label}
        </text>
      </g>
    );
  };
  const c = size / 2;
  return (
    <svg viewBox={`0 0 ${size} ${size}`} className="va-map" role="img"
      aria-label="Emotion map: valence left to right, arousal bottom to top">
      <rect x={pad} y={pad} width={size - 2 * pad} height={size - 2 * pad} rx="14"
        fill="rgba(255,255,255,0.02)" stroke="rgba(255,255,255,0.08)" />
      <line x1={pad} y1={c} x2={size - pad} y2={c} stroke="rgba(255,255,255,0.10)" />
      <line x1={c} y1={pad} x2={c} y2={size - pad} stroke="rgba(255,255,255,0.10)" />
      <text x={size - pad - 4} y={c - 6} textAnchor="end" className="va-axis">pleasant</text>
      <text x={pad + 4} y={c - 6} className="va-axis">unpleasant</text>
      <text x={c + 6} y={pad + 12} className="va-axis">excited</text>
      <text x={c + 6} y={size - pad - 6} className="va-axis">calm</text>
      {Object.entries(EMOTION_VA).map(([k, va]) => {
        const [x, y] = toXY(va);
        return <text key={k} x={x} y={y} textAnchor="middle" className="va-label">{k}</text>;
      })}
      {trail(caller, '#f87171', callerName)}
      {trail(you, '#38e1ff', 'You')}
    </svg>
  );
};

// Week 4 demo mode: replay a sample training session without the backend
const DemoReplay = ({ onClose }) => {
  const s = DEMO_SESSION;
  const [step, setStep] = useState(0); // number of turns shown
  const [playing, setPlaying] = useState(true);
  const total = s.turns.length;

  useEffect(() => {
    if (!playing || step >= total) return undefined;
    const t = setTimeout(() => setStep((n) => n + 1), step === 0 ? 1200 : 2600);
    return () => clearTimeout(t);
  }, [playing, step, total]);

  const shown = s.turns.slice(0, step);
  const agitation = step ? shown[step - 1].agitation : s.start;
  const done = step >= total;
  const calm = shown.filter((t) => t.reasons.some((r) => r.startsWith('calm tone'))).length;
  const empathy = shown.filter((t) => t.reasons.some((r) => r.startsWith('de-escalating'))).length;
  const escalating = shown.filter((t) => t.reasons.some((r) => r.startsWith('escalating'))).length;

  return (
    <section className="demo-replay" aria-label="Sample training session replay">
      <div className="demo-replay-head">
        <div>
          <p className="demo-kicker">Sample session · negotiation training</p>
          <h3>{s.scenario.name}: {s.scenario.title} <span className={`level-badge level-${s.scenario.level.toLowerCase()}`}>{s.scenario.level}</span></h3>
          <p className="xp-muted">{s.scenario.brief}</p>
        </div>
        <div className="demo-replay-controls">
          {done ? (
            <button className="xp-btn xp-btn-ghost" onClick={() => { setStep(0); setPlaying(true); }}>Replay</button>
          ) : (
            <button className="xp-btn xp-btn-ghost" onClick={() => setPlaying((p) => !p)}>{playing ? 'Pause' : 'Play'}</button>
          )}
          <button className="xp-link" onClick={onClose}>Close</button>
        </div>
      </div>

      <div className="agitation">
        <div className="agitation-head">
          <span>{s.scenario.name}'s agitation</span>
          <strong style={{ color: agitationColor(agitation) }}>{agitation}</strong>
        </div>
        <div className="agitation-track">
          <div className="agitation-goal" style={{ left: `${s.goal}%` }} title="goal" />
          <div className="agitation-fill" style={{ width: `${agitation}%`, background: agitationColor(agitation) }} />
        </div>
      </div>

      <div className="demo-replay-chat">
        <div className="demo-msg demo-msg-caller"><strong>{s.scenario.name}</strong>{s.opening}</div>
        {shown.map((t, i) => (
          <div key={i} className="demo-turn">
            <div className="demo-msg demo-msg-user">
              <strong>Trainee <span className="demo-emotion">{EMOTION_EMOJI[t.emotion]} {t.emotion}</span></strong>
              {t.user}
              <span className={t.delta > 0 ? 'delta-up' : 'delta-down'}>
                {t.delta > 0 ? '+' : ''}{t.delta} · {t.reasons.join(' · ')}
              </span>
            </div>
            <div className="demo-msg demo-msg-caller"><strong>{s.scenario.name}</strong>{t.caller}</div>
          </div>
        ))}
      </div>

      {done && (
        <div className="demo-score">
          <strong className="score-pass">De-escalated</strong>
          <span>{s.start} → {agitation} agitation</span>
          <span>{total} turns</span>
          <span>{Math.round((calm / total) * 100)}% calm tone</span>
          <span>{empathy} empathy turns</span>
          <span>{escalating} escalating turn</span>
        </div>
      )}
      <p className="xp-muted demo-note">
        Sample replay in the browser, following the flow of our live test. With the backend online, the
        trainee speaks and Auralis (Llama 3.2 + Piper) plays the caller live.
      </p>
    </section>
  );
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
  // Week 3 Day 5: when the WORDS decided, the backend already required the
  // text model to be >= 60% sure, and the fused number is naturally a bit
  // lower (0.7 x text + 0.3 x voice) -> trust the backend's "confident" flag.
  const sure =
    emotion.confident &&
    (emotion.source === 'words' || emotion.confidence >= EMOTION_SHOW_MIN);

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

// Week 3 Day 5: emotion test mode + voice/words fusion
const EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad'];
const EMOTION_GROUP = {
  angry: 'upset', disgust: 'upset', fear: 'upset',
  sad: 'low',
  happy: 'okay', neutral: 'okay',
};
const TEST_RECENT_ROWS = 8;

// "neutral 93%" style text for one side of the fusion
const formatGuess = (label, confidence) =>
  label ? `${label} ${Math.round((confidence ?? 0) * 100)}%` : '—';

const pctOf = (hit, total) => (total > 0 ? `${Math.round((hit / total) * 100)}%` : '—');

// ---------------------------------------------------------------------------
// Week 3 Day 6: system info chips + latency audit (compare machines)
// ---------------------------------------------------------------------------

// Latency runs are kept in this browser (localStorage) so a run measured on
// Priya's CPU backend (via ngrok) can be compared with one on the GPU laptop.
const AUDIT_STORAGE_KEY = 'auralis_latency_runs_v1';
// Week 4: record & replay of real sessions (backend messages + timing)
const SESSION_STORAGE_KEY = 'auralis_saved_sessions_v1';
const loadSavedSessions = () => {
  try {
    return JSON.parse(localStorage.getItem(SESSION_STORAGE_KEY) || '[]');
  } catch {
    return [];
  }
};
const AUDIT_MAX_RUNS = 10;

const loadRuns = () => {
  try {
    const runs = JSON.parse(localStorage.getItem(AUDIT_STORAGE_KEY) || '[]');
    return Array.isArray(runs) ? runs : [];
  } catch {
    return [];
  }
};

const storeRuns = (runs) => {
  try {
    localStorage.setItem(AUDIT_STORAGE_KEY, JSON.stringify(runs));
  } catch {
    // storage blocked (private window) -> runs live only until reload
  }
};

// Chips for the header. Older backends only send true/false flags, newer
// ones (Backend Week 3 Day 6) also send model names and devices.
const getSystemChips = (h) => {
  if (!h) return [];
  const chips = [];
  const stt = h.stt && typeof h.stt === 'object' ? h.stt : null;

  chips.push({
    key: 'stt',
    label: 'STT',
    value: stt ? [stt.model ?? 'whisper', stt.device].filter(Boolean).join(' · ') : 'whisper',
    ok: true,
  });
  const gpu = h.gpu ?? h.device?.gpu;
  if (gpu) chips.push({ key: 'gpu', label: 'GPU', value: gpu, ok: true });
  if (h.emotion !== undefined) {
    chips.push({
      key: 'emotion',
      label: 'Voice emotion',
      value: h.emotion ? h.emotion_model ?? 'on' : 'off',
      ok: !!h.emotion,
    });
  }
  if (h.text_emotion !== undefined) {
    chips.push({ key: 'text', label: 'Words', value: h.text_emotion ? 'on' : 'off', ok: !!h.text_emotion });
  }
  if (h.tts !== undefined) {
    chips.push({ key: 'tts', label: 'TTS', value: h.tts ? h.tts_engine ?? 'on' : 'off', ok: !!h.tts });
  }
  const llm = h.llm?.model ?? h.llm_model;
  if (llm) chips.push({ key: 'llm', label: 'LLM', value: llm, ok: true });
  return chips;
};

// Short machine name for a saved run, e.g. "cuda · RTX 2050"
const describeMachine = (h) => {
  if (!h) return 'unknown backend';
  const device = h.stt?.device;
  const gpu = h.gpu ?? h.device?.gpu;
  return [device ? `STT on ${device}` : null, gpu].filter(Boolean).join(' · ') || 'backend';
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

  // Week 3 Day 5: emotion test mode
  // testResults: { id, time, target, final, voice, words, source, transcript }
  const [testMode, setTestMode] = useState(false);
  const [testTarget, setTestTarget] = useState(null);
  const [testResults, setTestResults] = useState([]);
  // refs, because the DataChannel handler is created once at connect time
  const testModeRef = useRef(false);
  const testTargetRef = useRef(null);
  const lastTranscriptRef = useRef('');
  const testIdRef = useRef(0);

  // Week 3 Day 3: time to first audio (speech end -> Auralis starts talking)
  const [ttfas, setTtfas] = useState([]);
  const [echoCount, setEchoCount] = useState(0);

  // Week 4 Day 2: interruptions (barge-in from the backend, or STOP button)
  // { id, reason: 'barge_in'|'manual', stopMs, spokenMs }
  const [interruptions, setInterruptions] = useState([]);
  // true while Auralis's voice is cut locally (until its next reply starts)
  const aiCutRef = useRef(false);
  const aiUnmuteTimerRef = useRef(null);
  // id of the turn whose reply was cut (its late tokens / audio stay silent)
  const interruptedTurnIdRef = useRef(null);
  // true while Auralis is cut: UI shows STOPPED even if the backend keeps
  // sending the rest of the old reply (backend without barge-in)
  const [aiStopped, setAiStopped] = useState(false);

  // Week 3 Day 6: backend health (models / devices) + saved latency runs
  const [health, setHealth] = useState(null);
  const [healthError, setHealthError] = useState(null);
  const [healthMs, setHealthMs] = useState(null);
  // Week 4 demo mode: offline banner can be closed for this visit
  const [demoBannerClosed, setDemoBannerClosed] = useState(false);
  const [showReplay, setShowReplay] = useState(false);
  const [runs, setRuns] = useState(loadRuns);
  const [runLabel, setRunLabel] = useState('');

  // Week 4 Day 3: 'experience' (premium demo view) or 'engineer' (all metrics)
  const [view, setView] = useState(loadView);

  // Week 4 feature: 'assistant' or 'training' (crisis negotiation trainer)
  const [appMode, setAppMode] = useState('assistant');
  const appModeRef = useRef('assistant');
  const modeSyncedRef = useRef(false);
  // { scenario, agitation, start, goal, history: [..], steps: [{delta, reasons}] }
  const [training, setTraining] = useState(null);
  const [showScore, setShowScore] = useState(false);
  // Week 4: which of the 12 scenarios is picked
  const [scenarioId, setScenarioId] = useState('office_lockin');
  const scenarioIdRef = useRef('office_lockin');

  // Week 3 Day 4: one latency breakdown per turn (from 'turn_metrics')
  const [turnMetrics, setTurnMetrics] = useState([]);

  // Week 3 Day 1: AI audio (backend -> browser) status
  const [aiAudio, setAiAudio] = useState('waiting');
  const [aiPackets, setAiPackets] = useState(0);

  // Week 4: session recorder (every backend message of the live session,
  // with its time) and replay of a saved session through the same handler
  const recordingRef = useRef({ startedAt: null, mode: 'assistant', events: [] });
  const [recordCount, setRecordCount] = useState(0);
  const [savedSessions, setSavedSessions] = useState(loadSavedSessions);
  const [selectedSession, setSelectedSession] = useState('');
  const [replay, setReplay] = useState(null); // { name, date, done, total, shown }
  const replayTimersRef = useRef([]);
  const handlerRef = useRef(null);

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

  // Week 4 Day 1: live waveform. Analysers live in refs and the canvases are
  // drawn every animation frame directly (no React state -> no re-renders).
  const micWaveRef = useRef(null);      // { analyser, buffer } for your mic
  const aiWaveRef = useRef(null);       // { analyser, buffer, source } for Auralis
  const micCanvasRef = useRef(null);
  const aiCanvasRef = useRef(null);
  const micWavePanelRef = useRef(null);
  const aiWavePanelRef = useRef(null);
  const micWaveLabelRef = useRef(null);
  const aiWaveLabelRef = useRef(null);
  const userWaveColorRef = useRef(WAVE_EMOTION_COLORS.neutral);
  const aiWaveColorRef = useRef(WAVE_VOICE_COLORS.normal);

  // Week 4 Day 3: Experience view orb + chat scroll
  const orbCanvasRef = useRef(null);
  const orbLevelRef = useRef(0);
  const xpChatRef = useRef(null);

  // Week 4 Day 1: one animation loop draws both waveforms
  useEffect(() => {
    let raf = null;

    const paint = (canvas, wave, color, panel, label, idleText) => {
      const rms = drawWave(canvas, wave, color);
      const active = rms > WAVE_ACTIVE_RMS;

      if (panel) {
        panel.classList.toggle('wave-active', active);
        panel.style.setProperty('--wave-color', color);
      }
      if (label) {
        const db = rms > 0 ? 20 * Math.log10(rms) : -100;
        const text = !wave ? idleText : active ? `talking · ${db.toFixed(0)} dB` : 'quiet';
        if (label.textContent !== text) label.textContent = text;
      }
    };

    const tick = () => {
      paint(
        micCanvasRef.current, micWaveRef.current, userWaveColorRef.current,
        micWavePanelRef.current, micWaveLabelRef.current, 'mic off'
      );
      // Week 4 Day 2: muted Auralis = flat line, even if old audio still arrives
      const aiCut = aiCutRef.current && aiWaveRef.current;
      paint(
        aiCanvasRef.current, aiCut ? null : aiWaveRef.current, aiWaveColorRef.current,
        aiWavePanelRef.current, aiWaveLabelRef.current, aiCut ? 'stopped · muted' : 'not connected'
      );

      // Week 4 Day 3: voice orb follows whoever is talking
      if (orbCanvasRef.current) {
        const micLevel = readLevel(micWaveRef.current);
        const aiLevel = aiCut ? 0 : readLevel(aiWaveRef.current);
        const aiLeads = aiLevel > WAVE_ACTIVE_RMS && aiLevel >= micLevel;
        const target = Math.max(micLevel, aiLevel);
        // smooth: fast attack, slow release
        const prev = orbLevelRef.current;
        orbLevelRef.current = target > prev ? prev + (target - prev) * 0.5 : prev * 0.92;
        drawOrb(
          orbCanvasRef.current,
          orbLevelRef.current,
          aiLeads ? aiWaveColorRef.current : userWaveColorRef.current,
          aiLeads ? aiWaveRef.current : micLevel > WAVE_ACTIVE_RMS ? micWaveRef.current : null,
          performance.now()
        );
      }

      raf = requestAnimationFrame(tick);
    };

    tick();
    return () => cancelAnimationFrame(raf);
  }, []);

  // Auto-scroll transcript to the newest line
  // Week 3 Day 6: ask the backend what it is running, every 15 s
  useEffect(() => {
    let cancelled = false;

    const check = async () => {
      const started = performance.now();
      try {
        const res = await fetch(HEALTH_URL, {
          headers: { 'ngrok-skip-browser-warning': 'true' },
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        // ngrok / proxies can answer even when OUR backend is not there,
        // so only trust a real Auralis health reply.
        if (data?.status !== 'ok') throw new Error('not an Auralis backend');
        if (cancelled) return;
        setHealth(data);
        setHealthMs(Math.round(performance.now() - started));
        setHealthError(null);
      } catch (err) {
        if (cancelled) return;
        setHealth(null);
        setHealthMs(null);
        setHealthError(err?.message || 'offline');
      }
    };

    check();
    const timer = setInterval(check, HEALTH_POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    // Week 4 Day 3: Experience chat scrolls inside its own box
    if (xpChatRef.current) xpChatRef.current.scrollTop = xpChatRef.current.scrollHeight;
  }, [turns]);

  // Week 4 Day 3: remember the chosen view in this browser
  useEffect(() => {
    try {
      localStorage.setItem(VIEW_STORAGE_KEY, view);
    } catch {
      // storage blocked: the view just resets on reload
    }
  }, [view]);

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

  // Week 4 Day 1: waveform colours follow the last emotion / voice style
  const lastVoiceStyle =
    [...turns].reverse().find((t) => t.voice?.style)?.voice.style ?? 'normal';
  const userWaveColor = WAVE_EMOTION_COLORS[lastEmotionView?.text] ?? WAVE_EMOTION_COLORS.neutral;
  const aiWaveColor = WAVE_VOICE_COLORS[lastVoiceStyle] ?? WAVE_VOICE_COLORS.normal;

  useEffect(() => {
    userWaveColorRef.current = userWaveColor;
    aiWaveColorRef.current = aiWaveColor;
  }, [userWaveColor, aiWaveColor]);

  // Week 3 Day 5: emotion test mode scores
  const testTotal = testResults.length;
  const testFinalHit = testResults.filter((r) => r.final === r.target).length;
  const testGroupHit = testResults.filter(
    (r) => EMOTION_GROUP[r.final] === EMOTION_GROUP[r.target]
  ).length;
  const testVoiceHit = testResults.filter((r) => r.voice === r.target).length;
  const testWordsRows = testResults.filter((r) => r.words);
  const testWordsHit = testWordsRows.filter((r) => r.words === r.target).length;
  const testByWords = testResults.filter((r) => r.source === 'words').length;
  const testMatrix = EMOTION_LABELS.map((target) =>
    EMOTION_LABELS.map(
      (pred) => testResults.filter((r) => r.target === target && r.final === pred).length
    )
  );
  const testRecent = testResults.slice(-TEST_RECENT_ROWS).reverse();

  // Week 4 Day 2: interruption stats
  const interruptStopTimes = interruptions
    .map((i) => i.stopMs)
    .filter((ms) => typeof ms === 'number');
  const avgInterruptStop = average(interruptStopTimes);
  const bargeInCount = interruptions.filter((i) => i.reason === 'barge_in').length;
  const manualStopCount = interruptions.length - bargeInCount;
  const aiCanBeStopped =
    vadState === 'ai_speaking' ||
    turns.some((t) => t.aiState === 'streaming' || t.voice?.speaking);

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

  // Week 3 Day 6: system chips + latency run comparison
  const systemChips = getSystemChips(health);
  const fastestRun = runs.reduce(
    (best, run) => (run.p50 !== null && (best === null || run.p50 < best.p50) ? run : best),
    null
  );
  const slowestRun = runs.reduce(
    (worst, run) => (run.p50 !== null && (worst === null || run.p50 > worst.p50) ? run : worst),
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
  // Called from the DataChannel handler for every emotion event.
  // Reads refs (not state) because that handler is created once at connect time.
  const recordTest = (emotion, shown) => {
    const target = testTargetRef.current;
    if (!testModeRef.current || !target) return;

    testIdRef.current += 1;
    const id = testIdRef.current;
    const transcript = lastTranscriptRef.current;

    setTestResults((previous) => [
      ...previous,
      {
        id,
        time: new Date().toLocaleTimeString(),
        target,
        final: shown,
        voice: emotion.voiceLabel,
        words: emotion.textLabel,
        source: emotion.source ?? 'voice',
        transcript,
      },
    ]);
    addEvent(`Test: wanted ${target} → got ${shown} ${shown === target ? '✓' : '✗'}`);
  };

  const updateCurrentTurn = (update) => {
    const id = currentTurnIdRef.current;
    if (!id) return;

    setTurns((previous) =>
      previous.map((t) => (t.id === id ? { ...t, ...update(t) } : t))
    );
  };

  // ---------------------------------------------------------
  // Week 4 Day 2: INTERRUPTION (barge-in) helpers
  // ---------------------------------------------------------

  // Silence Auralis in the browser right away. The backend stops sending
  // audio too, but a few hundred ms are already buffered on the way.
  const cutAiVoice = () => {
    aiCutRef.current = true;
    setAiStopped(true);
    if (aiUnmuteTimerRef.current) {
      clearTimeout(aiUnmuteTimerRef.current);
      aiUnmuteTimerRef.current = null;
    }
    if (audioRef.current) audioRef.current.muted = true;

    // short "cut" flash on the AURALIS waveform
    const panel = aiWavePanelRef.current;
    if (panel) {
      panel.classList.remove('wave-cut');
      void panel.offsetWidth; // restart the CSS animation
      panel.classList.add('wave-cut');
    }
  };

  // Let Auralis be heard again (new reply starting, or old one fully drained)
  const restoreAiVoice = (delayMs = 0) => {
    if (aiUnmuteTimerRef.current) clearTimeout(aiUnmuteTimerRef.current);
    aiUnmuteTimerRef.current = setTimeout(() => {
      aiCutRef.current = false;
      aiUnmuteTimerRef.current = null;
      setAiStopped(false);
      if (audioRef.current) audioRef.current.muted = false;
    }, delayMs);
  };

  // Mark the current reply as cut off and record the interruption
  const markInterrupted = (reason, stopMs, spokenMs) => {
    updateCurrentTurn((t) => ({
      aiState: t.aiState === 'waiting' || t.aiState === 'error' ? t.aiState : 'interrupted',
      interrupt: { reason, stopMs },
      voice: t.voice ? { ...t.voice, speaking: false } : t.voice,
    }));
    setInterruptions((previous) =>
      [...previous, { id: `${Date.now()}-${previous.length}`, reason, stopMs, spokenMs }].slice(-50)
    );
  };

  // STOP AURALIS button: ask the backend to stop, and cut the voice locally
  // at once (works even with a backend that does not support it yet).
  const stopAuralis = () => {
    cutAiVoice();

    const channel = dataChannelRef.current;
    let sent = false;
    if (channel && channel.readyState === 'open') {
      try {
        channel.send(JSON.stringify({ type: 'interrupt', reason: 'manual' }));
        sent = true;
      } catch (err) {
        console.warn('[INTERRUPT] Could not send interrupt:', err);
      }
    }

    interruptedTurnIdRef.current = currentTurnIdRef.current;
    markInterrupted('manual', null, null);
    setStatus('Auralis stopped');
    addEvent(
      sent
        ? 'You stopped Auralis (interrupt sent to backend)'
        : 'You stopped Auralis (voice muted here; not connected)'
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

      // Week 4 Day 1: the waveform reads the same analyser with its own buffer
      micWaveRef.current = { analyser, buffer: new Float32Array(analyser.fftSize) };

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

    // Week 4: tell the backend which mode we are in, once per connection
    // (the first backend message proves its audio processor exists)
    if (!modeSyncedRef.current) {
      const channel = dataChannelRef.current;
      if (channel && channel.readyState === 'open') {
        try {
          channel.send(JSON.stringify({ type: 'set_mode', mode: appModeRef.current, scenario: scenarioIdRef.current }));
          modeSyncedRef.current = true;
        } catch (err) {
          console.warn('[MODE] could not send mode:', err);
        }
      }
    }

    switch (data.type) {
      // -------- Week 4 feature: training mode --------
      case 'mode':
        if (data.mode === 'training') {
          setTraining({
            scenario: data.scenario ? { ...findScenario(data.scenario.id), ...data.scenario } : null,
            agitation: data.agitation ?? 85,
            start: data.agitation ?? 85,
            goal: data.goal ?? 30,
            history: [data.agitation ?? 85],
            steps: [],
          });
          if (appModeRef.current !== 'training') {
            appModeRef.current = 'training';
            setAppMode('training');
          }
          if (data.scenario?.id) {
            scenarioIdRef.current = data.scenario.id;
            setScenarioId(data.scenario.id);
          }
          addEvent(`Training mode on: ${data.scenario?.title ?? 'scenario'}`);
        } else {
          setTraining(null);
          addEvent('Assistant mode');
        }
        return;

      case 'training_state':
        setTraining((t) => ({
          ...(t || { scenario: null, start: data.start ?? 85, goal: data.goal ?? 30, history: [], steps: [] }),
          agitation: data.agitation,
          history: [...(t?.history || []), data.agitation],
          steps: [...(t?.steps || []), { delta: data.delta, reasons: data.reasons || [] }],
        }));
        addEvent(`Agitation ${data.agitation} (${data.delta > 0 ? '+' : ''}${data.delta}): ${(data.reasons || []).join(', ')}`);
        return;

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
        lastTranscriptRef.current = text;

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
          // Week 3 Day 5: voice + words fusion (older backends don't send these)
          source: data.source === 'words' ? 'words' : data.source === 'voice' ? 'voice' : null,
          voiceLabel: typeof data.voice_label === 'string' ? data.voice_label : data.label,
          voiceConfidence:
            typeof data.voice_confidence === 'number' ? data.voice_confidence : data.confidence,
          textLabel: typeof data.text_label === 'string' ? data.text_label : null,
          textConfidence: typeof data.text_confidence === 'number' ? data.text_confidence : null,
          textMs: typeof data.text_ms === 'number' ? Math.round(data.text_ms) : null,
        };

        const view = getEmotionView(emotion);
        const by = emotion.source
          ? ` · by ${emotion.source} (voice ${formatGuess(emotion.voiceLabel, emotion.voiceConfidence)}, words ${formatGuess(emotion.textLabel, emotion.textConfidence)})`
          : '';

        setEmotions((previous) => [...previous, emotion].slice(-20));
        updateCurrentTurn(() => ({ emotion }));
        addEvent(
          view.sure
            ? `Emotion: ${emotion.label} (${view.pct}%, ${emotion.group}, ${formatMs(emotion.ms)})${by}`
            : `Emotion: neutral — low confidence, raw guess ${emotion.label} ${view.pct}% (${formatMs(emotion.ms)})${by}`
        );

        // Week 3 Day 5: test mode -> score this turn against the chosen emotion
        recordTest(emotion, view.text);
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
          updateCurrentTurn((t) =>
            t.aiState === 'interrupted'
              ? { ttftMs: ttft }
              : { aiState: 'streaming', ttftMs: ttft, aiText: t.aiText + token }
          );
        } else {
          // Week 4 Day 2: a cut-off reply stops growing
          updateCurrentTurn((t) =>
            t.aiState === 'interrupted' ? {} : { aiState: 'streaming', aiText: t.aiText + token }
          );
        }
        return;
      }

      case 'llm_done': {
        const total = typeof data.total_ms === 'number' ? Math.round(data.total_ms) : null;
        const ttft = typeof data.ttft_ms === 'number' ? Math.round(data.ttft_ms) : null;

        if (total !== null) {
          setTotals((previous) => [...previous, total].slice(-20));
        }

        updateCurrentTurn((t) =>
          t.aiState === 'interrupted'
            ? // Week 4 Day 2: keep only what was shown before the cut
              { ttftMs: t.ttftMs ?? ttft, totalMs: total }
            : {
                aiState: 'done',
                // Use the final text from the backend if it sent one
                aiText:
                  typeof data.text === 'string' && data.text.trim() ? data.text.trim() : t.aiText,
                ttftMs: t.ttftMs ?? ttft,
                totalMs: total,
              }
        );

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

        // Week 4 Day 2: a reply that was already cut stays silent;
        // a NEW reply makes Auralis audible again.
        const cutTurn = interruptedTurnIdRef.current === currentTurnIdRef.current;
        if (aiCutRef.current && !cutTurn) restoreAiVoice(0);

        updateCurrentTurn((t) =>
          t.aiState === 'interrupted'
            ? { voice: { style, ttfaMs: ttfa, speaking: false, sentences: 0, audioMs: 0 } }
            : { voice: { style, ttfaMs: ttfa, speaking: true, sentences: 0, audioMs: 0 } }
        );
        if (cutTurn) return;
        setStatus('Auralis is speaking...');
        addEvent(`AI voice started (TTFA ${formatMs(ttfa)}, ${style} voice)`);
        return;
      }

      case 'tts_sentence':
        updateCurrentTurn((t) => ({
          voice: {
            ...(t.voice || { style: 'normal', ttfaMs: null, speaking: t.aiState !== 'interrupted' }),
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

      // -------- Week 4 Day 2: barge-in (user spoke over Auralis) --------
      // Backend stopped the LLM + TTS for the current reply.
      // Fields (all optional): reason 'barge_in'|'manual', stop_ms (user
      // speech detected -> TTS stopped), spoken_ms (AI audio already sent)

      case 'interrupted': {
        const reason = data.reason === 'manual' ? 'manual' : 'barge_in';
        const stopMs = typeof data.stop_ms === 'number' ? Math.round(data.stop_ms) : null;
        const spokenMs = typeof data.spoken_ms === 'number' ? Math.round(data.spoken_ms) : null;

        cutAiVoice();

        // The STOP button already recorded this one -> just add the timing
        if (reason === 'manual' && interruptedTurnIdRef.current === currentTurnIdRef.current) {
          updateCurrentTurn((t) => ({ interrupt: { ...(t.interrupt || {}), reason, stopMs } }));
          setInterruptions((previous) => {
            if (previous.length === 0) return previous;
            const last = { ...previous[previous.length - 1], stopMs, spokenMs };
            return [...previous.slice(0, -1), last];
          });
        } else {
          interruptedTurnIdRef.current = currentTurnIdRef.current;
          markInterrupted(reason, stopMs, spokenMs);
        }

        setStatus(reason === 'manual' ? 'Auralis stopped' : 'Interrupted, listening...');
        addEvent(
          `Interrupted (${reason === 'manual' ? 'STOP button' : 'you spoke'}` +
          `${stopMs !== null ? `, voice stopped in ${stopMs} ms` : ''}` +
          `${spokenMs !== null ? `, after ${formatMs(spokenMs)} of speech` : ''})`
        );
        return;
      }

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

  handlerRef.current = handleDataChannelMessage;

  // ---------------------------------------------------------
  // SETUP DATA CHANNEL
  // ---------------------------------------------------------

  const setupDataChannel = (channel, source = 'DataChannel') => {
    if (!channel) return;

    dataChannelRef.current = channel;

    channel.onopen = () => addEvent(`${source} opened: ${channel.label}`);
    channel.onmessage = (event) => {
      recordMessage(event.data);
      handleDataChannelMessage(event.data);
    };
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
    modeSyncedRef.current = false;   // Week 4: send the mode again on this connection

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

      // Week 4 Day 1: tap Auralis's voice for the waveform.
      // The <audio> element still does the playing; the analyser is NOT
      // connected to the speakers, so the voice is never played twice.
      try {
        let ctx = audioCtxRef.current;
        if (!ctx || ctx.state === 'closed') {
          const AudioCtx = window.AudioContext || window.webkitAudioContext;
          ctx = new AudioCtx();
          audioCtxRef.current = ctx;
        }
        ctx.resume?.().catch(() => {});

        aiWaveRef.current?.source?.disconnect();
        const source = ctx.createMediaStreamSource(stream);
        const analyser = ctx.createAnalyser();
        analyser.fftSize = 1024;
        source.connect(analyser);
        aiWaveRef.current = { analyser, buffer: new Float32Array(analyser.fftSize), source };
      } catch (err) {
        console.warn('[WAVE] Could not tap AI audio for the waveform:', err);
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
    modeSyncedRef.current = false;
    // Week 4 Day 1: waveforms go back to a flat line
    aiWaveRef.current?.source?.disconnect();
    aiWaveRef.current = null;
    micWaveRef.current = null;

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
      audioRef.current.muted = false;
    }
    if (aiUnmuteTimerRef.current) clearTimeout(aiUnmuteTimerRef.current);
    aiUnmuteTimerRef.current = null;
    aiCutRef.current = false;
    interruptedTurnIdRef.current = null;
    setAiStopped(false);

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

  // ---------------------------------------------------------
  // Week 4: RECORD & REPLAY a real session
  // ---------------------------------------------------------

  const recordMessage = (raw) => {
    const rec = recordingRef.current;
    let data;
    try {
      data = typeof raw === 'string' ? JSON.parse(raw) : raw;
    } catch {
      return;
    }
    // ~50 frames/s: keep only frames that start/end a segment or update the counter
    if (data.type === 'audio_frame' && !data.segment_started && !data.segment_finished
        && (data.frame ?? 0) % 50 !== 0) return;
    if (!rec.startedAt) {
      rec.startedAt = Date.now();
      rec.mode = appModeRef.current;
      rec.events = [];
    }
    rec.events.push({ t: Date.now() - rec.startedAt, d: data });
    if (rec.events.length % 10 === 0 || data.type !== 'audio_frame') setRecordCount(rec.events.length);
  };

  const buildSession = (name) => ({
    name: name || `Session ${new Date().toLocaleString()}`,
    date: new Date(recordingRef.current.startedAt || Date.now()).toISOString(),
    mode: recordingRef.current.mode,
    snapshot: { frameCount, aiPackets, aiAudio },
    events: recordingRef.current.events,
  });

  const saveSession = () => {
    if (!recordingRef.current.events.length) return;
    const name = window.prompt('Name this session (e.g. "Review demo - training Meera")', '');
    if (name === null) return;
    const session = buildSession(name.trim());
    const next = [session, ...savedSessions.filter((x) => x.name !== session.name)].slice(0, 5);
    try {
      localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(next));
    } catch {
      addEvent('Session too big for browser storage: use Download instead');
    }
    setSavedSessions(next);
    setSelectedSession(session.name);
    addEvent(`Session saved: ${session.name} (${session.events.length} events)`);
  };

  const downloadSession = (session) => {
    if (!session) return;
    const blob = new Blob([JSON.stringify(session)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${session.name.replace(/[^a-z0-9-_ ]/gi, '').trim() || 'auralis-session'}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const loadSessionFile = async (file) => {
    if (!file) return;
    try {
      const session = JSON.parse(await file.text());
      if (!Array.isArray(session.events)) throw new Error('not an Auralis session');
      const next = [session, ...savedSessions.filter((x) => x.name !== session.name)].slice(0, 5);
      try {
        localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(next));
      } catch {
        /* keep it in memory only */
      }
      setSavedSessions(next);
      setSelectedSession(session.name);
      addEvent(`Session loaded from file: ${session.name}`);
    } catch (err) {
      addEvent(`Could not load session: ${err.message}`);
    }
  };

  const deleteSession = (name) => {
    const next = savedSessions.filter((x) => x.name !== name);
    try {
      localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(next));
    } catch {
      /* ignore */
    }
    setSavedSessions(next);
    setSelectedSession('');
  };

  const stopReplay = () => {
    replayTimersRef.current.forEach(clearTimeout);
    replayTimersRef.current = [];
    setReplay(null);
  };

  const startReplay = (session, speed = 'fast') => {
    if (!session) return;
    replayTimersRef.current.forEach(clearTimeout);
    replayTimersRef.current = [];
    // fresh screen in the mode the session was recorded in
    appModeRef.current = session.mode || 'assistant';
    setAppMode(session.mode || 'assistant');
    setTraining(null);
    setShowScore(false);
    clearHistory();
    modeSyncedRef.current = true; // replay must not talk to a backend
    setStatus('replay');
    const total = session.events.length;
    setReplay({ name: session.name, date: session.date, done: false, total, shown: 0 });

    // real time keeps the original gaps; fast squeezes long pauses
    let clock = 0;
    let prev = 0;
    session.events.forEach((ev, i) => {
      const gap = ev.t - prev;
      prev = ev.t;
      clock += speed === 'real' ? gap : Math.min(gap, 700) * 0.5;
      replayTimersRef.current.push(setTimeout(() => {
        handlerRef.current?.(ev.d);
        if (i % 5 === 0 || i === total - 1) setReplay((rp) => rp && { ...rp, shown: i + 1 });
      }, clock));
    });
    replayTimersRef.current.push(setTimeout(() => {
      const snap = session.snapshot || {};
      if (snap.frameCount) setFrameCount(snap.frameCount);
      if (snap.aiPackets) setAiPackets(snap.aiPackets);
      if (snap.aiAudio) setAiAudio(snap.aiAudio);
      setReplay((rp) => rp && { ...rp, done: true, shown: total });
      addEvent(`Replay finished: ${session.name}`);
    }, clock + 300));
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
    setInterruptions([]);
    setTurnMetrics([]);
    currentTurnIdRef.current = null;
  };

  // ---------------------------------------------------------
  // Week 3 Day 5: EMOTION TEST MODE
  // ---------------------------------------------------------

  const toggleTestMode = () => {
    const next = !testModeRef.current;
    testModeRef.current = next;
    setTestMode(next);
    addEvent(next ? 'Emotion test mode ON' : 'Emotion test mode OFF');
  };

  const chooseTestTarget = (label) => {
    testTargetRef.current = label;
    setTestTarget(label);
    if (!testModeRef.current) {
      testModeRef.current = true;
      setTestMode(true);
    }
  };

  const resetTests = () => setTestResults([]);

  const exportTests = () => {
    if (testResults.length === 0) return;
    const header = ['time', 'target', 'final', 'voice', 'words', 'source', 'correct', 'transcript'];
    const escape = (value) => `"${String(value ?? '').replace(/"/g, '""')}"`;
    const rows = testResults.map((r) =>
      [r.time, r.target, r.final, r.voice, r.words ?? '', r.source, r.final === r.target, r.transcript]
        .map(escape)
        .join(',')
    );
    const blob = new Blob([[header.join(','), ...rows].join('\n')], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `auralis_emotion_test_${new Date().toISOString().slice(0, 10)}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  };

  // ---------------------------------------------------------
  // Week 3 Day 6: LATENCY AUDIT (compare machines)
  // ---------------------------------------------------------

  const saveRun = () => {
    if (turnMetrics.length === 0) return;
    const now = new Date();
    const run = {
      id: now.toISOString(),
      label: runLabel.trim() || describeMachine(health),
      machine: describeMachine(health),
      savedAt: now.toLocaleString(),
      turns: turnMetrics.length,
      p50: p50Ttfa,
      p95: p95Ttfa,
      stages: Object.fromEntries(avgStages.map((stage) => [stage.key, stage.ms])),
    };
    const next = [...runs, run].slice(-AUDIT_MAX_RUNS);
    setRuns(next);
    storeRuns(next);
    setRunLabel('');
    addEvent(`Latency run saved: ${run.label} (${run.turns} turns, first audio p50 ${formatMs(run.p50)})`);
  };

  const deleteRun = (id) => {
    const next = runs.filter((run) => run.id !== id);
    setRuns(next);
    storeRuns(next);
  };

  const exportRuns = () => {
    if (runs.length === 0) return;
    const header = ['label', 'machine', 'saved_at', 'turns', 'stt_ms', 'llm_ms', 'phrase_ms',
      'voice_ms', 'first_audio_p50_ms', 'first_audio_p95_ms'];
    const escape = (value) => `"${String(value ?? '').replace(/"/g, '""')}"`;
    const rows = runs.map((r) =>
      [r.label, r.machine, r.savedAt, r.turns, r.stages?.stt, r.stages?.llm, r.stages?.piece,
        r.stages?.tts, r.p50, r.p95].map(escape).join(',')
    );
    const blob = new Blob([[header.join(','), ...rows].join('\n')], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `auralis_latency_runs_${new Date().toISOString().slice(0, 10)}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  };

  // ---------------------------------------------------------
  // VAD DISPLAY
  // ---------------------------------------------------------

  // Week 4 Day 2: Auralis was stopped while the backend still reports
  // ai_speaking (old backend finishing the cut reply silently)
  const aiStoppedNow = aiStopped && vadState === 'ai_speaking';
  const vadView = aiStoppedNow ? 'ai_stopped' : vadState;

  const getVadIcon = () =>
    aiStoppedNow
      ? '✋'
      : ({ speaking: '🗣️', listening: '🎧', processing: '⚙️', ai_speaking: '🔊' }[vadState] || '◉');

  const getVadLabel = () =>
    aiStoppedNow ? 'AURALIS STOPPED' : ({
      speaking: 'SPEAKING',
      listening: 'LISTENING',
      processing: 'PROCESSING',
      ai_speaking: 'AURALIS SPEAKING',
    }[vadState] || 'IDLE');

  // ---------------------------------------------------------
  // Week 4 Day 3: EXPERIENCE VIEW helpers
  // ---------------------------------------------------------

  const isLive = iceState === 'connected' || iceState === 'completed';
  const lastTurn = turns.length > 0 ? turns[turns.length - 1] : null;
  const isThinking =
    vadState === 'processing' ||
    (lastTurn && (lastTurn.aiState === 'waiting' || lastTurn.aiState === 'streaming') &&
      vadState !== 'ai_speaking' && vadState !== 'speaking');

  // Week 4: the person Auralis plays in training mode
  const caller = training?.scenario || findScenario(scenarioId);

  const xpPhase = !isLive
    ? isConnecting
      ? { key: 'connecting', title: 'Connecting…', sub: 'Setting up a private voice line' }
      : appMode === 'training'
        ? { key: 'ready', title: 'Negotiation training', sub: `Pick a scenario, start, and calm ${caller.name} down` }
        : { key: 'ready', title: 'Talk to Auralis', sub: 'Start a conversation and just speak' }
    : vadState === 'speaking'
      ? { key: 'listening', title: 'Listening', sub: 'Pause when you are done' }
      : aiStoppedNow
        ? { key: 'stopped', title: 'Stopped', sub: 'Go ahead, I am listening' }
        : vadState === 'ai_speaking'
          ? appMode === 'training'
            ? { key: 'speaking', title: `${caller.name} is speaking`, sub: `Talking over ${caller.pronoun} raises the agitation` }
            : { key: 'speaking', title: 'Auralis is speaking', sub: 'Talk over me any time to interrupt' }
          : isThinking
            ? { key: 'thinking', title: 'Thinking', sub: 'Understanding your words and your tone' }
            : appMode === 'training'
              ? { key: 'idle', title: 'Your turn', sub: `${caller.name} is waiting. Speak slowly and calmly.` }
              : { key: 'idle', title: 'I am listening', sub: 'Say something whenever you are ready' };

  const switchMode = (mode) => {
    if (mode === appModeRef.current) return;
    appModeRef.current = mode;
    setAppMode(mode);
    setShowScore(false);
    setTraining(null);
    clearHistory();
    const channel = dataChannelRef.current;
    if (channel && channel.readyState === 'open') {
      try {
        channel.send(JSON.stringify({ type: 'set_mode', mode, scenario: scenarioIdRef.current }));
        modeSyncedRef.current = true;
      } catch (err) {
        console.warn('[MODE] could not send mode:', err);
      }
    }
  };

  const restartTraining = (id = scenarioIdRef.current) => {
    scenarioIdRef.current = id;
    setScenarioId(id);
    setShowScore(false);
    setTraining(null);
    clearHistory();
    const channel = dataChannelRef.current;
    if (channel && channel.readyState === 'open') {
      channel.send(JSON.stringify({ type: 'set_mode', mode: 'training', scenario: id }));
    }
  };

  const pickRandomScenario = () => {
    const others = SCENARIO_LIST.filter((x) => x.id !== scenarioIdRef.current);
    restartTraining(others[Math.floor(Math.random() * others.length)].id);
  };

  // Week 4: score card numbers
  const trainingScore = training
    ? (() => {
        const steps = training.steps;
        const n = steps.length;
        const calm = steps.filter((st) => st.reasons.some((r) => r.startsWith('calm tone') || r.startsWith('soft tone'))).length;
        const talkedOver = steps.filter((st) => st.reasons.some((r) => r.startsWith('talked over'))).length;
        const escalating = steps.filter((st) => st.reasons.some((r) => r.startsWith('escalating'))).length;
        const empathy = steps.filter((st) => st.reasons.some((r) => r.startsWith('de-escalating'))).length;
        const end = training.agitation;
        const lowest = Math.min(...training.history);
        const success = end <= training.goal;
        const points = Math.max(0, Math.min(100, Math.round(
          (training.start - end) * 0.8 + (n ? (calm / n) * 15 : 0) + (n ? (empathy / n) * 10 : 0)
          - talkedOver * 5 - escalating * 8 + (success ? 10 : 0)
        )));
        return { n, calm, talkedOver, escalating, empathy, end, lowest, success, points };
      })()
    : null;

  // Week 4: emotion map points (last 10 turns)
  const youPoints = turns
    .map((t) => getEmotionView(t.emotion))
    .filter(Boolean)
    .map((ev) => EMOTION_VA[ev.sure ? ev.text : 'neutral'] || EMOTION_VA.neutral)
    .slice(-10);
  const callerPoints = training ? training.history.slice(-10).map(agitationToVA) : [];

  const startConversation = async () => {
    await startMic();
    await connectWebRTC();
  };

  const formatSeconds = (ms) =>
    ms === null || ms === undefined ? '—' : ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s`;

  const xpTurns = turns.slice(-12);

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
          <div className="header-right">
          {/* Week 4 Day 3: Experience / Engineer view switch */}
          <div className="view-toggle" role="tablist" aria-label="View">
            <button
              role="tab"
              aria-selected={view === 'experience'}
              className={view === 'experience' ? 'is-active' : ''}
              onClick={() => setView('experience')}
            >
              <Icon name="sparkle" size={14} /> Experience
            </button>
            <button
              role="tab"
              aria-selected={view === 'engineer'}
              className={view === 'engineer' ? 'is-active' : ''}
              onClick={() => setView('engineer')}
            >
              <Icon name="sliders" size={14} /> Engineer
            </button>
          </div>
          <div
            className={`system-status ${
              health ? '' : healthError ? 'is-offline' : 'is-connecting'
            }`}
            title={healthError ? `health check failed: ${healthError}` : HEALTH_URL}
          >
            <span className="status-dot"></span>
            {health
              ? `BACKEND ONLINE${healthMs !== null ? ` · ${healthMs} MS` : ''}`
              : healthError
                ? 'BACKEND OFFLINE'
                : 'CHECKING BACKEND…'}
          </div>
          </div>
        </header>

        {/* ================= SESSION RECORDER / REPLAY (Week 4) ================= */}
        <section className={`rec-bar ${replay ? 'is-replay' : ''}`}>
          {replay ? (
            <>
              <span className="rec-dot rec-dot-replay" />
              <strong>REPLAY</strong>
              <span className="rec-text">
                Recorded session “{replay.name}” · {new Date(replay.date).toLocaleString()} ·{' '}
                {replay.done ? 'finished' : `${replay.shown}/${replay.total} events`}
              </span>
              <button className="xp-link" onClick={() => startReplay(savedSessions.find((x) => x.name === replay.name), 'fast')}>Replay again</button>
              <button className="xp-link" onClick={() => { stopReplay(); clearHistory(); setTraining(null); setStatus('idle'); }}>Exit replay</button>
            </>
          ) : (
            <>
              <span className={`rec-dot ${isLive ? 'rec-dot-live' : ''}`} />
              <span className="rec-text">
                {isLive
                  ? `Recording this session · ${recordCount} events`
                  : recordCount
                    ? `Last session recorded · ${recordCount} events`
                    : 'Session recorder: live sessions are recorded so they can be saved and replayed'}
              </span>
              {recordCount > 0 && (
                <>
                  <button className="xp-link" onClick={saveSession}>Save</button>
                  <button className="xp-link" onClick={() => downloadSession(buildSession(''))}>Download</button>
                </>
              )}
              <span className="rec-spacer" />
              <select value={selectedSession} onChange={(e) => setSelectedSession(e.target.value)} aria-label="Saved sessions">
                <option value="">Saved sessions ({savedSessions.length})</option>
                {savedSessions.map((x) => (
                  <option key={x.name} value={x.name}>{x.name}</option>
                ))}
              </select>
              <button className="xp-link" disabled={!selectedSession || isLive}
                onClick={() => startReplay(savedSessions.find((x) => x.name === selectedSession), 'fast')}>Replay</button>
              <button className="xp-link" disabled={!selectedSession || isLive}
                onClick={() => startReplay(savedSessions.find((x) => x.name === selectedSession), 'real')}>Real time</button>
              {selectedSession && (
                <>
                  <button className="xp-link" onClick={() => downloadSession(savedSessions.find((x) => x.name === selectedSession))}>Download</button>
                  <button className="xp-link" onClick={() => deleteSession(selectedSession)}>Delete</button>
                </>
              )}
              <label className="xp-link rec-file">
                Load file
                <input type="file" accept="application/json,.json" onChange={(e) => { loadSessionFile(e.target.files?.[0]); e.target.value = ''; }} />
              </label>
            </>
          )}
        </section>

        {/* ================= DEMO MODE: backend offline (Week 4) ================= */}
        {!health && healthError && !isLive && !demoBannerClosed && !replay && (
          <section className="demo-banner" role="status">
            <div className="demo-banner-text">
              <p className="demo-kicker">Demo mode · live AI offline</p>
              <h2>The Auralis GPU backend is not running right now</h2>
              <p>
                This page is hosted on Vercel, but the AI (Faster-Whisper, WavLM-Large emotion,
                Llama 3.2 3B and Piper TTS) needs a GPU, so it runs on a team laptop behind a
                Cloudflare Tunnel. When that laptop is online this page connects automatically.
              </p>
              <ul className="demo-points">
                <li>Speech → text in ~130–195 ms (Faster-Whisper on GPU)</li>
                <li>Emotion from voice + words, fused (~78% on live turns)</li>
                <li>Interrupt Auralis mid-sentence (stops in ~200–450 ms)</li>
                <li>Crisis negotiation training with 12 scenarios and a score card</li>
              </ul>
            </div>
            <div className="demo-banner-actions">
              <button className="xp-btn xp-btn-primary" onClick={() => setShowReplay(true)}>
                Play a sample session
              </button>
              {DEMO_VIDEO_URL && (
                <a className="xp-btn xp-btn-primary" href={DEMO_VIDEO_URL} target="_blank" rel="noreferrer">
                  Watch the demo video
                </a>
              )}
              <a className="xp-btn xp-btn-ghost" href={REPO_URL} target="_blank" rel="noreferrer">
                View code on GitHub
              </a>
              <button className="xp-link" onClick={() => setDemoBannerClosed(true)}>
                Explore the interface anyway
              </button>
            </div>
          </section>
        )}

        {showReplay && <DemoReplay onClose={() => setShowReplay(false)} />}

        {/* ================= EXPERIENCE VIEW (Week 4 Day 3) ================= */}

        {view === 'experience' && (
          <div className={`xp phase-${xpPhase.key}`}>
            <section className="xp-stage">
              <div className="mode-toggle" role="tablist" aria-label="Mode">
                <button role="tab" aria-selected={appMode === 'assistant'}
                  className={appMode === 'assistant' ? 'is-active' : ''}
                  onClick={() => switchMode('assistant')}>
                  Assistant
                </button>
                <button role="tab" aria-selected={appMode === 'training'}
                  className={appMode === 'training' ? 'is-active' : ''}
                  onClick={() => switchMode('training')}>
                  Negotiation training
                </button>
              </div>
              <div className="xp-orb-wrap">
                <canvas ref={orbCanvasRef} className="xp-orb" />
              </div>

              <div className="xp-status" aria-live="polite">
                <h2 className="xp-title">
                  {xpPhase.title}
                  {(xpPhase.key === 'thinking' || xpPhase.key === 'connecting') && (
                    <span className="xp-dots"><span /><span /><span /></span>
                  )}
                </h2>
                <p className="xp-sub">{xpPhase.sub}</p>
              </div>

              <div className="xp-meta">
                <span
                  className="xp-pill"
                  style={{ '--pill-color': userWaveColor }}
                  title={lastEmotion?.source ? `decided by ${lastEmotion.source}` : 'voice tone + words'}
                >
                  <span className="xp-dot" />
                  {lastEmotionView ? `Feeling ${lastEmotionView.text}` : 'Emotion appears as you speak'}
                </span>
                <span className="xp-pill" style={{ '--pill-color': aiWaveColor }}>
                  <span className="xp-dot" />
                  {lastVoiceStyle} voice
                </span>
                {lastTtfa !== null && (
                  <span className="xp-pill xp-pill-quiet" title="end of your speech → Auralis's first sound">
                    replies in {formatSeconds(lastTtfa)}
                  </span>
                )}
              </div>

              <div className="xp-controls">
                {!isLive ? (
                  <button
                    className="xp-btn xp-btn-primary"
                    onClick={startConversation}
                    disabled={isConnecting || !health}
                  >
                    <Icon name="mic" />
                    {isConnecting ? 'Connecting…' : 'Start conversation'}
                  </button>
                ) : (
                  <>
                    <button
                      className="xp-btn xp-btn-ghost xp-btn-warn"
                      onClick={stopAuralis}
                      disabled={!aiCanBeStopped}
                    >
                      <Icon name="hand" /> Interrupt
                    </button>
                    <button className="xp-btn xp-btn-ghost" onClick={stopMic}>
                      <Icon name="stop" /> End
                    </button>
                  </>
                )}
              </div>

              {!health && !isLive && (
                <p className="xp-note">
                  {healthError
                    ? 'Live AI is offline (demo mode). The GPU backend runs on a team laptop; this page reconnects when it is online.'
                    : 'Checking the Auralis backend…'}
                </p>
              )}
              {error && <p className="xp-error">{error}</p>}
            </section>

            <section className="xp-chat-card">
              <div className="xp-chat-head">
                <span>Conversation</span>
                {turns.length > 0 && (
                  <button className="xp-link" onClick={clearHistory}>Clear</button>
                )}
              </div>

              <div className="xp-chat" ref={xpChatRef}>
                {xpTurns.length === 0 ? (
                  <div className="xp-empty">
                    <p>Your conversation will appear here.</p>
                    <p className="xp-empty-hint">
                      Try: “I had a really stressful day at work.”
                    </p>
                  </div>
                ) : (
                  xpTurns.map((turn) => {
                    const ev = getEmotionView(turn.emotion);
                    const color = ev ? WAVE_EMOTION_COLORS[ev.text] ?? WAVE_EMOTION_COLORS.neutral : null;
                    return (
                      <div className="xp-turn" key={turn.id}>
                        <div className="xp-msg xp-msg-user">
                          <p>{turn.userText}</p>
                          {ev && (
                            <span className="xp-tag" style={{ '--pill-color': color }}>
                              <span className="xp-dot" /> {ev.text}
                            </span>
                          )}
                        </div>

                        <div
                          className={`xp-msg xp-msg-ai ai-${turn.aiState}`}
                          title={[
                            turn.ttftMs != null && `first word ${turn.ttftMs} ms`,
                            turn.voice?.ttfaMs != null && `first audio ${turn.voice.ttfaMs} ms`,
                            turn.voice?.style && `${turn.voice.style} voice`,
                          ].filter(Boolean).join(' · ')}
                        >
                          <span className="xp-ai-mark" aria-hidden="true" />
                          <p>
                            {turn.aiState === 'waiting' && (
                              <span className="xp-dots"><span /><span /><span /></span>
                            )}
                            {turn.aiState === 'error' && <span className="xp-err">{turn.error}</span>}
                            {(turn.aiState === 'streaming' || turn.aiState === 'done' ||
                              turn.aiState === 'interrupted') && turn.aiText}
                            {turn.aiState === 'interrupted' && (
                              <span className="xp-cut"> — interrupted</span>
                            )}
                          </p>
                        </div>
                      </div>
                    );
                  })
                )}
              </div>
            </section>

            {/* Week 4 feature: training panel + emotion map */}
            <section className="xp-row2">
              {appMode === 'training' && (
                <div className="xp-panel training-panel">
                  <div className="xp-chat-head">
                    <span>Negotiation training</span>
                    {training && training.steps.length > 0 && (
                      <button className="xp-link" onClick={() => setShowScore(true)}>End session</button>
                    )}
                  </div>

                  <div className="scenario-picker">
                    <label htmlFor="scenario-select">Scenario</label>
                    <select id="scenario-select" value={scenarioId}
                      onChange={(e) => restartTraining(e.target.value)}>
                      {SCENARIO_LIST.map((x) => (
                        <option key={x.id} value={x.id}>
                          {x.name}: {x.title} ({x.level})
                        </option>
                      ))}
                    </select>
                    <button className="xp-link" onClick={pickRandomScenario}>Random</button>
                  </div>
                  <div className="scenario-meta">
                    <span className={`level-badge level-${caller.level.toLowerCase()}`}>{caller.level}</span>
                    <span className="xp-muted">starts at {caller.start} agitation · 12 scenarios</span>
                  </div>

                  <p className="training-brief">
                    {training?.scenario?.brief ??
                      `Auralis plays ${caller.name} (${caller.title.toLowerCase()}). Start the conversation and calm ${caller.pronoun} down.`}
                  </p>

                  <div className="agitation">
                    <div className="agitation-head">
                      <span>{caller.name}'s agitation</span>
                      <strong style={{ color: agitationColor(training?.agitation ?? caller.start) }}>
                        {training?.agitation ?? caller.start}
                      </strong>
                    </div>
                    <div className="agitation-track">
                      <div className="agitation-goal" style={{ left: `${training?.goal ?? 30}%` }} title="goal" />
                      <div className="agitation-fill" style={{
                        width: `${training?.agitation ?? caller.start}%`,
                        background: agitationColor(training?.agitation ?? caller.start),
                      }} />
                    </div>
                    <div className="agitation-scale"><span>calm</span><span>goal ≤ {training?.goal ?? 30}</span><span>panic</span></div>
                  </div>

                  {training && training.steps.length > 0 ? (
                    <ul className="training-steps">
                      {training.steps.slice(-4).reverse().map((st, i) => (
                        <li key={training.steps.length - i}>
                          <span className={st.delta > 0 ? 'delta-up' : 'delta-down'}>
                            {st.delta > 0 ? '+' : ''}{st.delta}
                          </span>
                          {st.reasons.join(' · ')}
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="training-tip">
                      Tips: introduce yourself, listen, name {caller.pronoun === 'her' ? 'her' : 'his'} feelings, speak slowly. Avoid "calm down", threats and talking over {caller.pronoun}.
                    </p>
                  )}
                </div>
              )}

              <div className="xp-panel va-panel">
                <div className="xp-chat-head">
                  <span>Emotion map</span>
                  <span className="xp-muted">arousal × valence</span>
                </div>
                <EmotionMap you={youPoints} caller={callerPoints} callerName={appMode === 'training' ? caller.name : 'Caller'} />
                <p className="xp-muted va-note">
                  {appMode === 'training'
                    ? `Blue: your tone each turn. Red: ${caller.name} moving from panic toward calm.`
                    : 'Your detected emotion each turn (last 10).'}
                </p>
              </div>
            </section>

            {/* Week 4 feature: score card */}
            {showScore && trainingScore && (
              <div className="score-overlay" role="dialog" aria-modal="true" aria-label="Training score">
                <div className="score-card">
                  <p className="score-kicker">Negotiation debrief · {caller.name}, {caller.title.toLowerCase()}</p>
                  <h2 className={trainingScore.success ? 'score-pass' : 'score-fail'}>
                    {trainingScore.success ? 'De-escalated' : 'Not de-escalated yet'}
                  </h2>
                  <div className="score-points">{trainingScore.points}<span>/100</span></div>
                  <div className="score-grid">
                    <div><strong>{training.start} → {trainingScore.end}</strong><span>agitation</span></div>
                    <div><strong>{trainingScore.n}</strong><span>turns</span></div>
                    <div><strong>{trainingScore.n ? Math.round((trainingScore.calm / trainingScore.n) * 100) : 0}%</strong><span>calm tone</span></div>
                    <div><strong>{trainingScore.empathy}</strong><span>empathy turns</span></div>
                    <div><strong>{trainingScore.escalating}</strong><span>escalating turns</span></div>
                    <div><strong>{trainingScore.talkedOver}</strong><span>talked over {caller.pronoun}</span></div>
                  </div>
                  <p className="score-tip">
                    {trainingScore.success
                      ? `Well done: calm tone and empathy brought ${caller.name} down. Next time try fewer turns or a harder scenario.`
                      : trainingScore.escalating > 0
                        ? `Commands like "calm down" or threats pushed ${caller.pronoun} up. Name the feelings instead.`
                        : 'Keep a calm tone, acknowledge his feelings and ask open questions.'}
                  </p>
                  <div className="score-actions">
                    <button className="xp-btn xp-btn-primary" onClick={() => restartTraining()}>Try again</button>
                    <button className="xp-btn xp-btn-ghost" onClick={pickRandomScenario}>New scenario</button>
                    <button className="xp-btn xp-btn-ghost" onClick={() => setShowScore(false)}>Close</button>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ================= ENGINEER VIEW (everything below) ================= */}

        {view === 'engineer' && (<>

        {/* SYSTEM CHIPS (Week 3 Day 6): what the backend is running */}
        {systemChips.length > 0 && (
          <div className="system-chips">
            {systemChips.map((chip) => (
              <span key={chip.key} className={`system-chip ${chip.ok ? '' : 'chip-off'}`}>
                <span className="chip-label">{chip.label}</span>
                <span className="chip-value">{chip.value}</span>
              </span>
            ))}
          </div>
        )}

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
                <div
                  className={`metric-value ${
                    aiStoppedNow ? 'latency-warn' : vadState === 'ai_speaking' ? 'latency-good' : ''
                  }`}
                >
                  {aiStoppedNow ? 'stopped' : vadState === 'ai_speaking' ? 'speaking' : 'quiet'}
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

            <div className={`vad-box vad-${vadView}`}>
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
                    : 'voice tone + words'}
                </div>
                {lastEmotion?.source && (
                  <div className="metric-sub emotion-mix-sub">
                    by <strong>{lastEmotion.source}</strong> · 🎙{' '}
                    {formatGuess(lastEmotion.voiceLabel, lastEmotion.voiceConfidence)} · 💬{' '}
                    {formatGuess(lastEmotion.textLabel, lastEmotion.textConfidence)}
                  </div>
                )}
              </div>
            </div>
          </section>

          {/* CONVERSATION (Week 2: transcript + streamed LLM reply) */}

          <section className="card full">
            <div className="card-title">
              <h2>Conversation</h2>
              <span>
                WHISPER + WAVLM + TEXT → LLAMA 3 → PIPER
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
                          {ev && turn.emotion.source && (
                            <span
                              className={`bubble-badge emotion-mix by-${turn.emotion.source}`}
                              title={`decided by ${turn.emotion.source}`}
                            >
                              <span className={turn.emotion.source === 'voice' ? 'mix-win' : ''}>
                                🎙 {formatGuess(turn.emotion.voiceLabel, turn.emotion.voiceConfidence)}
                              </span>
                              {' · '}
                              <span className={turn.emotion.source === 'words' ? 'mix-win' : ''}>
                                💬 {formatGuess(turn.emotion.textLabel, turn.emotion.textConfidence)}
                              </span>
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
                          {turn.interrupt && (
                            <span className="bubble-badge interrupt-badge">
                              ✋ {turn.interrupt.reason === 'manual' ? 'stopped' : 'interrupted'}
                              {turn.interrupt.stopMs !== null && turn.interrupt.stopMs !== undefined
                                ? ` · ${turn.interrupt.stopMs} ms`
                                : ''}
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
                          {turn.aiState === 'interrupted' && (
                            <>
                              {turn.aiText}
                              <span className="cut-mark"> — cut off</span>
                            </>
                          )}
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

          {/* LIVE WAVEFORM (Week 4 Day 1) */}

          <section className="card full">
            <div className="card-title">
              <h2>Live Waveform</h2>
              <span>
                YOUR MIC · AURALIS VOICE · SPEAK OVER AURALIS TO INTERRUPT
              </span>
            </div>

            <div className="wave-grid">
              <div
                ref={micWavePanelRef}
                className={`wave-panel wave-user ${vadState === 'speaking' ? 'wave-turn' : ''}`}
                style={{ '--wave-color': userWaveColor }}
              >
                <div className="wave-head">
                  <span className="wave-who">🎙 YOU</span>
                  <span className="wave-tag">
                    {lastEmotionView
                      ? `${lastEmotionView.emoji} ${lastEmotionView.text}`
                      : 'emotion —'}
                  </span>
                  <span ref={micWaveLabelRef} className="wave-level">mic off</span>
                </div>
                <canvas ref={micCanvasRef} className="wave-canvas" />
              </div>

              <div
                ref={aiWavePanelRef}
                className={`wave-panel wave-ai ${vadState === 'ai_speaking' ? 'wave-turn' : ''}`}
                style={{ '--wave-color': aiWaveColor }}
              >
                <div className="wave-head">
                  <span className="wave-who">🔊 AURALIS</span>
                  <span className="wave-tag">{lastVoiceStyle} voice</span>
                  <span ref={aiWaveLabelRef} className="wave-level">not connected</span>
                </div>
                <canvas ref={aiCanvasRef} className="wave-canvas" />
                <span className="wave-cut-label">✋ CUT</span>
              </div>
            </div>

            {/* Week 4 Day 2: interruption controls + stats */}
            <div className="interrupt-bar">
              <button
                className="btn btn-stop-ai"
                onClick={stopAuralis}
                disabled={!aiCanBeStopped}
                title="Stop Auralis mid-sentence (same as talking over it)"
              >
                ✋ STOP AURALIS
              </button>
              <div className="interrupt-stats">
                <span>
                  interruptions <strong>{interruptions.length}</strong>
                </span>
                <span>
                  by voice <strong>{bargeInCount}</strong> · by button <strong>{manualStopCount}</strong>
                </span>
                <span>
                  avg stop{' '}
                  <strong className={getLatencyClass(avgInterruptStop)}>
                    {formatMs(avgInterruptStop)}
                  </strong>
                </span>
              </div>
            </div>
          </section>

          {/* EMOTION TEST MODE (Week 3 Day 5) */}

          <section className="card full">
            <div className="card-title">
              <h2>Emotion Test Mode</h2>
              <span>
                PICK AN EMOTION → SAY IT → SEE WHO GOT IT RIGHT
                {testTotal > 0 && (
                  <>
                    <button className="btn-link" onClick={exportTests}>&nbsp;· EXPORT CSV</button>
                    <button className="btn-link" onClick={resetTests}>&nbsp;· RESET</button>
                  </>
                )}
              </span>
            </div>

            <div className="test-controls">
              <button
                className={`test-toggle ${testMode ? 'test-on' : ''}`}
                onClick={toggleTestMode}
              >
                {testMode ? '● TEST MODE ON' : '○ TEST MODE OFF'}
              </button>
              <div className="test-targets">
                {EMOTION_LABELS.map((label) => (
                  <button
                    key={label}
                    className={`test-target emotion-${EMOTION_GROUP[label]} ${
                      testTarget === label ? 'test-target-active' : ''
                    }`}
                    onClick={() => chooseTestTarget(label)}
                  >
                    {EMOTION_EMOJI[label]} {label}
                  </button>
                ))}
              </div>
            </div>
            <div className="test-hint">
              {testMode && testTarget
                ? `Now say something in a ${testTarget.toUpperCase()} way. Every turn is scored until you pick another emotion or turn test mode off.`
                : 'Pick the emotion you are going to act, then speak. Neutral words test the voice; emotional words test the text model.'}
            </div>

            <div className="audit-summary">
              <div className="metric">
                <div className="metric-label">Final (shown)</div>
                <div className="metric-value">{pctOf(testFinalHit, testTotal)}</div>
                <div className="metric-sub">
                  {testTotal > 0
                    ? `${testFinalHit}/${testTotal} · 3-group ${pctOf(testGroupHit, testTotal)}`
                    : 'what the app showed'}
                </div>
              </div>
              <div className="metric">
                <div className="metric-label">🎙 Voice only</div>
                <div className="metric-value">{pctOf(testVoiceHit, testTotal)}</div>
                <div className="metric-sub">
                  {testTotal > 0 ? `${testVoiceHit}/${testTotal} · WavLM tone` : 'WavLM tone'}
                </div>
              </div>
              <div className="metric">
                <div className="metric-label">💬 Words only</div>
                <div className="metric-value">{pctOf(testWordsHit, testWordsRows.length)}</div>
                <div className="metric-sub">
                  {testWordsRows.length > 0
                    ? `${testWordsHit}/${testWordsRows.length} · text model`
                    : 'needs the Day 5 backend'}
                </div>
              </div>
              <div className="metric">
                <div className="metric-label">Tests</div>
                <div className="metric-value">{testTotal}</div>
                <div className="metric-sub">
                  {testTotal > 0 ? `${testByWords} decided by words` : 'none yet'}
                </div>
              </div>
            </div>

            {testTotal > 0 && (
              <div className="test-grid">
                <div className="audit-table-wrap">
                  <table className="audit-table test-matrix">
                    <thead>
                      <tr>
                        <th>said ↓ / shown →</th>
                        {EMOTION_LABELS.map((label) => (
                          <th key={label} title={label}>{label.slice(0, 3)}</th>
                        ))}
                        <th>acc</th>
                      </tr>
                    </thead>
                    <tbody>
                      {EMOTION_LABELS.map((target, i) => {
                        const rowTotal = testMatrix[i].reduce((a, b) => a + b, 0);
                        return (
                          <tr key={target}>
                            <td>{EMOTION_EMOJI[target]} {target}</td>
                            {testMatrix[i].map((count, j) => (
                              <td
                                key={EMOTION_LABELS[j]}
                                className={
                                  count === 0 ? 'cell-zero' : i === j ? 'cell-hit' : 'cell-miss'
                                }
                              >
                                {count}
                              </td>
                            ))}
                            <td>{pctOf(testMatrix[i][i], rowTotal)}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                <div className="audit-table-wrap">
                  <table className="audit-table">
                    <thead>
                      <tr>
                        <th>Time</th>
                        <th>Said</th>
                        <th>Shown</th>
                        <th>Voice</th>
                        <th>Words</th>
                        <th>By</th>
                        <th>Transcript</th>
                      </tr>
                    </thead>
                    <tbody>
                      {testRecent.map((r) => (
                        <tr key={r.id}>
                          <td>{r.time}</td>
                          <td>{r.target}</td>
                          <td className={r.final === r.target ? 'audit-ok' : 'audit-miss'}>
                            {r.final} {r.final === r.target ? '✓' : '✗'}
                          </td>
                          <td className={r.voice === r.target ? 'audit-ok' : ''}>{r.voice}</td>
                          <td className={r.words === r.target ? 'audit-ok' : ''}>{r.words ?? '—'}</td>
                          <td>{r.source}</td>
                          <td>{r.transcript}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
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

          {/* LATENCY AUDIT: COMPARE MACHINES (Week 3 Day 6) */}

          <section className="card full">
            <div className="card-title">
              <h2>Latency Audit · Compare Machines</h2>
              <span>
                SAVE A RUN PER BACKEND (CPU / GPU) → COMPARE
                {runs.length > 0 && (
                  <button className="btn-link" onClick={exportRuns}>&nbsp;· EXPORT CSV</button>
                )}
              </span>
            </div>

            <div className="audit-save">
              <input
                className="audit-input"
                value={runLabel}
                onChange={(e) => setRunLabel(e.target.value)}
                placeholder={`run name, e.g. "Priya CPU" (default: ${describeMachine(health)})`}
                maxLength={40}
              />
              <button
                className="audit-save-btn"
                onClick={saveRun}
                disabled={turnMetrics.length === 0}
                title={turnMetrics.length === 0 ? 'say something first' : 'save this session as a run'}
              >
                SAVE THIS RUN ({turnMetrics.length} turns)
              </button>
            </div>
            <div className="test-hint">
              How to compare: connect to one backend, speak 5+ turns, save the run, press CLEAR in
              Conversation, switch VITE_BACKEND_URL to the other backend, repeat. Runs stay in this browser.
            </div>

            {runs.length === 0 ? (
              <div className="transcript-placeholder">No runs saved yet.</div>
            ) : (
              <>
                <div className="audit-table-wrap audit-runs">
                  <table className="audit-table">
                    <thead>
                      <tr>
                        <th>Run</th>
                        <th>Turns</th>
                        {LATENCY_STAGES.map((stage) => (
                          <th key={stage.key}>{stage.label}</th>
                        ))}
                        <th>First audio p50</th>
                        <th>p95</th>
                        <th>vs {LATENCY_TARGET_MS} ms</th>
                        <th></th>
                      </tr>
                    </thead>
                    <tbody>
                      {runs.map((run) => (
                        <tr key={run.id}>
                          <td title={`${run.machine} · ${run.savedAt}`}>
                            {run === fastestRun && runs.length > 1 ? '🏆 ' : ''}
                            {run.label}
                          </td>
                          <td>{run.turns}</td>
                          {LATENCY_STAGES.map((stage) => (
                            <td key={stage.key}>{formatMs(run.stages?.[stage.key])}</td>
                          ))}
                          <td className={getLatencyClass(run.p50)}>{formatMs(run.p50)}</td>
                          <td className={getLatencyClass(run.p95)}>{formatMs(run.p95)}</td>
                          <td className={getLatencyClass(run.p50)}>
                            {run.p50 !== null ? `${(run.p50 / LATENCY_TARGET_MS).toFixed(1)}×` : '—'}
                          </td>
                          <td>
                            <button className="btn-link" onClick={() => deleteRun(run.id)} title="delete run">
                              ✕
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {fastestRun && slowestRun && fastestRun !== slowestRun && fastestRun.p50 > 0 && (
                  <div className="audit-verdict">
                    <strong>{fastestRun.label}</strong> starts speaking{' '}
                    <strong>{(slowestRun.p50 / fastestRun.p50).toFixed(1)}× faster</strong> than{' '}
                    <strong>{slowestRun.label}</strong> (first audio p50 {formatMs(fastestRun.p50)} vs{' '}
                    {formatMs(slowestRun.p50)}).
                  </div>
                )}
              </>
            )}
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

        </>)}

        <div className="footer">AURALIS // REAL-TIME MULTI-MODEL VOICE SYSTEM</div>

        <audio ref={audioRef} autoPlay playsInline />
      </div>
    </div>
  );
}

export default App;