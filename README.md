# Auralis — Backend

Real-time Voice-to-Voice Emotion Engine: **backend service**.

A Python `aiohttp` + `aiortc` server that receives microphone audio over WebRTC, detects end of speech with Silero VAD, transcribes with Faster-Whisper, **detects the emotion from the voice and the words**, streams an emotion-aware reply from a local Llama model, and **speaks it back sentence by sentence** in a matching voice style. If the user talks over Auralis, it **stops at once and listens** (barge-in). Every step is sent to the browser live over a WebRTC DataChannel.

Part of Infotact Solutions' Advanced Generative AI Engineering internship (Project 2).

| Role | Owner | Branch |
|---|---|---|
| **Backend: WebRTC, VAD, STT, emotion, LLM + TTS streaming, barge-in (this module)** | Priya Nirmal (with Ankit Dash in Weeks 3–4) | `Backend` |
| Frontend: WebRTC client, live dashboard, Experience view | Ankit Dash | `frontend` |
| ML: emotion recognition, fusion, TTS + barge-in research | Siddhant (Weeks 1–2), Ankit Dash + Priya Nirmal (Weeks 3–4) | `ML` |

---

## Why Auralis

Most voice assistants chain Speech-to-Text → LLM → Text-to-Speech as separate request/response calls. That adds 3–5 s of delay and throws away the speaker's tone.

Auralis instead:
- Streams audio **continuously** over WebRTC (UDP)
- Detects the exact moment the user stops speaking
- Hears **how** something was said (WavLM voice emotion) and **what** was said (text emotion), and fuses both
- Streams the LLM reply token by token and **speaks the first phrase while the rest is still being written**
- Lets the user **interrupt** Auralis mid-sentence (full duplex)

**Use case:** a Crisis Negotiation Training Simulator, where a trainee speaks under pressure and the AI de-escalates in a calm, matched tone.

---

## Architecture (end of Week 4)

```
Browser mic ──WebRTC (Opus/UDP)──► POST /webrtc/offer → aiortc RTCPeerConnection
                                            │
                              IncomingAudioTrack (20 ms frames)
                                            │
                       AudioProcessor  (fast, every frame)
                         ├─ Silero VAD   48→16 kHz, 512-sample chunks
                         ├─ Segmenter    pre-roll · hangover · min-voice filter
                         └─ Barge-in     user talks over Auralis ≥ 400 ms → interrupt  (Week 4)
                                            │ segment finished
                                            ▼
                Background "turn" task (never blocks the frame loop)
                  ├─ Faster-Whisper base.en (GPU or CPU) ─┐ in parallel
                  ├─ WavLM-Large voice emotion ───────────┘
                  ├─ DistilRoBERTa text emotion → fusion (voice + words)
                  ├─ Llama 3.2 3B via Ollama, emotion-aware prompt, streamed
                  └─ Piper TTS, sentence by sentence, calm / soft / normal voice
                                            │
        TTSAudioTrack (outgoing audio, real-time paced) ──WebRTC──► browser speaker
        JSON events over the DataChannel ─────────────────────────► frontend dashboard
```

Signalling is one HTTP exchange (offer → answer). After that, audio both ways and all events flow over the peer connection.

---

## Features

| Feature | Details |
|---|---|
| WebRTC signalling | `aiortc`, SDP offer/answer via `POST /webrtc/offer` |
| Health check | `GET /health` → status + which models and devices run (STT device, GPU name, emotion model, TTS voice, LLM) |
| Voice Activity Detection | Silero VAD (ONNX), threshold **0.5**, start after **3** speech chunks (~96 ms), end after **22** silence chunks (~700 ms) |
| Segmentation | **300 ms** pre-roll, **100 ms** hangover, **≥ 200 ms of real voice** required, forced split at **15 s** |
| Speech-to-Text | `faster-whisper` **base.en**, **GPU auto-select** with a self-test and safe CPU fallback, `beam_size=5`, project-name `initial_prompt`, hallucination filter |
| Voice emotion | **WavLM-Large** (layer 11) + logistic regression adapted to our own voices (ML branch), runs **in parallel** with Whisper |
| Text emotion + fusion | `j-hartmann/emotion-english-distilroberta-base` (~30 ms CPU). Emotional words decide (0.7 text / 0.3 voice); clearly neutral words stay neutral unless the voice is very sure (0.4 / 0.6); otherwise the voice decides (0.2 / 0.8) |
| LLM | **Llama 3.2 3B** via Ollama, async streaming, "Auralis" persona, emotion injected as possibly-imperfect context, remembers the last 3 exchanges |
| Safety | Self-harm phrases override the emotion hint: Auralis responds with care, asks if the user is safe and gives the Tele-MANAS helpline (14416) |
| Text-to-Speech | **Piper** `en_US-lessac-medium`, spoken **sentence by sentence** (first piece may end at a comma), voice style from the emotion: upset → **calm**, low → **soft**, okay → **normal** |
| Echo guard | Speech that starts while Auralis is talking is first treated as its own echo |
| **Barge-in (Week 4)** | If that speech has **≥ 20 real-voice frames (400 ms)**, it is the user: LLM + TTS cancelled, queued audio dropped, `interrupted` event sent, the new question answered. The frontend STOP button sends `{"type": "interrupt"}` |
| Latency metrics | Per-turn breakdown (`stt_ms`, `ttft_ms`, `first_piece_ms`, `ttfa_ms`, …) sent live and appended to `logs/latency.csv` |
| CORS | Allowed origins: `localhost:5173` / `5174` |

---

## DataChannel Protocol

**Backend → browser**

| `type` | Fields | When |
|---|---|---|
| `status` | `status: "connected"` | Channel opens |
| `audio_frame` | `frame, is_speech, segment_started, segment_finished, …, echo` | ~50/s |
| `vad` | `vad_state: speaking \| processing \| ai_speaking \| idle` | On change |
| `transcript` | `text, stt_ms` | Whisper finished |
| `emotion` | `label, confidence, group, confident, source, voice_label, voice_confidence, text_label, text_confidence, ms, text_ms, ready_ms` | After the transcript (fused result; `source` = `voice` or `words`) |
| `llm_start` / `llm_token` / `llm_done` | `text, first, ttft_ms` / `text, ttft_ms, total_ms, tokens` | Reply streaming |
| `tts_start` | `ttfa_ms, style, group` | First audio of the reply |
| `tts_sentence` | `index, text, synth_ms, audio_ms` | Each spoken piece |
| `tts_end` | `sentences, audio_ms, ttfa_ms, style, underruns` | Playback finished |
| `turn_metrics` | `stt_ms, emotion_ms, ttft_ms, first_piece_ms, ttfa_ms, first_synth_ms, total_ms, audio_ms, pieces, voice` | One per turn |
| `echo_ignored` | — | Short sound while Auralis spoke, ignored |
| **`interrupted`** | **`reason` (`barge_in` \| `manual`), `stop_ms`, `spoken_ms`, `dropped_ms`** | **Auralis was stopped (Week 4)** |
| `llm_error` / `tts_error` / `audio_error` | `error` | Failures |

**Browser → backend**

| `type` | Effect |
|---|---|
| `interrupt` | STOP AURALIS button: same as a barge-in (`reason: "manual"`) |

**One turn:** `vad: speaking` → `processing` → `transcript` → `emotion` → `llm_start` → `llm_token` × N → `tts_start` → `vad: ai_speaking` → `tts_sentence` × N → `llm_done` → `tts_end` → `turn_metrics` → `vad: idle`

**With barge-in:** … `vad: ai_speaking` → user talks → `interrupted` → `vad: speaking` → `tts_end` → new `transcript` → new reply

| Metric (from end of speech) | Meaning |
|---|---|
| `stt_ms` | → transcript ready |
| `ttft_ms` | → first LLM word |
| `first_piece_ms` | → first speakable piece of text |
| `ttfa_ms` | → first AI audio queued (**what the user feels**) |
| `total_ms` | → last LLM word |

---

## Results

### Week 2: Mid-Project Review (laptop CPU)

| Problem found by the transcription audit | Fix |
|---|---|
| Sentences split at short pauses | VAD threshold **0.5**, end after **~700 ms** silence |
| Names misheard ("Ankida", "BTEK") | **`base.en`**, `beam_size=5`, `initial_prompt` |
| Phantom "Thank you." from noise | Min-length counts **real voice** only + hallucination filter |
| Silence counted twice (~1.2 s wait) | Hangover **100 ms** |
| UI stuck on "PROCESSING" / WebRTC froze during Whisper | `idle` after each turn; background turn tasks |

Live audit: **5 / 5 segments** transcribed correctly. Speech → text ~1.2 s; TTFT with Llama 3 8B on CPU 2.6–8.9 s → switched to **Llama 3.2 3B**.

### Week 3: emotion + voice

| Item | Result |
|---|---|
| TTS choice (ML benchmark, CPU) | Piper 31–175 ms per sentence vs Kokoro 422–1887 ms → **Piper** |
| Voice emotion model | WavLM-Large: **74.6 %** on unseen CREMA-D actors; adapted to our voices: 19 % → **53 %** |
| Live emotion (7 acted turns) | voice only **29 %**, words only 71 %, **fused 71 %** (3-group 86 %) |
| Whisper on GPU (RTX 2050) | speech → text **~2.6 s → ~0.55–0.65 s** |

**Latency breakdown, Ankit's laptop (RTX 2050), backend on localhost:**

| Stage (average) | Whisper on CPU (10 turns) | Whisper on GPU |
|---|---|---|
| Speech → text | 2,557 ms | ~570 ms |
| Llama thinking | 1,705 ms | ~1.2 s |
| First phrase written | 644 ms | similar |
| Voice synthesis | 1,117 ms | similar |
| **First audio (TTFA)** | **p50 4,781 ms**, p95 13,326 ms | **~2.7 s** |

The 800 ms target is not reached yet: ~0.7 s is the deliberate end-of-speech window, and Llama + Piper share one 4 GB laptop GPU / CPU. On a CPU-only laptop the first audio arrives after ~7–8 s.

### Week 4: interruption handling (barge-in)

| Item | Result |
|---|---|
| Live barge-in (7 interruptions) | Auralis stops **~380 ms** after the user starts talking and answers the new question; STOP button: **0 ms** |
| Self-interruptions from echo | **0** in the live test (browser echo cancellation on) |
| ML evaluation (`ml/src/bargein/eval_barge_in.py`, ML branch) | Silero VAD also calls speaker echo "speech", so frame counting alone cannot separate echo from the user; a **loudness gate** + 10 frames gave 98 % detection and 0 % false triggers in simulation → next step: a gate relative to the user's own speech level |
| Emotion fix | Plain questions ("Tell me about AI") no longer show up as "sad": clearly neutral words need a very sure voice to override them |

---

## Getting Started

**Prerequisites:** Python 3.12, [Ollama](https://ollama.com)

```bash
ollama pull llama3.2:3b
```

```bash
git clone https://github.com/Ankit-builds1/Auralis.git
cd Auralis
git checkout Backend
cd Backend

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
copy .env.example .env          # macOS/Linux: cp .env.example .env
```

**Model files (not in git, ask the team / ML branch):**

| File | Used for |
|---|---|
| `models/ssl_wavlm-large-adapted_emotion.joblib` | voice emotion classifier |
| `models/tts/en_US-lessac-medium.onnx` + `.onnx.json` | Piper voice |

WavLM-Large (~1.2 GB), the text-emotion model and Whisper download automatically on first start. Set `HF_HOME` to a big drive first:

```powershell
$env:HF_HOME = "D:\path\to\hf-cache"
```

**Optional GPU Whisper (NVIDIA, Windows):**

```powershell
python -m pip install --no-cache-dir nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
```

Run (Ollama must be running):

```bash
python -m app.main
```

Wait for `[STARTUP] Ready`. Check `http://localhost:8001/health`.

For a frontend on another network: `ngrok http 8001` and use that URL (no trailing `/`) as `VITE_BACKEND_URL`.

All settings are documented in [`.env.example`](./Backend/.env.example) (emotion, Whisper device, TTS, `BARGE_IN`, `BARGE_IN_MIN_VOICED_FRAMES`, `TEXT_NEUTRAL_STRONG`).

---

## Testing

```bash
pytest
```

Covers VAD, PCM conversion, latency tracking, speech segmentation, the audio processor and the WebRTC offer flow.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Cannot reach Ollama at localhost:11434` | Ollama not running | Start Ollama (`ollama serve`); if models were moved, set `OLLAMA_MODELS` first |
| `ollama list` → "timed out waiting for server to start" | Ollama looking for a moved models folder | Stop `ollama*` processes, set `OLLAMA_MODELS`, run `ollama serve` |
| `/health` shows STT on `cpu` on a GPU laptop | CUDA libraries missing | Install `nvidia-cublas-cu12` + `nvidia-cudnn-cu12==9.*` in the venv, restart |
| `[TTS] Disabled: voice file not found` | Piper voice missing | Copy the two `en_US-lessac-medium` files into `models/tts/` |
| Emotion always neutral | `EMOTION_MODEL` / joblib missing, or confidence gate | Check `.env` and `models/`; see `EMOTION_MIN_CONFIDENCE` |
| Auralis stops by itself while speaking | Speaker echo counted as the user | Use headphones, or raise `BARGE_IN_MIN_VOICED_FRAMES` to 30 |
| Auralis does not stop when interrupted | Speech too short / too quiet, or `BARGE_IN=0` | Talk a bit longer; check `.env` |
| 403 / CORS error | Frontend not on `localhost:5173`/`5174` | Use an allowed port |
| `ERR_CONNECTION_REFUSED` via ngrok | ngrok stopped or pointing to another app | Restart `ngrok http 8001` |

---

## Progress Log

**Week 1: WebRTC Foundation ✅**
aiohttp + aiortc server, `/webrtc/offer`, `/health`, bi-directional audio, DataChannel events, PCM extraction.

**Week 2: VAD + STT + LLM ✅**
Silero VAD with tuned end-of-speech, segmentation, Faster-Whisper `base.en` with accuracy fixes, Llama streamed token by token, TTFT measurement, background turns, warm-up, mid-project review.

**Week 3: emotion + voice ✅**

| Day | Work |
|---|---|
| 1 | Outgoing audio track with chunked, real-time paced streaming |
| 2 | WavLM-Large voice emotion in the live turn (parallel with Whisper), `emotion` event, emotion-aware prompt |
| 3 | Piper TTS spoken sentence by sentence, calm / soft / normal voice styles, `ai_speaking` state, echo guard |
| 4 | Per-turn latency breakdown (`turn_metrics`, `logs/latency.csv`), first spoken piece at a comma |
| 5 | Text emotion + voice/words fusion, safety rule for self-harm phrases |
| 6 | Whisper GPU auto-select with self-test + CPU fallback, detailed `/health` |
| 7 | Documentation |

**Week 4: interruption handling ✅**
- Full-duplex **barge-in**: talk over Auralis → LLM + TTS cancelled, queued audio dropped, new question answered (~380 ms)
- STOP AURALIS button (`interrupt` message from the browser), `interrupted` event
- Cut-off replies kept in the conversation history as "[interrupted by the user]"
- Fusion fix: plain neutral questions stay neutral
- Fixed `requirements.txt` (transformers, scikit-learn, joblib, piper-tts, librosa, scipy, soundfile), full `.env.example`, safer `.gitignore`

---

## Known Limitations

- **Latency above the 800 ms target:** ~2.7 s to the first audio with GPU Whisper on a 4 GB laptop GPU, ~7–8 s on CPU only
- **Voice-only emotion is weak on real speakers** (~50 %); the words carry most of the signal, so fusion is essential
- **Whisper `base.en`** mishears some terms with our accent (e.g. "AI" → "A.A.")
- **Barge-in relies on browser echo cancellation**; without headphones, loud speakers can make Auralis stop on its own echo (a relative loudness gate is the next step)
- Piper changes speed / pitch / volume only; XTTS-v2 carried more emotion in tests but needs a GPU and is not deployed
- Free ngrok URLs change on every restart

---

## Tech Stack

| Layer | Tech |
|---|---|
| Web server | `aiohttp`, `aiohttp-cors` |
| Real-time transport | `aiortc` (WebRTC), `PyAV` |
| VAD | `silero-vad` (ONNX Runtime) |
| Speech-to-Text | `faster-whisper` (CTranslate2), optional CUDA |
| Emotion | WavLM-Large (`transformers`, `torch`) + `scikit-learn`; DistilRoBERTa text emotion |
| LLM | Ollama + Llama 3.2 3B |
| Text-to-Speech | `piper-tts` (+ `librosa` pitch shift) |
| Testing / config | `pytest`, `pytest-asyncio`, `python-dotenv` |

Pinned versions: [`requirements.txt`](./Backend/requirements.txt)

---

## Project Structure

```
Backend/
├── app/
│   ├── main.py                     # aiohttp app, routes, CORS, warm-up, /health
│   ├── config/settings.py          # HOST, PORT from .env
│   ├── webrtc/
│   │   ├── server.py               # /webrtc/offer, tracks, DataChannel (incl. interrupt messages)
│   │   ├── connection.py           # RTCPeerConnection wrapper
│   │   ├── audio_track.py          # frame loop, turns (STT + emotion → LLM → TTS), barge-in
│   │   └── tts_track.py            # outgoing audio track, real-time pacing, clear() for barge-in
│   ├── audio/
│   │   ├── processor.py            # VAD + segmentation (+ voiced-frame count for barge-in)
│   │   ├── vad/detector.py         # Silero VAD
│   │   ├── segments/ pcm/ metrics/
│   ├── stt/
│   │   ├── transcriber.py          # Faster-Whisper, GPU/CPU auto-select, hallucination filter
│   │   └── engine.py               # one shared Whisper model
│   ├── llm/ollama_client.py        # async streaming Llama client, persona, safety rule
│   ├── tts/engine.py               # Piper voice, voice styles, sentence splitter
│   └── ml_clients/
│       ├── emotion_engine.py       # WavLM-Large voice emotion
│       └── text_emotion.py         # text emotion + fusion
├── models/                         # .joblib + Piper voice (NOT in git)
├── logs/latency.csv                # per-turn latency (NOT in git)
├── tests/
├── requirements.txt
├── .env.example
└── pytest.ini
```
