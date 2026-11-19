# Auralis ΓÇö Backend

Real-time Voice-to-Voice Emotion Engine: **backend service**.

A Python `aiohttp` + `aiortc` server that receives microphone audio over WebRTC, detects end of speech with Silero VAD, transcribes with Faster-Whisper, **detects the emotion from the voice and the words**, streams an emotion-aware reply from a local Llama model, and **speaks it back sentence by sentence** in a matching voice style. If the user talks over Auralis, it **stops at once and listens** (barge-in). Every step is sent to the browser live over a WebRTC DataChannel.

Part of Infotact Solutions' Advanced Generative AI Engineering internship (Project 2).

| Role | Owner | Branch |
|---|---|---|
| **Backend: WebRTC, VAD, STT, emotion, LLM + TTS streaming, barge-in (this module)** | Priya Nirmal (with Ankit Dash in Weeks 3ΓÇô4) | `Backend` |
| Frontend: WebRTC client, live dashboard, Experience view | Ankit Dash | `frontend` |
| ML: emotion recognition, fusion, TTS + barge-in research | Siddhant (Weeks 1ΓÇô2), Ankit Dash + Priya Nirmal (Weeks 3ΓÇô4) | `ML` |

---

## Why Auralis

Most voice assistants chain Speech-to-Text ΓåÆ LLM ΓåÆ Text-to-Speech as separate request/response calls. That adds 3ΓÇô5 s of delay and throws away the speaker's tone.

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
Browser mic ΓöÇΓöÇWebRTC (Opus/UDP)ΓöÇΓöÇΓû║ POST /webrtc/offer ΓåÆ aiortc RTCPeerConnection
                                            Γöé
                              IncomingAudioTrack (20 ms frames)
                                            Γöé
                       AudioProcessor  (fast, every frame)
                         Γö£ΓöÇ Silero VAD   48ΓåÆ16 kHz, 512-sample chunks
                         Γö£ΓöÇ Segmenter    pre-roll ┬╖ hangover ┬╖ min-voice filter
                         ΓööΓöÇ Barge-in     user talks over Auralis (loudness-gated) ΓåÆ interrupt  (Week 4)
                                            Γöé segment finished
                                            Γû╝
                Background "turn" task (never blocks the frame loop)
                  Γö£ΓöÇ Faster-Whisper base.en (GPU or CPU) ΓöÇΓöÉ in parallel
                  Γö£ΓöÇ WavLM-Large voice emotion ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÿ
                  Γö£ΓöÇ DistilRoBERTa text emotion ΓåÆ fusion (voice + words)
                  Γö£ΓöÇ Llama 3.2 3B via Ollama, emotion-aware prompt, streamed
                  ΓööΓöÇ Piper TTS, sentence by sentence, calm / soft / normal voice
                                            Γöé
        TTSAudioTrack (outgoing audio, real-time paced) ΓöÇΓöÇWebRTCΓöÇΓöÇΓû║ browser speaker
        JSON events over the DataChannel ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓû║ frontend dashboard
```

Signalling is one HTTP exchange (offer ΓåÆ answer). After that, audio both ways and all events flow over the peer connection.

---

## Features

| Feature | Details |
|---|---|
| WebRTC signalling | `aiortc`, SDP offer/answer via `POST /webrtc/offer` |
| Health check | `GET /health` ΓåÆ status + which models and devices run (STT device, GPU name, emotion model, TTS voice, LLM) |
| Voice Activity Detection | Silero VAD (ONNX), threshold **0.5**, start after **3** speech chunks (~96 ms), end after **22** silence chunks (~700 ms) |
| Segmentation | **300 ms** pre-roll, **100 ms** hangover, **ΓëÑ 200 ms of real voice** required, forced split at **15 s** |
| Speech-to-Text | `faster-whisper` **base.en**, **GPU auto-select** with a self-test and safe CPU fallback, `beam_size=5`, project-name `initial_prompt`, hallucination filter |
| Voice emotion | **WavLM-Large** (layer 11) + logistic regression adapted to our own voices (ML branch), runs **in parallel** with Whisper |
| Text emotion + fusion | `j-hartmann/emotion-english-distilroberta-base` (~30 ms CPU). Emotional words decide (0.7 text / 0.3 voice); clearly neutral words stay neutral unless the voice is very sure (0.4 / 0.6); otherwise the voice decides (0.2 / 0.8) |
| LLM | **Llama 3.2 3B** via Ollama, async streaming, "Auralis" persona, emotion injected as possibly-imperfect context, remembers the last 3 exchanges |
| Safety | Self-harm phrases override the emotion hint: Auralis responds with care, asks if the user is safe and gives the Tele-MANAS helpline (14416) |
| Text-to-Speech | **Piper** `en_US-lessac-medium`, spoken **sentence by sentence** (first piece may end at a comma), voice style from the emotion: upset ΓåÆ **calm**, low ΓåÆ **soft**, okay ΓåÆ **normal** |
| Echo guard | Speech that starts while Auralis is talking is first treated as its own echo |
| **Barge-in (Week 4)** | If that speech is really the user, Auralis stops: LLM + TTS cancelled, queued audio dropped, `interrupted` event sent, the new question answered. The frontend STOP button sends `{"type": "interrupt"}` |
| **Relative loudness gate (Week 4)** | Auralis learns the user's normal speaking level from their own speech; while it talks, only voiced frames at most **5 dB** below that level count, and **10** of them (200 ms) trigger barge-in. Before the level is known: 20 voiced frames (400 ms). Based on the ML branch finding that Silero VAD also calls speaker echo "speech" |
| Latency metrics | Per-turn breakdown (`stt_ms`, `ttft_ms`, `first_piece_ms`, `ttfa_ms`, ΓÇª) sent live and appended to `logs/latency.csv` |
| **Training mode (Week 4)** | Crisis negotiation trainer: Auralis plays a distressed person from one of **12 scenarios**; the trainee's tone, words and interruptions move an **agitation** score (0ΓÇô100) and Llama plays the character at that level (`app/training/negotiation.py`) |
| **Emotion fix (Week 4)** | Personal voice baseline learned from turns whose words are neutral, damping of voice-only "disgust", neutral-words weighting ΓåÆ live turns 4/9 ΓåÆ 7/9 correct |
| CORS | `localhost:5173` / `5174` always, plus the origins in `CORS_ORIGINS` (e.g. the Vercel URL) |
| TURN | STUN always; a TURN relay from `TURN_URL` / `TURN_USERNAME` / `TURN_CREDENTIAL` for users on other networks |

---

## DataChannel Protocol

**Backend ΓåÆ browser**

| `type` | Fields | When |
|---|---|---|
| `status` | `status: "connected"` | Channel opens |
| `audio_frame` | `frame, is_speech, segment_started, segment_finished, ΓÇª, echo` | ~50/s |
| `vad` | `vad_state: speaking \| processing \| ai_speaking \| idle` | On change |
| `transcript` | `text, stt_ms` | Whisper finished |
| `emotion` | `label, confidence, group, confident, source, voice_label, voice_confidence, text_label, text_confidence, ms, text_ms, ready_ms` | After the transcript (fused result; `source` = `voice` or `words`) |
| `llm_start` / `llm_token` / `llm_done` | `text, first, ttft_ms` / `text, ttft_ms, total_ms, tokens` | Reply streaming |
| `tts_start` | `ttfa_ms, style, group` | First audio of the reply |
| `tts_sentence` | `index, text, synth_ms, audio_ms` | Each spoken piece |
| `tts_end` | `sentences, audio_ms, ttfa_ms, style, underruns` | Playback finished |
| `turn_metrics` | `stt_ms, emotion_ms, ttft_ms, first_piece_ms, ttfa_ms, first_synth_ms, total_ms, audio_ms, pieces, voice` | One per turn |
| `echo_ignored` | ΓÇö | Short sound while Auralis spoke, ignored |
| **`interrupted`** | **`reason` (`barge_in` \| `manual`), `stop_ms`, `spoken_ms`, `dropped_ms`, `gate_db`, `user_level_db`** | **Auralis was stopped (Week 4)** |
| `llm_error` / `tts_error` / `audio_error` | `error` | Failures |

**Browser ΓåÆ backend**

| `type` | Effect |
|---|---|
| `interrupt` | STOP AURALIS button: same as a barge-in (`reason: "manual"`) |
| `set_mode` | `{"mode": "assistant" \| "training", "scenario": "<id>"}`: switch to the negotiation trainer (new conversation) |

**Training mode events (backend ΓåÆ browser):** `mode` (`mode, scenario, scenarios, agitation, goal`) when the mode changes, and `training_state` (`agitation, delta, reasons, turn, start, goal, scenario_id`) after every trainee turn.

**One turn:** `vad: speaking` ΓåÆ `processing` ΓåÆ `transcript` ΓåÆ `emotion` ΓåÆ `llm_start` ΓåÆ `llm_token` ├ù N ΓåÆ `tts_start` ΓåÆ `vad: ai_speaking` ΓåÆ `tts_sentence` ├ù N ΓåÆ `llm_done` ΓåÆ `tts_end` ΓåÆ `turn_metrics` ΓåÆ `vad: idle`

**With barge-in:** ΓÇª `vad: ai_speaking` ΓåÆ user talks ΓåÆ `interrupted` ΓåÆ `vad: speaking` ΓåÆ `tts_end` ΓåÆ new `transcript` ΓåÆ new reply

| Metric (from end of speech) | Meaning |
|---|---|
| `stt_ms` | ΓåÆ transcript ready |
| `ttft_ms` | ΓåÆ first LLM word |
| `first_piece_ms` | ΓåÆ first speakable piece of text |
| `ttfa_ms` | ΓåÆ first AI audio queued (**what the user feels**) |
| `total_ms` | ΓåÆ last LLM word |

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

Live audit: **5 / 5 segments** transcribed correctly. Speech ΓåÆ text ~1.2 s; TTFT with Llama 3 8B on CPU 2.6ΓÇô8.9 s ΓåÆ switched to **Llama 3.2 3B**.

### Week 3: emotion + voice

| Item | Result |
|---|---|
| TTS choice (ML benchmark, CPU) | Piper 31ΓÇô175 ms per sentence vs Kokoro 422ΓÇô1887 ms ΓåÆ **Piper** |
| Voice emotion model | WavLM-Large: **74.6 %** on unseen CREMA-D actors; adapted to our voices: 19 % ΓåÆ **53 %** |
| Live emotion (7 acted turns) | voice only **29 %**, words only 71 %, **fused 71 %** (3-group 86 %) |
| Whisper on GPU (RTX 2050) | speech ΓåÆ text **~2.6 s ΓåÆ ~0.55ΓÇô0.65 s** |

**Latency breakdown, Ankit's laptop (RTX 2050), backend on localhost:**

| Stage (average) | Whisper on CPU (10 turns) | Whisper on GPU |
|---|---|---|
| Speech ΓåÆ text | 2,557 ms | ~570 ms |
| Llama thinking | 1,705 ms | ~1.2 s |
| First phrase written | 644 ms | similar |
| Voice synthesis | 1,117 ms | similar |
| **First audio (TTFA)** | **p50 4,781 ms**, p95 13,326 ms | **~2.7 s** |

The 800 ms target is not reached yet: ~0.7 s is the deliberate end-of-speech window, and Llama + Piper share one 4 GB laptop GPU / CPU. On a CPU-only laptop the first audio arrives after ~7ΓÇô8 s.

### Week 4: interruption handling (barge-in)

| Item | Result |
|---|---|
| Live barge-in (7 interruptions, frame rule) | Auralis stops **~380 ms** after the user starts talking and answers the new question; STOP button: **0 ms** |
| Relative loudness gate (simulation: user ΓêÆ30 dBFS, echo ΓêÆ50 dBFS) | Old rule: the echo stopped Auralis **twice**; with the gate the echo is ignored and the real user stops Auralis in **~190 ms** |
| Self-interruptions from echo | **0** in the live test (browser echo cancellation on) |
| ML evaluation (`ml/src/bargein/eval_barge_in.py`, ML branch) | Silero VAD also calls speaker echo "speech", so frame counting alone cannot separate echo from the user; a **loudness gate** + 10 frames gave 98 % detection and 0 % false triggers in simulation ΓåÆ next step: a gate relative to the user's own speech level |
| Emotion fix | Plain questions ("Tell me about AI") no longer show up as "sad": clearly neutral words need a very sure voice to override them |
| Emotion fix 2 (personal baseline + disgust damping) | Live turns correct: **4/9 ΓåÆ 7/9** (~78 %) |
| Whisper benchmark on our voices (ML branch) | `base.en` 59.9 % word accuracy / 129 ms; **`distil-small.en` 66.3 % / 195 ms** (now live); `small.en` 68.4 % / 270 ms |
| Training mode (live, "Locked-in office worker") | Agitation **85 ΓåÆ 9** in 5 turns, score **86/100**, De-escalated |

### Week 4: crisis negotiation training mode

| Part | How it works |
|---|---|
| Roles | Auralis = the distressed caller, the user = trainee negotiator |
| Agitation (0ΓÇô100) | Starts at the scenario's level (70ΓÇô90). Per turn: calm tone ΓêÆ8, soft tone ΓêÆ4, upset tone +12; de-escalating words (empathy, listening, safety, options) up to ΓêÆ18; escalating words ("calm down", threats, rushing) up to +24; talking over the caller +8; very long turn +4. One turn can lower it by at most 20 |
| Goal | Agitation Γëñ 30 = "De-escalated" |
| LLM | Llama gets the scenario persona + the current agitation level (very agitated ΓåÆ calm and cooperative) and stays in character |
| Voice | Tense (normal) voice while agitation ΓëÑ 50, calmer voice below |
| Safety | No weapons or self-harm in any scenario; crisis phrases from the trainee still trigger the normal safety reply |

**12 scenarios:** locked-in office worker (Hard), cancelled flight (Medium), failed exam (Easy), noisy neighbour (Easy), hospital wait (Medium), rejected loan (Medium), fake delivery complaint (Easy), unpaid factory salary (Hard), child late from school (Hard), road accident argument (Medium), online fraud victim (Medium), sudden eviction (Medium).

### Week 4: deployment (hybrid)

```
Browser ΓöÇΓöÇΓû║ Vercel (frontend)
   Γö£ΓöÇ signalling (HTTPS) ΓöÇΓöÇΓû║ Cloudflare Tunnel ΓöÇΓöÇΓû║ this backend :8001 (GPU laptop)
   ΓööΓöÇ audio (WebRTC) ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓû║ Metered TURN relay ΓöÇΓöÇΓû║ this backend
```

The backend needs a GPU (Whisper, WavLM-Large, Llama 3.2 3B and Piper together), so it runs on a team laptop:

1. In `.env`: `CORS_ORIGINS=https://<app>.vercel.app` and `TURN_URL`, `TURN_USERNAME`, `TURN_CREDENTIAL` (never commit them).
2. `python -m app.main` ΓåÆ check the startup line `[STARTUP] CORS origins: ΓÇª`.
3. `cloudflared tunnel --url http://localhost:8001` ΓåÆ put the `https://ΓÇª.trycloudflare.com` URL into the frontend's `VITE_BACKEND_URL` on Vercel and redeploy.

Live demo: https://auralis-hazel.vercel.app (works while the backend laptop is online).

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

**Where does the time go?** After a few turns:

```bash
python tools/latency_report.py --last 20
```

prints median / p95 for every stage (Whisper, Llama, first phrase, Piper) from `logs/latency.csv`, so a setting change (`VAD_SILENCE_MS`, `WHISPER_BEAM_SIZE`, ΓÇª) can be measured.

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
| `ollama list` ΓåÆ "timed out waiting for server to start" | Ollama looking for a moved models folder | Stop `ollama*` processes, set `OLLAMA_MODELS`, run `ollama serve` |
| `/health` shows STT on `cpu` on a GPU laptop | CUDA libraries missing | Install `nvidia-cublas-cu12` + `nvidia-cudnn-cu12==9.*` in the venv, restart |
| `[TTS] Disabled: voice file not found` | Piper voice missing | Copy the two `en_US-lessac-medium` files into `models/tts/` |
| Emotion always neutral | `EMOTION_MODEL` / joblib missing, or confidence gate | Check `.env` and `models/`; see `EMOTION_MIN_CONFIDENCE` |
| Auralis stops by itself while speaking | Speaker echo counted as the user | Use headphones, or raise `BARGE_IN_MIN_VOICED_FRAMES` to 30 |
| Auralis does not stop when interrupted | Speech too short / too quiet, or `BARGE_IN=0` | Talk a bit longer; check `.env` |
| 403 / CORS error | Frontend origin not allowed | Use `localhost:5173`/`5174`, or add the site to `CORS_ORIGINS` in `.env` |
| `ERR_CONNECTION_REFUSED` via ngrok | ngrok stopped or pointing to another app | Restart `ngrok http 8001` |

---

## Progress Log

**Week 1: WebRTC Foundation Γ£à**
aiohttp + aiortc server, `/webrtc/offer`, `/health`, bi-directional audio, DataChannel events, PCM extraction.

**Week 2: VAD + STT + LLM Γ£à**
Silero VAD with tuned end-of-speech, segmentation, Faster-Whisper `base.en` with accuracy fixes, Llama streamed token by token, TTFT measurement, background turns, warm-up, mid-project review.

**Week 3: emotion + voice Γ£à**

| Day | Work |
|---|---|
| 1 | Outgoing audio track with chunked, real-time paced streaming |
| 2 | WavLM-Large voice emotion in the live turn (parallel with Whisper), `emotion` event, emotion-aware prompt |
| 3 | Piper TTS spoken sentence by sentence, calm / soft / normal voice styles, `ai_speaking` state, echo guard |
| 4 | Per-turn latency breakdown (`turn_metrics`, `logs/latency.csv`), first spoken piece at a comma |
| 5 | Text emotion + voice/words fusion, safety rule for self-harm phrases |
| 6 | Whisper GPU auto-select with self-test + CPU fallback, detailed `/health` |
| 7 | Documentation |

**Week 4: interruption handling Γ£à**
- Full-duplex **barge-in**: talk over Auralis ΓåÆ LLM + TTS cancelled, queued audio dropped, new question answered (~380 ms)
- STOP AURALIS button (`interrupt` message from the browser), `interrupted` event
- Cut-off replies kept in the conversation history as "[interrupted by the user]"
- Fusion fix: plain neutral questions stay neutral
- **Relative loudness gate** for barge-in (from the ML W4D1 evaluation): echo no longer stops Auralis, real interruptions are detected in ~200 ms
- **Latency sprint**: shorter first sentence in the prompt (Auralis starts speaking sooner), `VAD_SILENCE_MS` setting, `tools/latency_report.py`
- Fixed `requirements.txt` (transformers, scikit-learn, joblib, piper-tts, librosa, scipy, soundfile), full `.env.example`, safer `.gitignore`
- **Emotion fix 2**: personal voice baseline, disgust damping, request-first prompt
- **Crisis negotiation training mode** (`set_mode`, `mode`, `training_state`) with **12 scenarios**
- **Deployment**: `CORS_ORIGINS` and TURN settings from `.env`, Cloudflare Tunnel + Vercel

---

## Known Limitations

- **Latency above the 800 ms target:** ~2.7 s to the first audio with GPU Whisper on a 4 GB laptop GPU, ~7ΓÇô8 s on CPU only
- **Voice-only emotion is weak on real speakers** (~50 %); the words carry most of the signal, so fusion is essential
- **Whisper** still mishears some terms with our accent; `distil-small.en` reaches ~66 % word accuracy on our recordings
- **Barge-in**: the loudness gate needs ~1 s of the user's speech to learn their level; a user who interrupts much more quietly than they normally speak may not stop Auralis
- Piper changes speed / pitch / volume only; XTTS-v2 carried more emotion in tests but needs a GPU and is not deployed
- **Live demo depends on the laptop:** a quick Cloudflare tunnel gets a new URL on every restart, and the page works only while the backend laptop is online

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
Γö£ΓöÇΓöÇ app/
Γöé   Γö£ΓöÇΓöÇ main.py                     # aiohttp app, routes, CORS, warm-up, /health
Γöé   Γö£ΓöÇΓöÇ config/settings.py          # HOST, PORT from .env
Γöé   Γö£ΓöÇΓöÇ webrtc/
Γöé   Γöé   Γö£ΓöÇΓöÇ server.py               # /webrtc/offer, tracks, DataChannel (incl. interrupt messages)
Γöé   Γöé   Γö£ΓöÇΓöÇ connection.py           # RTCPeerConnection wrapper
Γöé   Γöé   Γö£ΓöÇΓöÇ audio_track.py          # frame loop, turns (STT + emotion ΓåÆ LLM ΓåÆ TTS), barge-in
Γöé   Γöé   ΓööΓöÇΓöÇ tts_track.py            # outgoing audio track, real-time pacing, clear() for barge-in
Γöé   Γö£ΓöÇΓöÇ audio/
Γöé   Γöé   Γö£ΓöÇΓöÇ processor.py            # VAD + segmentation (+ voiced flag and frame level for barge-in)
Γöé   Γöé   Γö£ΓöÇΓöÇ vad/detector.py         # Silero VAD
Γöé   Γöé   Γö£ΓöÇΓöÇ segments/ pcm/ metrics/
Γöé   Γö£ΓöÇΓöÇ stt/
Γöé   Γöé   Γö£ΓöÇΓöÇ transcriber.py          # Faster-Whisper, GPU/CPU auto-select, hallucination filter
Γöé   Γöé   ΓööΓöÇΓöÇ engine.py               # one shared Whisper model
Γöé   Γö£ΓöÇΓöÇ llm/ollama_client.py        # async streaming Llama client, persona, safety rule, training prompt
Γöé   Γö£ΓöÇΓöÇ training/negotiation.py     # 12 scenarios, agitation rules, caller persona prompt
Γöé   Γö£ΓöÇΓöÇ tts/engine.py               # Piper voice, voice styles, sentence splitter
Γöé   ΓööΓöÇΓöÇ ml_clients/
Γöé       Γö£ΓöÇΓöÇ emotion_engine.py       # WavLM-Large voice emotion
Γöé       ΓööΓöÇΓöÇ text_emotion.py         # text emotion + fusion
Γö£ΓöÇΓöÇ models/                         # .joblib + Piper voice (NOT in git)
Γö£ΓöÇΓöÇ logs/latency.csv                # per-turn latency (NOT in git)
Γö£ΓöÇΓöÇ tools/latency_report.py         # median / p95 per stage from logs/latency.csv
Γö£ΓöÇΓöÇ tests/
Γö£ΓöÇΓöÇ requirements.txt
Γö£ΓöÇΓöÇ .env.example
ΓööΓöÇΓöÇ pytest.ini
```
