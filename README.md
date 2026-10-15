# Auralis — Backend

Real-time Voice-to-Voice Emotion Engine: **backend service**.

A Python `aiohttp` + `aiortc` server that receives microphone audio over WebRTC, detects end of speech with Silero VAD, transcribes with Faster-Whisper, and streams a reply from a local Llama model, sending every step to the browser live over a WebRTC DataChannel.

Part of Infotact Solutions' Advanced Generative AI Engineering internship (Project 2).

| Role | Owner | Branch |
|---|---|---|
| **Backend: WebRTC, VAD, STT, LLM streaming (this module)** | Priya Nirmal | `Backend` |
| Frontend: WebRTC client + live dashboard | Ankit Dash | `frontend` |
| ML: emotion recognition + LLM prompts | Siddhant | `ML` |

---

## Why Auralis

Most voice assistants chain Speech-to-Text → LLM → Text-to-Speech as separate request/response calls. That adds 3–5 s of delay and throws away the speaker's tone.

Auralis instead:
- Streams audio **continuously** over WebRTC (UDP)
- Detects the exact moment the user stops speaking
- Transcribes and replies as **background tasks**, so audio never stops flowing
- Streams the LLM reply **token by token** and measures **TTFT** live
- *(Week 3)* Adds vocal emotion and replies with an emotion-matched voice

**Use case:** a Crisis Negotiation Training Simulator, where a trainee speaks under pressure and the AI de-escalates.

---

## Architecture

```
Browser mic ──WebRTC (Opus/UDP)──► POST /webrtc/offer → aiortc RTCPeerConnection
                                            │
                              IncomingAudioTrack (20 ms frames)
                                            │
                       AudioProcessor  (fast, runs on every frame)
                         ├─ Silero VAD   48→16 kHz, 512-sample chunks
                         └─ Segmenter    pre-roll · hangover · min-voice filter
                                            │ segment finished
                                            ▼
                         Background "turn" task (never blocks the frame loop)
                         ├─ Faster-Whisper base.en  → transcript   (stt_ms)
                         └─ Llama 3.2 3B via Ollama → streamed reply (ttft_ms)
                                            │
                     JSON events over the DataChannel → frontend dashboard
                                            │
                     TTSAudioTrack (outgoing audio, Week 3: test tone today)
```

Signalling is one HTTP exchange (offer → answer). After that, audio and events flow over the peer connection.

---

## Features

| Feature | Details |
|---|---|
| WebRTC signalling | `aiortc`, SDP offer/answer via `POST /webrtc/offer` |
| Health check | `GET /health` → `{"status": "ok"}` |
| Voice Activity Detection | Silero VAD (ONNX), stateful 48→16 kHz resampler, threshold **0.5**, start after **3** speech chunks (~96 ms), end after **22** silence chunks (~700 ms) |
| Segmentation | **300 ms** pre-roll, **100 ms** hangover, **≥200 ms of real voice** required (coughs/taps never reach Whisper), forced split at **15 s** |
| Speech-to-Text | `faster-whisper` **base.en** (CPU, int8), `beam_size=5`, `initial_prompt` with project names, `condition_on_previous_text=False`, hallucination filter |
| LLM | **Llama 3.2 3B** via Ollama, async streaming, "Auralis" persona, remembers the last 3 exchanges |
| Non-blocking turns | Whisper + LLM run as background tasks; the frame loop keeps receiving audio |
| Model warm-up | Whisper and Llama are loaded at server start, before the first user connects |
| Latency metrics | `stt_ms`, `ttft_ms`, `total_ms`, all measured from end of speech |
| Outgoing audio | Chunked outgoing track (test tone today), groundwork for streamed TTS |
| No echo | Incoming mic audio is never sent back to the browser |
| CORS | Allowed origins: `localhost:5173` / `5174` |

> The LLM client (`app/llm/ollama_client.py`) is based on Siddhant's `ml/src/llm/ollama_client.py` and `emotion_prompt.py` (same persona and rules), made async so it doesn't block WebRTC.

---

## DataChannel Protocol

| `type` | Fields | When |
|---|---|---|
| `status` | `status: "connected"` | Once, when the channel opens |
| `audio_frame` | `frame, sample_rate, samples, pts, is_speech, segment_started, segment_finished, segment_frame_count, latency_ms` | ~50/s |
| `vad` | `vad_state: "speaking" \| "processing" \| "idle"` | On state change |
| `transcript` | `text, stt_ms` | Whisper finished |
| `llm_start` | — | LLM starts replying |
| `llm_token` | `text, first, ttft_ms` *(first token only)* | Each streamed token |
| `llm_done` | `text, ttft_ms, total_ms, tokens` | Reply complete |
| `llm_error` | `error` | Ollama unreachable / failed |
| `audio_error` | `frame, error` | Frame processing failed |

**One turn:** `vad: speaking` → `vad: processing` → `transcript` → `llm_start` → `llm_token` × N → `llm_done` → `vad: idle`

| Metric | Meaning |
|---|---|
| `stt_ms` | End of speech → transcript ready |
| `ttft_ms` | End of speech → first LLM word *(mid-project Latency Check)* |
| `total_ms` | End of speech → last LLM word |

---

## Week 2 Results (Mid-Project Review)

### Transcription Audit: problems found and fixed

| Problem | Root cause | Fix |
|---|---|---|
| Sentences split at short pauses | VAD ended speech after ~0.4 s silence, threshold 0.2 | Threshold **0.5**, end after **~700 ms** silence |
| Names misheard ("Ankida", "BTEK") | Whisper `base`, greedy decoding, no context | **`base.en`**, `beam_size=5`, `initial_prompt` |
| Phantom "Thank you." from noise | Coughs/taps sent to Whisper; min-length counted silence frames | Min-length now counts **real voice** only; hallucination filter |
| Silence counted twice (~1.2 s wait) | 500 ms hangover on top of the VAD window | Hangover reduced to **100 ms** |
| UI stuck on "PROCESSING" | No `idle` sent after transcription | `idle` sent when each turn finishes |
| Slow first sentence | Whisper loaded on first use | Warm-up at startup |
| WebRTC froze during Whisper | STT on the event loop | Background turn tasks |

Live audit result: **5 / 5 segments transcribed correctly**, including names.

### Latency Check (laptop CPU, no GPU)

| Stage | Measured |
|---|---|
| Speech → Text (`base.en`, `beam_size=5`) | ~1.2 s |
| TTFT with Llama 3 **8B** | 2.6 – 8.9 s (`ollama ps` showed **100 % CPU**) |
| TTFT with Llama 3.2 **3B** | re-measuring |

~0.7 s of every turn is the deliberate silence window (keeps sentences whole). The rest is Whisper and the LLM on CPU. The 800 ms target needs GPU inference; the streaming architecture is already built for it.

---

## Getting Started

**Prerequisites:** Python 3.12+, [Ollama](https://ollama.com)

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

`.env`:

```env
HOST=0.0.0.0
PORT=8001
ML_SERVICE_URL=http://localhost:9000
```

Run (Ollama must be running):

```bash
python -m app.main
```

Wait for:

```
[STT] Warm-up done in ... ms
[LLM] Warm-up done (model=llama3.2:3b)
[STARTUP] Ready
```

For a frontend on another network: `ngrok http 8001`, and share the URL as `VITE_BACKEND_URL`.

Optional: change the model with `OLLAMA_MODEL=llama3` (and `OLLAMA_URL` if Ollama isn't on `localhost:11434`).

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
| `[LLM] Warm-up failed` / `llm_error` | Ollama not running or model not pulled | `ollama pull llama3.2:3b`, then start Ollama |
| Very slow replies (TTFT > 5 s) | Large model on CPU | Check `ollama ps`; use `llama3.2:3b` |
| 403 / CORS error | Frontend not on `localhost:5173`/`5174` | Use an allowed port or add the origin in `app/main.py` |
| `ERR_CONNECTION_REFUSED` | ngrok stopped or URL changed | Restart `ngrok http 8001`, share the new URL |
| Sentences split | VAD settings changed | Check `detector.py`: threshold 0.5, 22 silence chunks |
| Junk text from noise | Min-voice filter changed | Check `MIN_VOICED_FRAMES` in `processor.py` |

---

## Progress Log

**Week 1: WebRTC Foundation ✅**
- aiohttp + aiortc server, `/webrtc/offer`, `/health`
- Bi-directional audio over WebRTC, DataChannel events
- PCM extraction, audio processing foundation

**Week 2: VAD + STT + LLM ✅**
- Real-time Silero VAD with tuned end-of-speech detection
- Speech segmentation (pre-roll, hangover, min-voice filter, 15 s cap)
- Faster-Whisper `base.en` with accuracy fixes and hallucination filter
- **LLM integration: Llama via Ollama, streamed token by token**
- **TTFT measurement** (`stt_ms`, `ttft_ms`, `total_ms`)
- Non-blocking background turns, model warm-up at startup
- Mid-project review: transcription audit + latency check

**Week 3: in progress 🔄**
- Day 1: outgoing audio track with chunked streaming (test tone)
- Next: connect Siddhant's emotion model, emotion-conditioned TTS, stream TTS audio

---

## Known Limitations

- Outgoing audio is a **test tone**; real TTS comes in Week 3
- Emotion model is copied into `app/ml_clients/` but **not yet connected** to the live pipeline
- CPU-only latency is above the 800 ms target (GPU needed)
- Free ngrok URLs change on every restart

---

## Roadmap

- **Week 3:** emotion detection in the live turn (parallel with Whisper), emotion-aware prompt, emotion-conditioned TTS streamed in chunks
- **Week 4:** interruption handling (stop AI audio when the user speaks), latency optimisation

---

## Tech Stack

| Layer | Tech |
|---|---|
| Web server | `aiohttp`, `aiohttp-cors` |
| Real-time transport | `aiortc` (WebRTC) |
| VAD | `silero-vad` (ONNX Runtime), `torch` |
| Speech-to-Text | `faster-whisper` (CTranslate2) |
| LLM | Ollama + Llama 3.2 3B |
| Audio resampling | `PyAV` (`av`) |
| Testing | `pytest`, `pytest-asyncio` |
| Config | `python-dotenv` |

Pinned versions: [`requirements.txt`](./requirements.txt)

---

## Project Structure

```
Backend/
├── app/
│   ├── main.py                     # aiohttp app, routes, CORS, model warm-up at startup
│   ├── config/settings.py          # HOST, PORT, ML_SERVICE_URL from .env
│   ├── webrtc/
│   │   ├── server.py               # /webrtc/offer, track + DataChannel wiring
│   │   ├── connection.py           # RTCPeerConnection wrapper
│   │   ├── audio_track.py          # frame loop + background STT → LLM turns, DataChannel events
│   │   └── tts_track.py            # outgoing audio track (Week 3)
│   ├── audio/
│   │   ├── processor.py            # VAD + segmentation (no STT here)
│   │   ├── vad/detector.py         # Silero VAD
│   │   ├── segments/speech_segment.py
│   │   ├── pcm/converter.py
│   │   └── metrics/latency.py
│   ├── stt/
│   │   ├── transcriber.py          # Faster-Whisper wrapper + hallucination filter
│   │   └── engine.py               # one shared Whisper model for the server
│   ├── llm/
│   │   └── ollama_client.py        # async streaming Llama client + persona
│   └── ml_clients/                 # Siddhant's emotion code (Week 3 integration)
├── tests/
├── requirements.txt
├── .env.example
└── pytest.ini
```
