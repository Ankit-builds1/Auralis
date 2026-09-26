# Auralis — Backend

Real-time Voice-to-Voice Emotion Engine — **backend service**.

This is the backend component of **Auralis**, Project 2 of 3 in the Infotact Solutions "Advanced Generative AI Engineering" internship (team of 3). Auralis is a low-latency, real-time conversational audio system built for a crisis-negotiation training simulator use case: the browser streams microphone audio over WebRTC, and the backend processes it through a Voice Activity Detection (VAD) → speech segmentation → Speech-to-Text (STT) pipeline.

> 🎧 Frontend (Vite + React, mic capture + WebRTC handshake) lives in a separate repo/folder.

## Status

- ✅ **Week 1** — complete
- 🔄 **Week 2** — in progress (~2 days in)

## Architecture

```
Browser (mic)
    │  WebRTC offer (SDP)
    ▼
POST /webrtc/offer  ──►  aiortc RTCPeerConnection
    │
    ▼
Incoming audio track (per WebRTC frame)
    │
    ▼
AudioProcessor
    ├─ VoiceActivityDetector (Silero VAD, ONNX)
    ├─ SpeechSegment buffer (start/accumulate/finish on silence)
    └─ SpeechToText (faster-whisper) — runs once a segment finishes
    │
    ▼
Events pushed back to frontend over the WebRTC DataChannel
(status, per-frame audio/VAD metadata, vad state, errors)
```

Key design point: the backend **does not** send the microphone audio back to the browser (avoids the delayed echo / voice-repetition problem). All feedback to the frontend goes out as JSON events on the DataChannel.

## Tech Stack

| Layer | Tech |
|---|---|
| Web server | `aiohttp` + `aiohttp-cors` |
| Real-time transport | `aiortc` (WebRTC, Python) |
| Voice Activity Detection | `silero-vad` (ONNX runtime) |
| Speech-to-Text | `faster-whisper` (CTranslate2 backend) |
| Audio resampling | `PyAV` (`av`) — 48kHz → 16kHz mono |
| Numerics | `numpy`, `torch`, `torchaudio` |
| Testing | `pytest`, `pytest-asyncio` |
| Config | `python-dotenv` |

Full pinned versions are in [`requirements.txt`](./requirements.txt).

## Project Structure

```
Backend/
├── app/
│   ├── main.py                  # aiohttp app + routes + CORS
│   ├── config/
│   │   └── settings.py          # HOST, PORT, ML_SERVICE_URL (from .env)
│   ├── webrtc/
│   │   ├── server.py            # /webrtc/offer handler, SDP offer/answer, track + datachannel wiring
│   │   ├── connection.py        # RTCPeerConnection wrapper
│   │   └── audio_track.py       # per-frame processing loop, sends events over DataChannel
│   ├── audio/
│   │   ├── processor.py         # VAD → segment → STT pipeline orchestration
│   │   ├── vad/detector.py      # Silero VAD (16kHz, 512-sample chunks)
│   │   ├── segments/speech_segment.py  # buffers frames for one utterance
│   │   ├── pcm/converter.py     # PCM conversion helpers
│   │   └── metrics/latency.py   # per-frame latency tracking
│   └── stt/
│       └── transcriber.py       # faster-whisper wrapper, 48kHz→16kHz resample + transcribe
├── tests/                       # pytest suite (VAD, PCM, latency, segments, WebRTC, processor)
├── requirements.txt
├── .env.example
└── pytest.ini
```

## API

| Method | Route | Description |
|---|---|---|
| `GET` | `/health` | Health check — `{"status": "ok", "service": "auralis-backend"}` |
| `POST` | `/webrtc/offer` | Accepts a WebRTC SDP offer `{sdp, type}`, returns an SDP answer. Attaches the incoming mic track to the audio pipeline. |

**DataChannel events** (server → browser, JSON):
- `{"type": "status", "status": "connected"}`
- `{"type": "audio_frame", "frame", "sample_rate", "samples", "pts", "is_speech", "segment_started", "segment_finished", "segment_frame_count", "latency_ms"}`
- `{"type": "vad", "vad_state": "speaking" | "processing"}`
- `{"type": "audio_error", "frame", "error"}`

CORS is currently allowed for `http://localhost:5173` and `http://localhost:5174` (the Vite dev server).

## Setup

**Requirements:** Python 3.12+ (tested with a 3.12/3.14 venv), pip.

```bash
# from the Backend/ folder
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### Environment Variables

Copy `.env.example` to `.env` and adjust as needed:

```
HOST=0.0.0.0
PORT=8001
ML_SERVICE_URL=http://localhost:9000
```

`.env` is git-ignored — never commit it.

## Running

```bash
python -m app.main
```

Server starts on `HOST:PORT` (default `0.0.0.0:8001`). To expose it to the frontend during dev, tunnel it (e.g. `ngrok http 8001`) and point the frontend's `VITE_BACKEND_URL` at the public URL.

## Testing

```bash
pytest
```

Covers VAD, PCM conversion, latency tracking, speech segmentation, and the WebRTC offer flow (`tests/`).

## Roadmap

- [x] **Week 1** — WebRTC signaling (`/webrtc/offer`), audio track ingestion, health endpoint
- [ ] **Week 2** *(in progress)* — VAD-driven segment state streamed to frontend, STT integration hardening
- [ ] **Week 3** — LLM response generation + emotion-aware context, interruption handling
- [ ] **Week 4** — Streaming TTS output, latency dashboard, polish

## Notes

- STT currently runs the Whisper `base` model on CPU with `int8` compute for low resource usage — swap `model_size`/`device` in `app/audio/processor.py` if a GPU becomes available.
- Silero VAD uses a fixed 512-sample chunk size at 16kHz; incoming WebRTC frames are buffered until a full chunk is available.
