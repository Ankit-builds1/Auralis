Auralis — Backend

Real-time Voice-to-Voice (V2V) Emotion Engine — backend service.

Python/aiohttp + aiortc service that receives streamed microphone audio over WebRTC, runs it through a VAD → speech-segmentation → speech-to-text pipeline, and pushes live results back to the browser over a WebRTC DataChannel.

Part of Infotact Solutions' Advanced Generative AI Engineering internship (Project 2 of 3).

Role	Owner	Branch
Backend / WebRTC + VAD + STT (this module)	Priya Nirmal	Backend
Frontend / streaming UI	Ankit Dash	frontend
Audio ML / emotion recognition	Siddhant	ML
Why Auralis

Most conversational AI chains Speech-to-Text → LLM → Text-to-Speech over slow request/response calls, adding 3–5 s of delay and discarding the speaker's emotional tone. Auralis streams audio continuously over WebRTC, detects when the user stops speaking, transcribes it in real time, and (in later weeks) responds with an emotion-matched voice — targeting sub-800 ms responses.

Use case: a Crisis Negotiation Training Simulator, where a trainee speaks under pressure and the AI de-escalates in a tone matched to their emotional state.

Architecture
Browser (mic)
    │  WebRTC SDP offer
    ▼
POST /webrtc/offer  ──►  aiortc RTCPeerConnection (SDP answer returned)
    │
    ▼
Incoming audio track (per 20 ms frame)
    │
    ▼
AudioProcessor
    ├─ VoiceActivityDetector  (Silero VAD, ONNX, 16 kHz, 512-sample chunks)
    ├─ SpeechSegment buffer   (pre-roll + silence hangover + min speech length)
    └─ SpeechToText           (faster-whisper, off the event loop via asyncio.to_thread)
    │
    ▼
JSON events over the WebRTC DataChannel (`vad-state`) — no polling
    │
    ▼
(Week 3, Day 1) Outgoing audio track — chunked streaming groundwork,
currently emitting a test tone ahead of real TTS output

Signalling is a single HTTP exchange (offer → answer); after that, audio and events flow entirely over the peer connection.

Features
Feature	What it does
WebRTC signalling	aiortc RTCPeerConnection, SDP offer/answer over POST /webrtc/offer
Health check	GET /health for uptime/monitoring
Voice Activity Detection	Silero VAD (ONNX), 48 kHz → 16 kHz mono resampling, fixed 512-sample chunks
Segmentation smoothing	500 ms silence hangover, 300 ms pre-roll buffer, minimum speech length to reject noise
Speech-to-Text	faster-whisper (base, CPU, int8), 48 kHz → 16 kHz resample via PyAV
Non-blocking pipeline	VAD + STT run in a worker thread (asyncio.to_thread) so WebRTC never freezes
Realtime feedback	Per-frame telemetry, VAD state, transcripts and errors pushed over the DataChannel
Latency tracking	Per-frame and per-segment latency_ms / processing_ms
No echo	Incoming mic audio is not sent back to the browser — avoids delayed echo
Outgoing audio (new)	Chunked outgoing audio track streaming a test tone — groundwork for streamed TTS
CORS	Locked to the frontend's Vite dev origins (localhost:5173 / 5174)
DataChannel protocol

The backend sends JSON messages over the vad-state DataChannel:

type	Fields	Frequency	Purpose
status	status: "connected"	Once, on channel open	Lets the frontend know the pipeline is live
audio_frame	frame, sample_rate, samples, pts, is_speech, segment_started, segment_finished, segment_frame_count, latency_ms	~50/s (one per 20 ms frame)	Per-frame telemetry
vad	vad_state: "speaking" | "processing"	Per utterance	Drives the frontend's VAD badge
transcript	text	Per finished segment	Appends to the live transcript
audio_error	frame, error	On failure	Surfaces backend errors to the UI

Per-utterance sequence: vad: speaking → vad: processing → transcript.

Week 2 results — pipeline fixes (mid-project review)

The transcription audit run from the frontend exposed real pipeline bugs, which were found and fixed here:

Issue found	Root cause	Fix
Segments ended on the first quiet frame, splitting/clipping sentences	Silence detected too eagerly, no buffer before speech onset	Added a 500 ms silence hangover and a 300 ms pre-roll buffer; added a minimum speech length to skip noise
WebRTC froze during transcription	faster-whisper ran directly on the asyncio event loop	Moved VAD + STT processing to a worker thread via asyncio.to_thread
Growing processing backlog	Per-frame debug logging (100+ lines/s)	Removed per-frame logging

Latency after fixes (steady, end-to-end ≈ 1.7 s):

Stage	Time
Silence wait (end-of-speech detection)	500 ms
faster-whisper (base, CPU, int8)	~1,100–1,250 ms
Total	~1,700 ms

Whisper accounts for ~70% of total latency. Paths toward the 800 ms target: a smaller model (tiny.en), GPU inference, or a shorter silence wait.

Getting Started

Prerequisites: Python 3.12+, pip.

bash
git clone https://github.com/Ankit-builds1/Auralis.git
cd Auralis
git checkout Backend
cd Backend

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt

Copy .env.example to .env:

HOST=0.0.0.0
PORT=8001
ML_SERVICE_URL=http://localhost:9000

Run:

bash
python -m app.main

To expose it to the frontend during dev, tunnel it (e.g. ngrok http 8001) and give the frontend team the public URL for their VITE_BACKEND_URL.

Testing
bash
pytest

Covers VAD, PCM conversion, latency tracking, speech segmentation, and the WebRTC offer flow (tests/).

Troubleshooting
Symptom	Cause	Fix
Frontend gets a 403 / CORS error	Frontend dev server not on localhost:5173/5174	Restart frontend on an allowed port, or add the origin in app/main.py's cors_config
Frontend shows frames stuck at 0	Backend not running, or offer request failing	Check GET /health, confirm backend logs show [WEBRTC] SDP answer created
ERR_CONNECTION_REFUSED on the frontend	ngrok not running, or URL rotated	Restart ngrok http 8001, share the new URL — free ngrok URLs change on every restart
Latency creeping above ~2 s	Debug logging reintroduced, or STT running back on the event loop	Confirm no per-frame prints, confirm asyncio.to_thread still wraps VAD/STT
Sentences still clipped/split	Silence hangover/pre-roll regressed	Check SpeechSegment config: 500 ms hangover, 300 ms pre-roll, min speech length filter
Progress log

Week 1 — WebRTC foundation ✅

Initialize Auralis backend
WebRTC signalling foundation (/webrtc/offer, SDP offer/answer)
Audio processing foundation
VAD pipeline foundation
Streaming latency groundwork
README + project timeline

Week 2 — Voice Activity Detection & Speech-to-Text ✅

PCM audio extraction
Real-time Silero VAD integration
Speech segment tracking (groups frames into utterances)
WebRTC + audio processing improvements
Speech-to-text pipeline (faster-whisper)
Full realtime voice pipeline completed
Fix: silence hangover, pre-roll, minimum speech length, VAD/STT off the event loop
Perf: removed per-frame debug prints causing processing backlog
README updated per week

Week 3 — Outgoing audio (in progress) 🔄

Day 1: outgoing audio track with chunked streaming and a test tone — groundwork for streaming TTS playback back to the browser
Known limitations
Outgoing audio track currently streams a test tone only — not yet wired to real TTS output.
TTFT (time-to-first-token) not measurable yet — the LLM/context engine integration is pending.
End-to-end latency (~1.7 s) is still above the 800 ms target, dominated by Whisper on CPU.
Whisper base occasionally mis-transcribes fast or unclear speech.
Free ngrok URLs rotate on every restart and must be re-shared with the frontend.
Roadmap
Week 3 (remaining): connect the outgoing audio track to real streamed TTS output; integrate the local LLM/context engine; implement interruption handling (halt outgoing audio the moment incoming speech is detected)
Week 4 (Refine & Polish): latency dashboard, Whisper optimization (smaller model and/or GPU inference), general cleanup
Tech Stack
Layer	Tech
Web server	aiohttp + aiohttp-cors
Real-time transport	aiortc (WebRTC, Python)
Voice Activity Detection	silero-vad (ONNX runtime)
Speech-to-Text	faster-whisper (CTranslate2 backend)
Audio resampling	PyAV (av) — 48 kHz → 16 kHz mono
Numerics	numpy, torch, torchaudio
Testing	pytest, pytest-asyncio
Config	python-dotenv

Full pinned versions are in requirements.txt.

Project Structure
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
│   │   ├── vad/detector.py      # Silero VAD (16 kHz, 512-sample chunks)
│   │   ├── segments/speech_segment.py  # pre-roll + hangover buffering into utterances
│   │   ├── pcm/converter.py     # PCM conversion helpers
│   │   └── metrics/latency.py   # per-frame latency tracking
│   └── stt/
│       └── transcriber.py       # faster-whisper wrapper, resample + transcribe
├── tests/                       # pytest suite (VAD, PCM, latency, segments, WebRTC, processor)
├── requirements.txt
├── .env.example
└── pytest.ini

(The Week 3 outgoing-audio-track module isn't reflected in the tree above yet — add its file path here once you share that code.)
