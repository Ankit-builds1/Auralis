# Auralis: Frontend + Backend (WebRTC Foundation)

Real-time Voice-to-Voice (V2V) Emotion Engine, Week 1 module (React + WebRTC frontend, aiortc backend).

Part of Infotact Solutions' Advanced Generative AI Engineering internship (Project 2 of 3). Team of 3, role-swapped from OmniSight. Project roles: **frontend/streaming UI**, backend/streaming (Siddhant), audio ML (Priya). For Week 1, the WebRTC backend was also built as part of this module.

## What Auralis does

Most conversational AI pipelines (STT → LLM → TTS) take 3-5 seconds and strip out emotional tone entirely. Auralis fixes both: it streams audio in real time over WebRTC, detects the speaker's emotional state (via Wav2Vec2 on the backend), and generates an emotion-matched voice response with sub-800ms latency, closer to a real human conversation than a typical voice bot.

**Use case:** a Crisis Negotiation Training Simulator, where a trainee speaks in a panicked voice and the AI responds with a de-escalation tone that adapts to their emotional intensity.

## Responsibilities

**Frontend (browser-side real-time audio pipeline)**
- Capturing raw microphone audio via `getUserMedia`
- Establishing a low-latency WebRTC peer connection to the backend (bypassing slow HTTP request/response cycles)
- Streaming mic audio to the backend and playing back the AI's response audio as it arrives, in chunks
- (Week 4) A cinematic, emotion-reactive UI: live waveform, live transcript, emotion meter, and latency stats

**Backend (WebRTC server)**
- Exposes a signaling endpoint (`POST /offer`) that accepts the browser's SDP offer and returns an answer
- Receives the incoming mic audio track over the peer connection
- Streams audio back to the browser over the same connection
- Runs on port 8001, exposed publicly with ngrok for frontend testing

## Progress: Week 1

**Day 1: Complete**
- Scaffolded the React app with Vite (`npm create vite@latest frontend -- --template react`), ESLint configured
- Implemented microphone capture using `navigator.mediaDevices.getUserMedia({ audio: true })`
- Verified raw `MediaStreamTrack` capture end-to-end: Start/Stop Mic controls, status display, and console-logged audio track confirming the browser has live mic access
- Commit: `Day 1: React scaffold + mic capture working` (branch: `frontend`)

**Day 2: Complete**
- Set up `RTCPeerConnection` and attached the captured mic track
- Generated the SDP offer (after ICE gathering completes) and sent it to the backend's `POST /offer` endpoint
- Built the backend signaling endpoint (aiortc) that returns the SDP answer; frontend applies it with `setRemoteDescription`
- Backend served on port 8001 and exposed via ngrok; frontend reads its URL from `VITE_BACKEND_URL`
- Commit: `Day 2: ...` (add your commit message)

**Day 3: Complete**
- Confirmed audio flows from browser to backend over the peer connection
- Connection state changes logged on the frontend; incoming audio track confirmed on the backend
- Commit: `Day 3: ...` (add your commit message)

**Day 4: Complete**
- Handled the return path: the backend streams audio back and the frontend plays it in real time through an `<audio>` element via `ontrack`, without waiting for a full response
- Bi-directional audio stream working end-to-end
- Commit: `Day 4: ...` (add your commit message)

**Day 5: Up next**
- Error handling (mic permission denied, connection drops, backend unreachable)
- Cleanup on Stop (close the peer connection, stop tracks)
- Final Week 1 README update

## Week 1 goal (per official plan)

> **WebRTC Foundation:** Build the frontend React app and backend aiortc server to establish a bi-directional audio stream, bypassing slow HTTP protocols.

**Status:** Core goal achieved (Days 1-4). Only hardening and documentation remain (Day 5).

## Tech stack

- **Frontend:** React + Vite, native WebRTC APIs (`RTCPeerConnection`, `getUserMedia`), ESLint
- **Backend:** Python, aiortc, exposed via ngrok for testing

## Running locally

```bash
# Backend (port 8001)
# start your aiortc server, then expose it:
ngrok http 8001

# Frontend
cd frontend
# .env -> VITE_BACKEND_URL=<your ngrok URL>
npm install
npm run dev   # http://localhost:5173
```

## Looking ahead

- **Week 2:** Reflect Voice Activity Detection (VAD) state from the backend, showing "listening / user speaking / AI speaking" in the UI
- **Week 3:** Wire in real transcript + emotion data from the backend/ML pipeline; build interruption-handling UI (AI stops talking when the user speaks)
- **Week 4 (Refine & Polish):** Cinematic waveform visualizer with emotion-driven color shifts, live transcript, emotion meter, and latency dashboard, the final polished frontend experience
