# Auralis — Real-Time Voice-to-Voice Emotion Engine

**Week 1: WebRTC Foundation — Complete ✅**

Part of Infotact Solutions' Advanced Generative AI Engineering internship (Project 2 of 3). Team of 3, role-swapped from OmniSight. Roles: **frontend/streaming UI + Week 1 backend (this module)**, backend/streaming (Siddhant), audio ML (Priya).

## Why Auralis

Standard conversational AI pipelines (STT → LLM → TTS) run 3-5 seconds behind real speech and discard emotional tone entirely. Auralis closes both gaps: it streams audio over WebRTC instead of HTTP, detects the speaker's emotional state in real time via Wav2Vec2, and generates an emotion-matched voice response at sub-800ms latency — fast and expressive enough to feel like talking to a person, not a bot.

**Flagship use case:** a Crisis Negotiation Training Simulator — a trainee speaks in a panicked voice, and the AI responds with a de-escalation tone that adapts live to their emotional intensity.

## Week 1 Result

A fully working, bi-directional, low-latency audio stream between browser and server — the real-time backbone the entire emotion pipeline depends on. **Goal met, ahead of hardening work most teams leave for later.**

```
🎙️  Browser mic  ──WebRTC──▶  aiortc server  ──WebRTC──▶  🔊 Browser playback
                     (< 1 HTTP round trip, streamed both ways)
```

## What was built

**Frontend — browser-side real-time audio pipeline**
- Live microphone capture via `getUserMedia`
- `RTCPeerConnection` setup, SDP offer/answer exchange, ICE handling
- Real-time playback of the AI's streamed response audio as it arrives — no waiting for a full response
- Connection-state monitoring and graceful error handling (mic denied, dropped connection, unreachable backend)
- Full mic/connection lifecycle cleanup on Stop

**Backend — WebRTC signaling & media server**
- `POST /offer` signaling endpoint (aiortc) accepting the browser's SDP offer and returning an answer
- Live ingestion of the incoming mic audio track
- Streamed audio return path back to the browser over the same peer connection
- Served on port 8001, tunneled via ngrok for external/frontend testing

## Day-by-day log

| Day | Focus | Status |
|-----|-------|--------|
| 1 | React + Vite scaffold, `getUserMedia` mic capture, Start/Stop controls | ✅ Complete |
| 2 | `RTCPeerConnection` + SDP offer, backend `/offer` signaling endpoint, ngrok tunnel | ✅ Complete |
| 3 | Verified live audio flow browser → backend over the peer connection | ✅ Complete |
| 4 | Return-path streaming: backend → browser real-time playback via `ontrack` | ✅ Complete |
| 5 | Error handling, connection-drop recovery, resource cleanup, docs polish | ✅ Complete |

**Day 1**
- Scaffolded with `npm create vite@latest frontend -- --template react`, ESLint configured
- `navigator.mediaDevices.getUserMedia({ audio: true })` wired to Start/Stop Mic controls with live status display
- Verified raw `MediaStreamTrack` capture end-to-end
- Commit: `Day 1: React scaffold + mic capture wo

**Day 2**
- Created `RTCPeerConnection`, attached the captured mic track
- Generated SDP offer after ICE gathering completed, POSTed to backend `/offer`
- Built the aiortc signaling endpoint; frontend applies the returned answer via `setRemoteDescription`
- Backend live on port 8001, exposed via ngrok; frontend reads the tunnel URL from `VITE_BACKEND_URL`
- Commit: Day 2a: Add RTCPeerConnection, attach mic track, generate SDP offer
Day 2b: Build aiortc /offer signaling endpoint, expose backend via ngrok

**Day 3**
- Confirmed audio flows browser → backend over the live peer connection
- Logged connection-state transitions on the frontend; confirmed incoming track on the backend
- Commit: Day 3: Confirm mic audio streaming browser → backend, add connection-state logging

**Day 4**
- Wired the return path: backend streams audio back, frontend plays it live via an `<audio>` element on `ontrack`
- No buffering for a full response — playback starts as chunks arrive
- Full bi-directional stream confirmed working end-to-end
- Commit:Day 4: Stream backend audio back to browser, real-time playback via ontrack

**Day 5**
- Added error handling for mic-permission denial, connection drops, and unreachable backend
- Cleaned up peer connection and released mic tracks properly on Stop
- Finalized this README
- Commit: Day 5: Add error handling for mic denial/connection drops, cleanup on Stop, finalize Week 1 README

## Week 1 goal (official plan)

> **WebRTC Foundation:** Build the frontend React app and backend aiortc server to establish a bi-directional audio stream, bypassing slow HTTP protocols.

**Delivered.** Bi-directional streaming, live in both directions, with error handling and cleanup already in place — Week 2 starts from a stable foundation instead of a fragile prototype.

## Tech stack

- **Frontend:** React, Vite, native WebRTC APIs (`RTCPeerConnection`, `getUserMedia`), ESLint
- **Backend:** Python, aiortc, ngrok (dev tunneling)

## Running locally

```bash
# Backend (port 8001)
# start the aiortc server, then tunnel it:
ngrok http 8001

# Frontend
cd frontend
# .env -> VITE_BACKEND_URL=<your ngrok URL>
npm install
npm run dev   # http://localhost:5173
```

## Roadmap

- **Week 2:** Live VAD (Voice Activity Detection) state from the backend — "listening / user speaking / AI speaking" reflected in the UI
- **Week 3:** Real transcript + emotion data wired in from the ML pipeline; interruption-handling UI (AI stops talking when the user speaks)
- **Week 4 — Refine & Polish:** Cinematic waveform visualizer with emotion-driven color shifts, live transcript, emotion meter, and latency dashboard — the final production-ready frontend experience
