# Auralis — Frontend

Real-time Voice-to-Voice (V2V) Emotion Engine — Frontend module (React + WebRTC).

Part of Infotact Solutions' Advanced Generative AI Engineering internship (Project 2 of 3). Team of 3, role-swapped from OmniSight — this project: **frontend/streaming UI (this module)**, backend/streaming (Siddhant), audio ML (Priya).

## What Auralis does

Most conversational AI pipelines (STT → LLM → TTS) take 3-5 seconds and strip out emotional tone entirely. Auralis fixes both: it streams audio in real time over WebRTC, detects the speaker's emotional state (via Wav2Vec2 on the backend), and generates an emotion-matched voice response with sub-800ms latency — closer to a real human conversation than a typical voice bot.

**Use case:** a Crisis Negotiation Training Simulator, where a trainee speaks in a panicked voice and the AI responds with a de-escalation tone that adapts to their emotional intensity.

## Frontend responsibilities

This module owns the **browser-side real-time audio pipeline**:

- Capturing raw microphone audio via `getUserMedia`
- Establishing a low-latency WebRTC peer connection to the backend (bypassing slow HTTP request/response cycles)
- Streaming mic audio to the backend and playing back the AI's response audio as it arrives, in chunks
- (Week 4) A cinematic, emotion-reactive UI: live waveform, live transcript, emotion meter, and latency stats

## Progress — Week 1

**Day 1 — Complete**
- Scaffolded the React app with Vite (`npm create vite@latest frontend -- --template react`), ESLint configured
- Implemented microphone capture using `navigator.mediaDevices.getUserMedia({ audio: true })`
- Verified raw `MediaStreamTrack` capture end-to-end: Start/Stop Mic controls, status display, and console-logged audio track confirming the browser has live mic access
- Commit: `Day 1: React scaffold + mic capture working` (branch: `frontend`)

**Day 2 — Up next**
- Set up `RTCPeerConnection`, attach the captured mic track, generate an SDP offer
- Begin signaling exchange with the backend's WebRTC endpoint (Siddhant's side)

**Day 3 (planned)**
- Confirm audio is actually flowing from browser to backend over the peer connection

**Day 4 (planned)**
- Handle the return path: play back streamed audio chunks from the backend in real time, without waiting for the full response

**Day 5 (planned)**
- Error handling (mic permission denied, connection drops) + this README's ongoing updates

## Week 1 goal (per official plan)

> **WebRTC Foundation:** Build the frontend React app and backend aiortc server to establish a bi-directional audio stream, bypassing slow HTTP protocols.

## Tech stack

- React + Vite
- Native WebRTC APIs (`RTCPeerConnection`, `getUserMedia`)
- ESLint

## Looking ahead

- **Week 2:** Reflect Voice Activity Detection (VAD) state from the backend — show "listening / user speaking / AI speaking" in the UI
- **Week 3:** Wire in real transcript + emotion data from the backend/ML pipeline; build interruption-handling UI (AI stops talking when the user speaks)
- **Week 4 (Refine & Polish):** Cinematic waveform visualizer with emotion-driven color shifts, live transcript, emotion meter, and latency dashboard — the final polished frontend experience
