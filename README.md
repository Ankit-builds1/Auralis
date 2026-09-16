# Auralis — Backend

Real-time Voice-to-Voice (V2V) Emotion Engine — Backend module (Python, aiortc, Streaming AI Pipeline).

Part of Infotact Solutions' Advanced Generative AI Engineering internship (Project 2 of 3). Team of 3, role-swapped from OmniSight — this project: frontend/streaming UI (Teammate), backend/streaming (this module), audio ML (Teammate).

What Auralis does

Most conversational AI pipelines (STT → LLM → TTS) take 3-5 seconds and strip out emotional tone entirely. Auralis fixes both: it streams audio in real time over WebRTC, detects the speaker's emotional state (via a Wav2Vec2 sentiment classifier), and generates an emotion-matched voice response with sub-800ms latency — closer to a real human conversation than a typical voice bot.

Use case: a Crisis Negotiation Training Simulator, where a trainee speaks in a panicked voice and the AI responds with a de-escalation tone that adapts to their emotional intensity.

Backend responsibilities

This module owns the server-side real-time streaming pipeline:

Accepting the browser's WebRTC connection and exchanging SDP/ICE via a signaling endpoint
Receiving raw microphone audio frames over the peer connection (via aiortc)
Running Speech-to-Text (Whisper / Faster-Whisper) on incoming audio in near real time
Running Voice Activity Detection (Silero VAD) to know exactly when the user stops speaking
Feeding transcribed text + emotional context into a local LLM (Llama 3 via Ollama/vLLM) to generate a de-escalation response
Generating emotion-conditioned speech via a streaming TTS model (XTTSv2 / Bark) and pushing audio back over WebRTC in byte-chunks
Handling full-duplex interruption: instantly halting TTS generation if the user starts speaking mid-response
Progress — Week 1

Day 1 — Complete

Scaffolded the Python backend (FastAPI + aiortc), virtual environment and dependencies pinned
Implemented the WebRTC signaling endpoint (/offer) to accept the frontend's SDP offer and return an SDP answer
Verified a peer connection can be established end-to-end with the frontend's mic track (connection state logged to connected)
Commit: Day 1: FastAPI + aiortc signaling server working (branch: backend)

Day 2 — Up next

Attach an on_track handler to receive the incoming audio MediaStreamTrack
Pipe raw audio frames into a buffer and confirm audio bytes are actually arriving from the browser

Day 3 (planned)

Wire the buffered audio into Whisper.cpp / Faster-Whisper for streaming transcription
Log transcribed text in real time to confirm STT is working end-to-end

Day 4 (planned)

Integrate Silero VAD to detect end-of-speech and trigger the response pipeline
Begin measuring Time-to-First-Token (TTFT) from end-of-speech to first LLM token

Day 5 (planned)

Error handling (dropped connections, silence timeouts, reconnect logic) + this README's ongoing updates
Week 1 goal (per official plan)

WebRTC Foundation: Build the frontend React app and backend aiortc server to establish a bi-directional audio stream (bypassing slow HTTP protocols).

Tech stack
Python, FastAPI
aiortc (WebRTC peer connection, media handling)
asyncio for concurrent streaming
Whisper.cpp / Faster-Whisper (planned, Week 1–2)
Silero VAD (planned, Week 2)
Looking ahead



Week 2: Connect transcribed text to a local Llama 3 model (Ollama/vLLM) pre-prompted with a negotiation persona; implement Silero VAD for end-of-speech detection; expose VAD/listening state to the frontend
Week 3: Integrate zero-shot streaming TTS (XTTSv2 / Bark) conditioned on emotional baseline (calm vs. panicked); write chunked audio streaming logic back over WebRTC before the full sentence finishes generating
Week 4 (Refine & Polish): Implement full-duplex interruption handling — instantly halt TTS generation and resume listening the moment the human speaks; expose latency stats (TTFT, end-to-end round trip) for the frontend dashboard
