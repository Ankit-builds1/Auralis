# Auralis

Auralis is a voice-to-voice assistant and a crisis negotiation training simulator. In assistant mode, a user speaks and Auralis answers out loud. In training mode, the user practises a difficult conversation with a simulated caller and sees how each turn changes the caller’s agitation.

**Infotact Solutions · Advanced Generative AI Engineering (Vol. II) · Project 2**

[Open the deployed frontend](https://auralis-hazel.vercel.app/) · [Mid-review snapshot](https://github.com/Ankit-builds1/Auralis/blob/91702177c873b26798c11561d6f5efb3511ccf38/README.md)

## What we added after the mid-review

At mid-review, we could stream microphone audio to the backend, detect the end of a sentence, transcribe it with Faster-Whisper, and display a local Llama reply token by token. We had also trained a separate acoustic emotion classifier. Our transcription audit captured **5 out of 5** speech segments. The remaining work was to bring emotion into the conversation, let the user hear and interrupt the reply, and turn the engine into a training experience.

| Area | What we added | What a reviewer can see |
|---|---|---|
| Spoken replies | Piper synthesises the first phrase while Llama continues generating text. The backend streams the audio back over WebRTC and selects a voice style for the situation. | The reply appears as text and plays through the browser. The dashboard measures when the first audio arrives. |
| Emotion from voice and words | We added a WavLM-based voice classifier, adapted it using team recordings, and combined it with a DistilRoBERTa text classifier. Confidence thresholds and a personal voice baseline help handle uncertain predictions. | Engineer view shows the voice prediction, the text prediction, and the emotion Auralis used. |
| Interruptions | The user can speak over Auralis or press Stop. The backend cancels the current reply and clears queued audio. A loudness check helps distinguish the user from speaker echo. | The interrupted reply is marked in the conversation. The dashboard reports how long the voice took to stop. |
| Negotiation training | We built **12 scenarios** with distinct callers and starting conditions. Rules update simulated agitation from tone, wording, and interruptions. Llama uses the updated score when playing the caller. | The trainee sees the agitation meter, reasons for each change, and an end-of-session score. |
| Review and replay | We added Experience and Engineer views, waveforms for both voices, emotion testing, latency breakdowns, and controls to save, export, import, and replay sessions. Replay labels distinguish recorded metrics from live backend health. | A reviewer can replay a saved conversation and inspect individual turns even while the backend is offline. |
| Deployment | The React frontend is hosted on Vercel. A team laptop runs the Python backend and models. A Cloudflare Tunnel carries HTTPS signalling, and WebRTC uses a TURN relay when needed for audio. | The frontend reports backend health separately. A live microphone conversation requires a reachable backend. |

## How a live turn works

1. The React client streams microphone audio over WebRTC. Silero VAD detects speech and the pause at the end of the turn.
2. Faster-Whisper produces the transcript. WavLM analyses the voice, and DistilRoBERTa analyses the words. The backend combines their predictions.
3. In assistant mode, Llama receives conversation context and an emotion cue. In training mode, the simulator updates the caller’s agitation and supplies the scenario and current state.
4. Ollama streams Llama’s reply. Piper synthesises phrases as they become available. The browser plays the audio and displays transcript, emotion, timing, and interruption events.

## What we measured

| Check | Recorded result | Interpretation |
|---|---|---|
| Voice emotion on CREMA-D | **48.9%** six-class accuracy for the mid-review temporal SVM and **74.6%** for WavLM Large | The newer model improved recognition on held-out actors. [WavLM results](ml/results/week3_ssl/wavlm-large_summary.csv) |
| Speaker adaptation | **19.4% → 52.8%** on **36 held-out team recordings** | Adaptation helped on our voices. The small sample limits what we can conclude about other speakers. [Adaptation results](ml/results/week3_adapt/wavlm-large_adapt_summary.csv) |
| Time to first audible reply | **4.78 s p50** and **13.33 s p95** over **10 localhost turns** on Ankit’s laptop with Whisper on CPU | This measures the wait for speech, including transcription, reply generation, and synthesis. The **800 ms** end-to-end goal remains open. [Latency audit](docs/frontend-README.md) |
| Negotiation walkthrough | In the locked-in office worker scenario, agitation moved from **85 to 9** over **5 turns**, with a final score of **86/100**. | This is one run of the simulator’s scoring rules. [Training results](docs/Backend-README.md) |

## Run the project

The `main` branch contains [the frontend](frontend/), [the backend](Backend/), and [the ML experiments](ml/). For a recorded review, open the deployed frontend, select a saved session or load an exported session, and choose **Replay**. Use **Engineer** view to inspect individual turns.

For a live session on Windows, run the following from the repository root in separate terminals. Start Ollama first. The backend configuration lists the trained emotion bundle and Piper voice files required for the full demo. [Backend configuration](Backend/.env.example)

**Backend (PowerShell)**

```powershell
cd Backend
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (!(Test-Path .env)) { Copy-Item .env.example .env }
ollama pull llama3.2:3b
.\.venv\Scripts\python.exe -m app.main
```

**Frontend (PowerShell)**

```powershell
cd frontend
npm ci
$env:VITE_BACKEND_URL = "http://localhost:8001"
npm run dev
```

Open the local URL printed by Vite. Check backend health, then start the microphone and connect. Deployment settings, model setup, and experiment instructions are in the [frontend notes](docs/frontend-README.md), [backend notes](docs/Backend-README.md), and [ML report](docs/ML-README.md).

## Team

| Member | Role | Owns |
|---|---|---|
| **Ankit Dash** | Frontend & Real-Time UI | WebRTC client, live dashboard, latency instrumentation, conversation UI |
| **Priya Nirmal** | Backend & Streaming | aiortc server, VAD, STT pipeline, LLM integration, audio streaming |
| **Siddhant** | ML & Generative Models | Emotion recognition, feature engineering, LLM prompting, TTS |
