import asyncio
import os
import shutil
import subprocess

from aiohttp import web
import aiohttp_cors

from app.config.settings import HOST, PORT
from app.llm import ollama_client
from app.ml_clients import emotion_engine, text_emotion
from app.stt.engine import get_stt, load_and_warmup
from app.tts import engine as tts_engine
from app.webrtc.server import offer


def _gpu_name():
    """Name of the NVIDIA GPU (None if there is none). Checked once at startup."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=3,
        )
        name = out.stdout.strip().splitlines()[0] if out.returncode == 0 else ""
        return name.replace("NVIDIA ", "").replace("GeForce ", "") or None
    except Exception:
        return None


GPU_NAME = _gpu_name()


async def health(request):
    """
    Backend status. The first five keys are the original ones (Week 1-5);
    the rest (Week 3 Day 6) tell the dashboard which models run where.
    """
    emotion_on = emotion_engine.is_ready()
    text_on = text_emotion.is_ready()
    tts_on = tts_engine.is_ready()

    return web.json_response({
        "status": "ok",
        "service": "auralis-backend",
        "emotion": emotion_on,
        "text_emotion": text_on,
        "tts": tts_on,
        "stt": get_stt().info(),
        "gpu": GPU_NAME,
        "emotion_model": emotion_engine.MODEL_KEY if emotion_on else None,
        "emotion_min_confidence": emotion_engine.MIN_CONFIDENCE,
        "text_model": text_emotion.MODEL_NAME if text_on else None,
        "tts_engine": "piper" if tts_on else None,
        "tts_voice": tts_engine.VOICE_PATH.stem if tts_on else None,
        "llm": {"model": ollama_client.OLLAMA_MODEL},
    })


async def on_startup(app):
    """
    Load all models BEFORE the first user connects, so the first
    sentence is not slowed down by model loading.
    """
    print("[STARTUP] Loading Whisper model...")
    await asyncio.to_thread(load_and_warmup)

    print("[STARTUP] Loading emotion model (WavLM-Large)...")
    await asyncio.to_thread(emotion_engine.load_and_warmup)

    print("[STARTUP] Loading text emotion model (DistilRoBERTa)...")
    await asyncio.to_thread(text_emotion.load_and_warmup)

    print("[STARTUP] Loading voice (Piper TTS)...")
    await asyncio.to_thread(tts_engine.load_and_warmup)

    print(f"[STARTUP] Warming up LLM ({ollama_client.OLLAMA_MODEL})...")
    await ollama_client.warmup()

    stt = get_stt().info()
    print(f"[STARTUP] Ready  (STT {stt['model']} on {stt['device']}"
          f"{', GPU ' + GPU_NAME if GPU_NAME else ''})")


def create_app():
    app = web.Application()

    app.on_startup.append(on_startup)

    app.router.add_get("/health", health)
    app.router.add_post("/webrtc/offer", offer)

    cors = aiohttp_cors.setup(app)

    # Deployment: browser origins allowed to call the backend.
    # Local dev ports are always allowed; add the Vercel URL(s) in .env:
    #   CORS_ORIGINS=https://auralis.vercel.app,https://other.vercel.app
    origins = ["http://localhost:5173", "http://localhost:5174"]
    origins += [o.strip().rstrip("/") for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
    cors_config = {
        origin: aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
        )
        for origin in dict.fromkeys(origins)
    }
    print(f"[STARTUP] CORS origins: {', '.join(cors_config)}")

    for route in list(app.router.routes()):
        cors.add(route, cors_config)

    return app


if __name__ == "__main__":
    app = create_app()
    web.run_app(app, host=HOST, port=PORT)