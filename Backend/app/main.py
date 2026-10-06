import asyncio

from aiohttp import web
import aiohttp_cors

from app.config.settings import HOST, PORT
from app.llm import ollama_client
from app.stt.engine import load_and_warmup
from app.webrtc.server import offer


async def health(request):
    return web.json_response({
        "status": "ok",
        "service": "auralis-backend"
    })


async def on_startup(app):
    """
    Load models BEFORE the first user connects, so the first sentence
    is not slowed down by model loading.
    """
    print("[STARTUP] Loading Whisper model...")
    await asyncio.to_thread(load_and_warmup)

    print(f"[STARTUP] Warming up LLM ({ollama_client.OLLAMA_MODEL})...")
    await ollama_client.warmup()

    print("[STARTUP] Ready")


def create_app():
    app = web.Application()

    app.on_startup.append(on_startup)

    app.router.add_get("/health", health)
    app.router.add_post("/webrtc/offer", offer)

    cors = aiohttp_cors.setup(app)

    cors_config = {
        "http://localhost:5173": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
        ),
        "http://localhost:5174": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
        ),
    }

    for route in list(app.router.routes()):
        cors.add(route, cors_config)

    return app


if __name__ == "__main__":
    app = create_app()

    web.run_app(
        app,
        host=HOST,
        port=PORT
    )