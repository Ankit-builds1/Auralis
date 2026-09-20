from aiohttp import web
import aiohttp_cors

from app.config.settings import HOST, PORT
from app.webrtc.server import offer


async def health(request):
    return web.json_response({
        "status": "ok",
        "service": "auralis-backend"
    })


def create_app():
    app = web.Application()

    app.router.add_get("/health", health)
    app.router.add_post("/webrtc/offer", offer)

    cors = aiohttp_cors.setup(app)

    cors_config = {
        "http://localhost:5173": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods=["POST", "OPTIONS"],
        )
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