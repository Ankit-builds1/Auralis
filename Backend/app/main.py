from aiohttp import web

from app.config.settings import HOST, PORT


async def health(request):
    return web.json_response({
        "status": "ok",
        "service": "auralis-backend"
    })


def create_app():
    app = web.Application()

    app.router.add_get("/health", health)

    return app


if __name__ == "__main__":
    app = create_app()

    web.run_app(
        app,
        host=HOST,
        port=PORT
    )