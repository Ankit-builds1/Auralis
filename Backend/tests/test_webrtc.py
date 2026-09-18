from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import pytest

from app.main import create_app


@pytest.mark.asyncio
async def test_health_endpoint():
    app = create_app()

    server = TestServer(app)
    client = TestClient(server)

    await client.start_server()

    response = await client.get("/health")

    assert response.status == 200

    data = await response.json()

    assert data["status"] == "ok"
    assert data["service"] == "auralis-backend"

    await client.close()


@pytest.mark.asyncio
async def test_webrtc_offer_validation():
    app = create_app()

    server = TestServer(app)
    client = TestClient(server)

    await client.start_server()

    response = await client.post(
        "/webrtc/offer",
        json={}
    )

    assert response.status == 400

    data = await response.json()

    assert data["error"] == "Invalid WebRTC offer"

    await client.close()


@pytest.mark.asyncio
async def test_cors_allows_vite_frontend():
    app = create_app()
    server = TestServer(app)
    client = TestClient(server)

    await client.start_server()

    response = await client.get(
        "/health",
        headers={
            "Origin": "http://localhost:5173"
        }
    )

    assert response.status == 200
    assert response.headers["Access-Control-Allow-Origin"] == "http://localhost:5173"

    await client.close()
