from aiohttp import web
from aiortc import RTCSessionDescription

from app.webrtc.connection import WebRTCConnection
from app.webrtc.audio_track import IncomingAudioTrack


async def offer(request):
    """
    Receives a WebRTC SDP offer from the browser
    and returns an SDP answer.
    """

    params = await request.json()

    if "sdp" not in params or "type" not in params:
        return web.json_response(
            {"error": "Invalid WebRTC offer"},
            status=400
        )

    connection = WebRTCConnection()
    pc = connection.pc

    offer = RTCSessionDescription(
        sdp=params["sdp"],
        type=params["type"]
    )

    @pc.on("track")
    def on_track(track):
        print(f"[WEBRTC] Received track: {track.kind}")

        if track.kind == "audio":
            audio_track = IncomingAudioTrack(track)
            pc.addTrack(audio_track)

    await pc.setRemoteDescription(offer)

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    })