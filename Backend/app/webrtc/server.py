import asyncio

from aiohttp import web
from aiortc import RTCSessionDescription

from app.webrtc.connection import WebRTCConnection
from app.webrtc.audio_track import IncomingAudioTrack


async def offer(request):
    """
    Receives a WebRTC SDP offer from the browser
    and returns an SDP answer.

    Incoming microphone audio is processed by the backend,
    but is NOT sent back to the browser.
    """

    params = await request.json()

    if "sdp" not in params or "type" not in params:
        return web.json_response(
            {"error": "Invalid WebRTC offer"},
            status=400,
        )

    connection = WebRTCConnection()
    pc = connection.pc

    audio_processor = None
    processing_task = None
    data_channel = None

    offer_description = RTCSessionDescription(
        sdp=params["sdp"],
        type=params["type"],
    )

    # --------------------------------------------------
    # DATA CHANNEL
    # --------------------------------------------------

    @pc.on("datachannel")
    def on_datachannel(channel):

        nonlocal data_channel

        data_channel = channel

        print(
            f"[WEBRTC] DataChannel received: "
            f"{channel.label}"
        )

        if audio_processor is not None:
            audio_processor.set_data_channel(
                channel
            )

        @channel.on("open")
        def on_open():

            print(
                f"[WEBRTC] DataChannel opened: "
                f"{channel.label}"
            )

            if audio_processor is not None:
                audio_processor.set_data_channel(
                    channel
                )

                audio_processor.send_event({
                    "type": "status",
                    "status": "connected",
                })

        @channel.on("close")
        def on_close():

            print(
                f"[WEBRTC] DataChannel closed: "
                f"{channel.label}"
            )

    # --------------------------------------------------
    # AUDIO TRACK
    # --------------------------------------------------

    @pc.on("track")
    def on_track(track):

        nonlocal audio_processor
        nonlocal processing_task

        print(
            f"[WEBRTC] Received track: "
            f"{track.kind}"
        )

        if track.kind != "audio":
            return

        print(
            "[WEBRTC] Incoming microphone track "
            "attached to audio processor"
        )

        audio_processor = IncomingAudioTrack(
            track,
            data_channel,
        )

        processing_task = asyncio.create_task(
            audio_processor.run()
        )

    # --------------------------------------------------
    # CONNECTION CLOSED
    # --------------------------------------------------

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():

        print(
            f"[WEBRTC] Connection state: "
            f"{pc.connectionState}"
        )

        if pc.connectionState in (
            "failed",
            "closed",
            "disconnected",
        ):

            if audio_processor is not None:
                audio_processor.stop()

            if processing_task is not None:
                processing_task.cancel()

            await connection.close()

    # --------------------------------------------------
    # SET REMOTE DESCRIPTION
    # --------------------------------------------------

    await pc.setRemoteDescription(
        offer_description
    )

    # --------------------------------------------------
    # CREATE ANSWER
    # --------------------------------------------------

    answer = await pc.createAnswer()

    await pc.setLocalDescription(
        answer
    )

    print(
        "[WEBRTC] SDP answer created"
    )

    return web.json_response({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
    })