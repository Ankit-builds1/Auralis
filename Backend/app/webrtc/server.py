import asyncio

from aiohttp import web
from aiortc import RTCSessionDescription

from app.webrtc.connection import WebRTCConnection
from app.webrtc.audio_track import IncomingAudioTrack
from app.webrtc.tts_track import TTSAudioTrack, play_test_tone


# Day 1 test: play a short beep when the data channel opens, to prove
# the outgoing audio path works. Set to False once real replies exist.
PLAY_CONNECT_BEEP = True


async def offer(request):
    """
    Receives a WebRTC SDP offer from the browser
    and returns an SDP answer.

    Incoming microphone audio is processed by the backend.
    Outgoing AI speech is sent to the browser through tts_track.
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

    # --------------------------------------------------
    # OUTGOING AUDIO TRACK (AI speech -> browser)
    # Must be added before the answer is created.
    # --------------------------------------------------

    tts_track = TTSAudioTrack()
    pc.addTrack(tts_track)

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

            if PLAY_CONNECT_BEEP:
                asyncio.create_task(
                    play_test_tone(tts_track)
                )

        @channel.on("close")
        def on_close():

            print(
                f"[WEBRTC] DataChannel closed: "
                f"{channel.label}"
            )

    # --------------------------------------------------
    # AUDIO TRACK (incoming microphone)
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