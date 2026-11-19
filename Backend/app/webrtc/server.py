import asyncio
import json

from aiohttp import web
from aiortc import RTCSessionDescription

from app.webrtc.connection import WebRTCConnection
from app.webrtc.audio_track import IncomingAudioTrack
from app.webrtc.tts_track import TTSAudioTrack, play_test_tone


# Day 1 test: play a short beep when the data channel opens, to prove
# the outgoing audio path works. Off since Week 3 Day 3: Auralis now
# speaks its real replies (Piper TTS).
PLAY_CONNECT_BEEP = False


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

        # Week 4 Day 2: messages FROM the browser.
        # {"type": "interrupt"} = STOP AURALIS button
        @channel.on("message")
        def on_message(message):

            try:
                data = json.loads(message)
            except (TypeError, ValueError):
                return

            if not isinstance(data, dict):
                return

            if data.get("type") == "interrupt" and audio_processor is not None:
                audio_processor.request_interrupt("manual")

            # Week 4: {"type": "set_mode", "mode": "training" | "assistant",
            #          "scenario": "<id from negotiation.SCENARIOS>" (optional)}
            if data.get("type") == "set_mode" and audio_processor is not None:
                audio_processor.set_mode(data.get("mode", "assistant"), data.get("scenario"))

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

        # Week 3 Day 3: the processor also gets the outgoing voice track,
        # so it can speak replies and ignore its own echo
        audio_processor = IncomingAudioTrack(
            track,
            data_channel,
            tts_track=tts_track,
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