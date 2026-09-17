from aiortc import MediaStreamTrack


class IncomingAudioTrack(MediaStreamTrack):
    """
    Receives audio frames from the browser.

    Day 2:
    - Receive frames
    - Count frames
    - Log basic information

    Later:
    - Send frames to the audio pipeline
    - VAD
    - STT
    - Emotion analysis
    """

    kind = "audio"

    def __init__(self, source):
        super().__init__()
        self.source = source
        self.frame_count = 0

    async def recv(self):
        frame = await self.source.recv()

        self.frame_count += 1

        if self.frame_count % 50 == 0:
            print(
                f"[AUDIO] Received {self.frame_count} frames"
            )

        return frame