from aiortc import MediaStreamTrack

from app.audio.processor import AudioProcessor


class IncomingAudioTrack(MediaStreamTrack):
    """
    Receives audio frames from the browser.

    Day 3:
    - Receive audio frames
    - Process frames through AudioProcessor
    - Inspect basic metadata

    Later:
    - Voice Activity Detection
    - Speech-to-Text
    - Emotion analysis
    """

    kind = "audio"

    def __init__(self, source):
        super().__init__()
        self.source = source
        self.frame_count = 0
        self.processor = AudioProcessor()

    async def recv(self):
        frame = await self.source.recv()

        self.frame_count += 1

        metadata = self.processor.process_frame(frame)

        if self.frame_count % 50 == 0:
            print(
                f"[AUDIO] frames={metadata['frame_count']} "
                f"sample_rate={metadata['sample_rate']} "
                f"samples={metadata['samples']} "
                f"pts={metadata['pts']}"
            )

        return frame