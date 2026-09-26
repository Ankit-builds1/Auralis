from aiortc import MediaStreamTrack

from app.audio.processor import AudioProcessor


class IncomingAudioTrack(MediaStreamTrack):
    """
    Receives audio frames from the browser and processes them
    through the audio/VAD pipeline.
    """

    kind = "audio"

    def __init__(self, source):
        super().__init__()
        self.source = source
        self.frame_count = 0
        self.processor = AudioProcessor()

    async def recv(self):
        """
        Receive one audio frame from the browser.
        """

        frame = await self.source.recv()

        self.frame_count += 1

        metadata = self.processor.process_frame(frame)

        if self.frame_count % 50 == 0:
            print(
                f"[AUDIO] "
                f"frames={metadata['frame_count']} "
                f"sample_rate={metadata['sample_rate']} "
                f"samples={metadata['samples']} "
                f"pts={metadata['pts']} "
                f"speech={metadata['is_speech']} "
                f"started={metadata['segment_started']} "
                f"finished={metadata['segment_finished']} "
                f"segment_frames={metadata['segment_frame_count']} "
                f"latency_ms={metadata['latency_ms']:.3f}"
            )

        return frame