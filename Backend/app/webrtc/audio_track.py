import asyncio
import json

from app.audio.processor import AudioProcessor


class IncomingAudioTrack:
    """
    Receives audio frames from the browser and continuously
    processes them through the audio/VAD pipeline.

    IMPORTANT:
    This class does NOT send the microphone audio back to the browser.
    That prevents the delayed echo/voice repetition problem.
    """

    def __init__(self, source, data_channel=None):
        self.source = source
        self.data_channel = data_channel

        self.frame_count = 0
        self.processor = AudioProcessor()

        self.running = True

        print("[AUDIO] IncomingAudioTrack initialized")

    def set_data_channel(self, channel):
        self.data_channel = channel
        print(
            f"[AUDIO] DataChannel attached: "
            f"{getattr(channel, 'label', 'unknown')}"
        )

    def send_event(self, payload):
        """
        Send a JSON event to the frontend DataChannel.
        """

        if self.data_channel is None:
            return

        try:
            if self.data_channel.readyState != "open":
                return

            self.data_channel.send(
                json.dumps(payload)
            )

        except Exception as exc:
            print(
                f"[AUDIO] DataChannel send error: "
                f"{type(exc).__name__}: {exc}"
            )

    async def run(self):
        """
        Continuously receive and process microphone frames.
        """

        print("[AUDIO] Audio processing loop started")

        try:
            while self.running:

                try:
                    frame = await self.source.recv()

                except asyncio.CancelledError:
                    break

                except Exception as exc:
                    print(
                        f"[AUDIO ERROR] source.recv() failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    break

                self.frame_count += 1

                try:
                    metadata = self.processor.process_frame(
                        frame
                    )

                    # -----------------------------------------
                    # Console output
                    # -----------------------------------------

                    print(
                        f"[AUDIO] "
                        f"frame={self.frame_count} "
                        f"speech={metadata['is_speech']} "
                        f"started={metadata['segment_started']} "
                        f"finished={metadata['segment_finished']} "
                        f"segment_frames={metadata['segment_frame_count']} "
                        f"latency_ms={metadata['latency_ms']:.3f}"
                    )

                    # -----------------------------------------
                    # Send frame/VAD information to frontend
                    # -----------------------------------------

                    self.send_event({
                        "type": "audio_frame",
                        "frame": self.frame_count,
                        "sample_rate": metadata["sample_rate"],
                        "samples": metadata["samples"],
                        "pts": metadata["pts"],
                        "is_speech": metadata["is_speech"],
                        "segment_started": metadata[
                            "segment_started"
                        ],
                        "segment_finished": metadata[
                            "segment_finished"
                        ],
                        "segment_frame_count": metadata[
                            "segment_frame_count"
                        ],
                        "latency_ms": round(
                            metadata["latency_ms"],
                            3
                        ),
                    })

                    # -----------------------------------------
                    # VAD state
                    # -----------------------------------------

                    if metadata["segment_started"]:
                        self.send_event({
                            "type": "vad",
                            "vad_state": "speaking",
                        })

                    elif metadata["segment_finished"]:
                        self.send_event({
                            "type": "vad",
                            "vad_state": "processing",
                        })

                except Exception as exc:
                    print(
                        f"[AUDIO ERROR] "
                        f"Processing frame={self.frame_count}: "
                        f"{type(exc).__name__}: {exc}"
                    )

                    self.send_event({
                        "type": "audio_error",
                        "frame": self.frame_count,
                        "error": str(exc),
                    })

        finally:
            self.running = False

            print(
                f"[AUDIO] Processing loop stopped "
                f"after {self.frame_count} frames"
            )

    def stop(self):
        self.running = False