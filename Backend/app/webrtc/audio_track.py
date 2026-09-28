import asyncio
import json

from app.audio.processor import AudioProcessor


# Print a progress line every N frames instead of every frame.
# 250 frames x 20 ms = one line every 5 seconds.
DEBUG_PRINT_EVERY = 250


class IncomingAudioTrack:
    """
    Receives microphone audio from the browser and continuously
    processes it through the audio/VAD/STT pipeline.

    The processed microphone audio is NOT returned to the browser.
    This prevents delayed audio/echo from being played back.

    Frame processing (VAD + Whisper) runs in a worker thread so the
    asyncio event loop is never blocked. This keeps WebRTC receiving
    audio and sending DataChannel messages while Whisper transcribes.
    """

    def __init__(self, source, data_channel=None):
        self.source = source
        self.data_channel = data_channel

        self.frame_count = 0
        self.processor = AudioProcessor()
        self.running = True

        print("[AUDIO] IncomingAudioTrack initialized")

    def set_data_channel(self, channel):
        """Attach the WebRTC DataChannel."""
        self.data_channel = channel

        print(
            f"[AUDIO] DataChannel attached: "
            f"{getattr(channel, 'label', 'unknown')}"
        )

    def send_event(self, payload):
        """Send a JSON event to the frontend DataChannel."""

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
        Continuously receive microphone frames and process them.
        """

        print("[AUDIO] Audio processing loop started")

        try:
            while self.running:

                # -----------------------------------------
                # RECEIVE AUDIO FRAME
                # -----------------------------------------

                try:
                    frame = await self.source.recv()

                except asyncio.CancelledError:
                    print(
                        "[AUDIO] Processing task cancelled"
                    )
                    break

                except Exception as exc:
                    print(
                        "[AUDIO ERROR] source.recv() failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    break

                self.frame_count += 1

                # Periodic progress line (printing every frame
                # at ~50 frames/sec slows the whole server down).
                if self.frame_count % DEBUG_PRINT_EVERY == 0:
                    print(
                        f"[AUDIO DEBUG] "
                        f"frame={self.frame_count} "
                        f"samples={getattr(frame, 'samples', None)} "
                        f"sample_rate={getattr(frame, 'sample_rate', None)}"
                    )

                # -----------------------------------------
                # PROCESS FRAME (in a worker thread)
                # -----------------------------------------

                try:
                    # VAD + Whisper are blocking calls. Running them
                    # in a thread keeps the event loop free, so WebRTC
                    # keeps receiving audio and sending messages while
                    # Whisper is transcribing.
                    metadata = await asyncio.to_thread(
                        self.processor.process_frame,
                        frame
                    )

                    # -----------------------------------------
                    # SEND FRAME INFORMATION
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
                    # SPEECH STARTED
                    # -----------------------------------------

                    if metadata["segment_started"]:

                        print(
                            f"[VAD] Speech started "
                            f"(frame={self.frame_count})"
                        )

                        self.send_event({
                            "type": "vad",
                            "vad_state": "speaking",
                        })

                    # -----------------------------------------
                    # SPEECH FINISHED
                    # -----------------------------------------

                    if metadata["segment_finished"]:

                        print(
                            f"[VAD] Speech finished "
                            f"(frame={self.frame_count}, "
                            f"segment_frames={metadata['segment_frame_count']}, "
                            f"processing_ms={metadata['latency_ms']:.1f})"
                        )

                        self.send_event({
                            "type": "vad",
                            "vad_state": "processing",
                        })

                        # -------------------------------------
                        # STT RESULT
                        # -------------------------------------

                        transcript = metadata.get(
                            "transcript",
                            ""
                        )

                        if transcript:

                            print(
                                f"[STT] Transcript: "
                                f"{transcript}"
                            )

                            self.send_event({
                                "type": "transcript",
                                "text": transcript,
                            })

                        else:

                            print(
                                "[STT] No transcript returned"
                            )

                        # Tell frontend that processing
                        # of the speech segment is complete.
                        self.send_event({
                            "type": "vad",
                            "vad_state": "idle",
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
        """
        Stop audio processing.
        """

        if self.running:
            print(
                "[AUDIO] Stop requested"
            )

        self.running = False