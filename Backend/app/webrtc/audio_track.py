import asyncio
import json
import time

from app.audio.processor import AudioProcessor
from app.llm import ollama_client
from app.stt.engine import get_stt


# Print a progress line every N frames instead of every frame.
# 250 frames x 20 ms = one line every 5 seconds.
DEBUG_PRINT_EVERY = 250

# How many previous messages (user + assistant) the LLM sees.
# 6 = the last 3 exchanges.
HISTORY_MESSAGES = 6


class IncomingAudioTrack:
    """
    Receives microphone audio from the browser and runs the pipeline:

        frame -> VAD/segmenting (fast, every frame)
                   |
                   +-- segment finished --> background "turn" task:
                                              Whisper  -> transcript
                                              Llama 3  -> llm_token ... llm_done

    The frame loop never waits for Whisper or the LLM, so audio keeps
    flowing and VAD keeps working while a reply is being generated.

    DataChannel messages sent to the frontend:
        audio_frame   ~50/s   frame info for the dashboard
        vad           speaking / processing / idle
        transcript    {text, stt_ms}
        llm_start     reply is starting
        llm_token     {text, first, ttft_ms (first token only)}
        llm_done      {text, ttft_ms, total_ms, tokens}
        llm_error     {error}
        audio_error   {frame, error}

    Timing (all measured from the moment the VAD detected end of speech):
        stt_ms   = end of speech -> transcript ready
        ttft_ms  = end of speech -> first LLM token   (mid-project "Latency Check")
        total_ms = end of speech -> full LLM reply
    """

    def __init__(self, source, data_channel=None):
        self.source = source
        self.data_channel = data_channel

        self.frame_count = 0
        self.processor = AudioProcessor()
        self.stt = get_stt()  # shared model, already warmed up at startup
        self.running = True

        # Turns are handled one at a time, in order
        self.turn_lock = asyncio.Lock()
        self.turn_tasks = set()

        # Short conversation memory for the LLM (this connection only)
        self.history = []

        # Is the user currently speaking? (decides which VAD state to
        # show when a reply finishes)
        self.user_speaking = False

        # Turns waiting for or running Whisper/LLM
        self.pending_turns = 0

        print("[AUDIO] IncomingAudioTrack initialized")

    # ==========================================================
    # DataChannel
    # ==========================================================

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

            self.data_channel.send(json.dumps(payload))

        except Exception as exc:
            print(
                f"[AUDIO] DataChannel send error: "
                f"{type(exc).__name__}: {exc}"
            )

    def _send_vad(self, state):
        self.send_event({"type": "vad", "vad_state": state})

    def _idle_or_speaking(self):
        """VAD state to show when nothing new has started."""
        if self.user_speaking:
            self._send_vad("speaking")
        elif self.pending_turns > 0:
            self._send_vad("processing")
        else:
            self._send_vad("idle")

    # ==========================================================
    # Frame loop
    # ==========================================================

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
                    print("[AUDIO] Processing task cancelled")
                    break

                except Exception as exc:
                    print(
                        "[AUDIO ERROR] source.recv() failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    break

                self.frame_count += 1

                if self.frame_count % DEBUG_PRINT_EVERY == 0:
                    print(
                        f"[AUDIO DEBUG] "
                        f"frame={self.frame_count} "
                        f"samples={getattr(frame, 'samples', None)} "
                        f"sample_rate={getattr(frame, 'sample_rate', None)}"
                    )

                # -----------------------------------------
                # VAD + SEGMENTING (fast, in a worker thread)
                # -----------------------------------------

                try:
                    metadata = await asyncio.to_thread(
                        self.processor.process_frame,
                        frame,
                    )
                except Exception as exc:
                    print(
                        f"[AUDIO ERROR] Processing frame={self.frame_count}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    self.send_event({
                        "type": "audio_error",
                        "frame": self.frame_count,
                        "error": str(exc),
                    })
                    continue

                self.send_event({
                    "type": "audio_frame",
                    "frame": self.frame_count,
                    "sample_rate": metadata["sample_rate"],
                    "samples": metadata["samples"],
                    "pts": metadata["pts"],
                    "is_speech": metadata["is_speech"],
                    "segment_started": metadata["segment_started"],
                    "segment_finished": metadata["segment_finished"],
                    "segment_frame_count": metadata["segment_frame_count"],
                    "latency_ms": round(metadata["latency_ms"], 3),
                })

                # -----------------------------------------
                # SPEECH STARTED
                # -----------------------------------------

                if metadata["segment_started"]:
                    self.user_speaking = True
                    print(f"[VAD] Speech started (frame={self.frame_count})")
                    self._send_vad("speaking")

                # -----------------------------------------
                # SPEECH FINISHED
                # -----------------------------------------

                if metadata["segment_finished"]:
                    self.user_speaking = False

                    print(
                        f"[VAD] Speech finished "
                        f"(frame={self.frame_count}, "
                        f"segment_frames={metadata['segment_frame_count']})"
                    )

                    if metadata.get("segment_skipped") or not metadata["segment"]:
                        # Too short (cough, click). No transcript will come.
                        self._idle_or_speaking()
                        continue

                    self.pending_turns += 1
                    self._send_vad("processing")

                    # Whisper + LLM run in the background so this loop
                    # keeps receiving audio
                    task = asyncio.create_task(
                        self._handle_turn(
                            metadata["segment"],
                            metadata.get("speech_end_time") or time.perf_counter(),
                        )
                    )
                    self.turn_tasks.add(task)
                    task.add_done_callback(self.turn_tasks.discard)

        finally:
            self.running = False
            self._cancel_turns()

            print(
                f"[AUDIO] Processing loop stopped "
                f"after {self.frame_count} frames"
            )

    # ==========================================================
    # One conversation turn: STT -> LLM
    # ==========================================================

    async def _handle_turn(self, segment, speech_end_time):
        async with self.turn_lock:
            try:
                # ---------------- STT ----------------
                transcript = await asyncio.to_thread(
                    self.stt.transcribe,
                    segment,
                )

                stt_ms = round((time.perf_counter() - speech_end_time) * 1000)

                if not transcript:
                    print(f"[STT] No transcript ({stt_ms} ms)")
                    return

                print(f"[STT] Transcript ({stt_ms} ms after speech end): {transcript}")

                self.send_event({
                    "type": "transcript",
                    "text": transcript,
                    "stt_ms": stt_ms,
                })

                # ---------------- LLM ----------------
                await self._stream_llm_reply(transcript, speech_end_time)

            except asyncio.CancelledError:
                print("[TURN] Cancelled")
                raise

            except Exception as exc:
                print(f"[TURN ERROR] {type(exc).__name__}: {exc}")
                self.send_event({"type": "llm_error", "error": str(exc)})

            finally:
                self.pending_turns = max(0, self.pending_turns - 1)
                if self.running:
                    self._idle_or_speaking()

    async def _stream_llm_reply(self, transcript, speech_end_time):
        # TODO (emotion integration): pass Siddhant's emotion label here
        messages = ollama_client.build_messages(
            self.history,
            transcript,
            emotion=None,
        )

        self.send_event({"type": "llm_start"})

        reply_parts = []
        ttft_ms = None

        try:
            async for token in ollama_client.stream_reply(messages):
                if ttft_ms is None:
                    ttft_ms = round((time.perf_counter() - speech_end_time) * 1000)
                    print(f"[LLM] First token: TTFT = {ttft_ms} ms")

                    self.send_event({
                        "type": "llm_token",
                        "text": token,
                        "first": True,
                        "ttft_ms": ttft_ms,
                    })
                else:
                    self.send_event({
                        "type": "llm_token",
                        "text": token,
                        "first": False,
                    })

                reply_parts.append(token)

        except RuntimeError as exc:
            print(f"[LLM ERROR] {exc}")
            self.send_event({"type": "llm_error", "error": str(exc)})
            return

        reply = "".join(reply_parts).strip()
        total_ms = round((time.perf_counter() - speech_end_time) * 1000)

        print(f"[LLM] Reply ({total_ms} ms, TTFT {ttft_ms} ms): {reply}")

        self.send_event({
            "type": "llm_done",
            "text": reply,
            "ttft_ms": ttft_ms,
            "total_ms": total_ms,
            "tokens": len(reply_parts),
        })

        # Remember this exchange for the next turn
        if reply:
            self.history.append({"role": "user", "content": transcript})
            self.history.append({"role": "assistant", "content": reply})
            self.history = self.history[-HISTORY_MESSAGES:]

    # ==========================================================
    # Stop
    # ==========================================================

    def _cancel_turns(self):
        for task in list(self.turn_tasks):
            task.cancel()

    def stop(self):
        """
        Stop audio processing and any reply in progress.
        """

        if self.running:
            print("[AUDIO] Stop requested")

        self.running = False
        self._cancel_turns()