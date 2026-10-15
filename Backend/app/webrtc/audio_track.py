import asyncio
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

from app.audio.processor import AudioProcessor
from app.llm import ollama_client
from app.ml_clients import emotion_engine
from app.stt.engine import get_stt
from app.tts import engine as tts_engine


# Print a progress line every N frames instead of every frame.
# 250 frames x 20 ms = one line every 5 seconds.
DEBUG_PRINT_EVERY = 250

# How many previous messages (user + assistant) the LLM sees.
# 6 = the last 3 exchanges.
HISTORY_MESSAGES = 6

# Echo protection: speech that starts while Auralis is talking (or within
# this many seconds after it stops) is treated as Auralis hearing itself
# through the speakers, and is ignored.
ECHO_TAIL_S = 0.4

# Week 3 Day 4: every turn's latency breakdown is appended to this CSV
# (for p50/p95 numbers in the report). Set LATENCY_LOG=0 to turn off.
LATENCY_LOG = os.getenv("LATENCY_LOG", "1") == "1"
LATENCY_LOG_PATH = Path(__file__).resolve().parents[2] / "logs" / "latency.csv"
LATENCY_FIELDS = [
    "time", "transcript", "emotion", "voice",
    "stt_ms", "emotion_ms", "ttft_ms", "first_piece_ms", "ttfa_ms",
    "first_synth_ms", "total_ms", "audio_ms", "pieces", "tokens",
]


def _log_latency(row):
    if not LATENCY_LOG:
        return
    try:
        LATENCY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        new_file = not LATENCY_LOG_PATH.exists()
        with open(LATENCY_LOG_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LATENCY_FIELDS, extrasaction="ignore")
            if new_file:
                writer.writeheader()
            writer.writerow(row)
    except Exception as exc:
        print(f"[LATENCY] Could not write log: {type(exc).__name__}: {exc}")


class IncomingAudioTrack:
    """
    Receives microphone audio from the browser and runs the pipeline:

        frame -> VAD/segmenting (fast, every frame)
                   |
                   +-- segment finished --> background "turn" task:
                                              Whisper  -> transcript   } at the
                                              WavLM    -> emotion      } same time
                                              Llama    -> llm_token ... llm_done
                                              Piper    -> spoken reply, sentence
                                                          by sentence (Week 3 Day 3)

    DataChannel messages sent to the frontend:
        audio_frame   ~50/s   frame info for the dashboard (+ echo flag)
        vad           speaking / processing / ai_speaking / idle
        transcript    {text, stt_ms}
        emotion       {label, confidence, group, confident, ms, ready_ms}
        llm_start     reply is starting
        llm_token     {text, first, ttft_ms (first token only)}
        llm_done      {text, ttft_ms, total_ms, tokens}
        tts_start     {ttfa_ms, style, group}        first audio of the reply
        tts_sentence  {index, text, synth_ms, audio_ms}
        tts_end       {sentences, audio_ms, ttfa_ms, style, underruns}
        turn_metrics  {stt_ms, emotion_ms, ttft_ms, first_piece_ms, ttfa_ms,
                       first_synth_ms, total_ms, audio_ms, pieces, voice}
                      one latency breakdown per turn (Week 3 Day 4)
        tts_error     {error}
        echo_ignored  {}   speech heard while Auralis was talking
        llm_error     {error}
        audio_error   {frame, error}

    Timing (all measured from the moment the VAD detected end of speech):
        stt_ms   = end of speech -> transcript ready
        ready_ms = end of speech -> emotion ready
        ttft_ms  = end of speech -> first LLM token
        first_piece_ms = end of speech -> first speakable piece of text ready
        ttfa_ms  = end of speech -> first AI audio queued   (the number users feel)
        total_ms = end of speech -> full LLM reply
    """

    def __init__(self, source, data_channel=None, tts_track=None):
        self.source = source
        self.data_channel = data_channel

        self.frame_count = 0
        self.processor = AudioProcessor()
        self.stt = get_stt()  # shared model, already warmed up at startup
        self.running = True

        self.turn_lock = asyncio.Lock()
        self.turn_tasks = set()

        self.history = []
        self.user_speaking = False
        self.pending_turns = 0

        # ---- Week 3 Day 3: AI voice ----
        self.tts_track = tts_track
        self.ai_speaking = False
        self.ai_finished_at = 0.0
        self.segment_is_echo = False
        self.tts_stats = None

        if self.tts_track is not None:
            # TTSAudioTrack calls this with "started" / "finished"
            self.tts_track._on_state = self._on_tts_state

        print(
            f"[AUDIO] IncomingAudioTrack initialized "
            f"(emotion {'ON' if emotion_engine.is_ready() else 'OFF'}, "
            f"voice {'ON' if self._tts_on() else 'OFF'})"
        )

    # ==========================================================
    # DataChannel
    # ==========================================================

    def set_data_channel(self, channel):
        self.data_channel = channel
        print(f"[AUDIO] DataChannel attached: {getattr(channel, 'label', 'unknown')}")

    def send_event(self, payload):
        if self.data_channel is None:
            return
        try:
            if self.data_channel.readyState != "open":
                return
            self.data_channel.send(json.dumps(payload))
        except Exception as exc:
            print(f"[AUDIO] DataChannel send error: {type(exc).__name__}: {exc}")

    def _send_vad(self, state):
        self.send_event({"type": "vad", "vad_state": state})

    def _idle_or_speaking(self):
        if self.user_speaking:
            self._send_vad("speaking")
        elif self.ai_speaking:
            self._send_vad("ai_speaking")
        elif self.pending_turns > 0:
            self._send_vad("processing")
        else:
            self._send_vad("idle")

    # ==========================================================
    # AI voice state (Week 3 Day 3)
    # ==========================================================

    def _tts_on(self):
        return self.tts_track is not None and tts_engine.is_ready()

    def _ai_busy(self):
        """True while Auralis is talking, plus a short echo tail after."""
        if self.ai_speaking:
            return True
        return (time.perf_counter() - self.ai_finished_at) < ECHO_TAIL_S

    def _on_tts_state(self, event):
        """Called by TTSAudioTrack when playback starts / ends."""
        if event == "started":
            self.ai_speaking = True
            self._send_vad("ai_speaking")

        elif event == "finished":
            self.ai_speaking = False
            self.ai_finished_at = time.perf_counter()

            stats = self.tts_stats or {}
            underruns = self.tts_track.underruns - stats.get("underruns_start", 0)

            print(
                f"[TTS] Playback finished: {stats.get('sentences', 0)} sentences, "
                f"{stats.get('audio_ms', 0)} ms audio, underruns {underruns}"
            )
            self.send_event({
                "type": "tts_end",
                "sentences": stats.get("sentences", 0),
                "audio_ms": stats.get("audio_ms", 0),
                "ttfa_ms": stats.get("ttfa_ms"),
                "style": stats.get("style"),
                "underruns": underruns,
            })

            if self.running:
                self._idle_or_speaking()

    # ==========================================================
    # Frame loop
    # ==========================================================

    async def run(self):
        print("[AUDIO] Audio processing loop started")

        try:
            while self.running:
                try:
                    frame = await self.source.recv()
                except asyncio.CancelledError:
                    print("[AUDIO] Processing task cancelled")
                    break
                except Exception as exc:
                    print(f"[AUDIO ERROR] source.recv() failed: {type(exc).__name__}: {exc}")
                    break

                self.frame_count += 1

                if self.frame_count % DEBUG_PRINT_EVERY == 0:
                    print(
                        f"[AUDIO DEBUG] frame={self.frame_count} "
                        f"samples={getattr(frame, 'samples', None)} "
                        f"sample_rate={getattr(frame, 'sample_rate', None)}"
                    )

                try:
                    metadata = await asyncio.to_thread(self.processor.process_frame, frame)
                except Exception as exc:
                    print(f"[AUDIO ERROR] Processing frame={self.frame_count}: "
                          f"{type(exc).__name__}: {exc}")
                    self.send_event({
                        "type": "audio_error",
                        "frame": self.frame_count,
                        "error": str(exc),
                    })
                    continue

                # Decide once, at the start of a segment, whether it is echo
                if metadata["segment_started"]:
                    self.segment_is_echo = self._ai_busy()

                is_echo = self.segment_is_echo

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
                    "echo": is_echo,
                })

                if metadata["segment_started"]:
                    if is_echo:
                        print(f"[ECHO] Speech while Auralis is talking "
                              f"(frame={self.frame_count}) - will be ignored")
                    else:
                        self.user_speaking = True
                        print(f"[VAD] Speech started (frame={self.frame_count})")
                        self._send_vad("speaking")

                if metadata["segment_finished"]:
                    self.segment_is_echo = False

                    if is_echo:
                        print("[ECHO] Segment ignored (Auralis was speaking)")
                        self.send_event({"type": "echo_ignored"})
                        continue

                    self.user_speaking = False
                    print(f"[VAD] Speech finished (frame={self.frame_count}, "
                          f"segment_frames={metadata['segment_frame_count']})")

                    if metadata.get("segment_skipped") or not metadata["segment"]:
                        self._idle_or_speaking()
                        continue

                    self.pending_turns += 1
                    self._send_vad("processing")

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
            print(f"[AUDIO] Processing loop stopped after {self.frame_count} frames")

    # ==========================================================
    # One conversation turn: (STT + emotion) -> LLM -> voice
    # ==========================================================

    def _transcribe_timed(self, segment, speech_end_time):
        text = self.stt.transcribe(segment)
        return text, round((time.perf_counter() - speech_end_time) * 1000)

    def _detect_emotion(self, segment, speech_end_time):
        if not emotion_engine.is_ready():
            return None
        try:
            audio = self.stt._frames_to_audio(segment)
            result = emotion_engine.predict(audio)
            if result is not None:
                result["ready_ms"] = round((time.perf_counter() - speech_end_time) * 1000)
            return result
        except Exception as exc:
            print(f"[EMOTION ERROR] {type(exc).__name__}: {exc}")
            return None

    async def _handle_turn(self, segment, speech_end_time):
        async with self.turn_lock:
            try:
                # Whisper and WavLM run at the same time in two threads
                (transcript, stt_ms), emotion = await asyncio.gather(
                    asyncio.to_thread(self._transcribe_timed, segment, speech_end_time),
                    asyncio.to_thread(self._detect_emotion, segment, speech_end_time),
                )

                if not transcript:
                    print(f"[STT] No transcript ({stt_ms} ms)")
                    return

                print(f"[STT] Transcript ({stt_ms} ms after speech end): {transcript}")
                self.send_event({"type": "transcript", "text": transcript, "stt_ms": stt_ms})

                if emotion is not None:
                    print(
                        f"[EMOTION] {emotion['label']} ({emotion['confidence']:.2f}) "
                        f"-> {emotion['group']} | model {emotion['ms']} ms, "
                        f"ready {emotion['ready_ms']} ms after speech end"
                    )
                    self.send_event({
                        "type": "emotion",
                        "label": emotion["label"],
                        "confidence": emotion["confidence"],
                        "group": emotion["group"],
                        "confident": emotion["confident"],
                        "ms": emotion["ms"],
                        "ready_ms": emotion["ready_ms"],
                    })

                reply = await self._stream_llm_reply(transcript, speech_end_time, emotion)

                # ---- Week 3 Day 4: one latency breakdown per turn ----
                self._report_turn_metrics(transcript, stt_ms, emotion, reply)

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

    async def _stream_llm_reply(self, transcript, speech_end_time, emotion=None):
        use_emotion = emotion is not None and emotion["confident"]

        messages = ollama_client.build_messages(
            self.history,
            transcript,
            emotion=emotion["label"] if use_emotion else None,
            hint=emotion["hint"] if use_emotion else None,
        )

        # ---- Week 3 Day 3: speak the reply while it is being written ----
        voice_group = emotion["group"] if use_emotion else "okay"
        speak = self._tts_on()
        splitter = tts_engine.SentenceSplitter()
        sentence_queue = asyncio.Queue() if speak else None
        speak_task = None

        if speak:
            speak_task = asyncio.create_task(
                self._speak_sentences(sentence_queue, voice_group, speech_end_time)
            )

        self.send_event({"type": "llm_start"})

        reply_parts = []
        ttft_ms = None
        llm_failed = False

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
                    self.send_event({"type": "llm_token", "text": token, "first": False})

                reply_parts.append(token)

                # A sentence is complete -> start speaking it right away
                if speak:
                    for sentence in splitter.feed(token):
                        sentence_queue.put_nowait((sentence, time.perf_counter()))

        except RuntimeError as exc:
            llm_failed = True
            print(f"[LLM ERROR] {exc}")
            self.send_event({"type": "llm_error", "error": str(exc)})

        except asyncio.CancelledError:
            if speak_task is not None:
                speak_task.cancel()
            raise

        finally:
            if speak:
                if not llm_failed:
                    for sentence in splitter.flush():
                        sentence_queue.put_nowait((sentence, time.perf_counter()))
                sentence_queue.put_nowait(None)   # no more sentences

        if llm_failed:
            if speak_task is not None:
                await speak_task
            return None

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

        if reply:
            self.history.append({"role": "user", "content": transcript})
            self.history.append({"role": "assistant", "content": reply})
            self.history = self.history[-HISTORY_MESSAGES:]

        # Keep the turn open until every sentence has been synthesized
        # (playback itself continues in TTSAudioTrack).
        voice = {}
        if speak_task is not None:
            voice = await speak_task or {}

        return {
            "ttft_ms": ttft_ms,
            "total_ms": total_ms,
            "tokens": len(reply_parts),
            "first_piece_ms": voice.get("first_piece_ms"),
            "ttfa_ms": voice.get("ttfa_ms"),
            "first_synth_ms": voice.get("first_synth_ms"),
            "audio_ms": voice.get("audio_ms", 0),
            "pieces": voice.get("sentences", 0),
            "voice": voice.get("style"),
        }

    async def _speak_sentences(self, sentence_queue, group, speech_end_time):
        """Turn queued sentences into audio and push them to the browser."""
        sample_rate = tts_engine.sample_rate()
        style = tts_engine.style_name(group)

        stats = {
            "sentences": 0,
            "audio_ms": 0,
            "ttfa_ms": None,
            "first_piece_ms": None,
            "first_synth_ms": None,
            "style": style,
            "underruns_start": self.tts_track.underruns,
        }
        self.tts_stats = stats

        try:
            while True:
                item = await sentence_queue.get()
                if item is None:
                    break

                sentence, ready_at = item
                if stats["first_piece_ms"] is None:
                    stats["first_piece_ms"] = round((ready_at - speech_end_time) * 1000)

                start = time.perf_counter()
                audio = await asyncio.to_thread(tts_engine.synthesize, sentence, group)
                synth_ms = round((time.perf_counter() - start) * 1000)

                if audio.size == 0:
                    continue

                audio_ms = round(len(audio) / sample_rate * 1000)

                if stats["ttfa_ms"] is None:
                    stats["ttfa_ms"] = round((time.perf_counter() - speech_end_time) * 1000)
                    stats["first_synth_ms"] = synth_ms
                    self.ai_speaking = True
                    print(f"[TTS] First audio: TTFA = {stats['ttfa_ms']} ms "
                          f"(voice '{style}')")
                    self.send_event({
                        "type": "tts_start",
                        "ttfa_ms": stats["ttfa_ms"],
                        "style": style,
                        "group": group,
                    })
                    self._send_vad("ai_speaking")

                self.tts_track.push_chunk(audio, sample_rate)

                stats["sentences"] += 1
                stats["audio_ms"] += audio_ms

                print(f"[TTS] Sentence {stats['sentences']} "
                      f"({synth_ms} ms synth, {audio_ms} ms audio): {sentence}")
                self.send_event({
                    "type": "tts_sentence",
                    "index": stats["sentences"],
                    "text": sentence,
                    "synth_ms": synth_ms,
                    "audio_ms": audio_ms,
                })

        except asyncio.CancelledError:
            raise

        except Exception as exc:
            print(f"[TTS ERROR] {type(exc).__name__}: {exc}")
            self.send_event({"type": "tts_error", "error": str(exc)})

        finally:
            if stats["sentences"] > 0:
                self.tts_track.end_clip()

        return stats

    # ==========================================================
    # Latency breakdown (Week 3 Day 4)
    # ==========================================================

    def _report_turn_metrics(self, transcript, stt_ms, emotion, reply):
        if reply is None:
            return

        metrics = {
            "stt_ms": stt_ms,
            "emotion_ms": emotion["ready_ms"] if emotion else None,
            "ttft_ms": reply["ttft_ms"],
            "first_piece_ms": reply["first_piece_ms"],
            "ttfa_ms": reply["ttfa_ms"],
            "first_synth_ms": reply["first_synth_ms"],
            "total_ms": reply["total_ms"],
            "audio_ms": reply["audio_ms"],
            "pieces": reply["pieces"],
            "tokens": reply["tokens"],
            "voice": reply["voice"],
        }

        def fmt(key):
            value = metrics[key]
            return "-" if value is None else f"{value}"

        print(
            f"[LATENCY] stt {fmt('stt_ms')} | emotion {fmt('emotion_ms')} | "
            f"first token {fmt('ttft_ms')} | first piece {fmt('first_piece_ms')} | "
            f"first audio {fmt('ttfa_ms')} | full text {fmt('total_ms')} ms"
        )

        self.send_event({"type": "turn_metrics", **metrics})

        _log_latency({
            "time": datetime.now().isoformat(timespec="seconds"),
            "transcript": transcript,
            "emotion": emotion["label"] if emotion else "",
            **metrics,
        })

    # ==========================================================
    # Stop
    # ==========================================================

    def _cancel_turns(self):
        for task in list(self.turn_tasks):
            task.cancel()

    def stop(self):
        if self.running:
            print("[AUDIO] Stop requested")
        self.running = False
        self._cancel_turns()