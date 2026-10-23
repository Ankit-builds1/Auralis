import asyncio
import csv
import json
import os
import statistics
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from app.audio.processor import AudioProcessor
from app.llm import ollama_client
from app.ml_clients import emotion_engine, text_emotion
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

# Week 4 Day 2: BARGE-IN (full duplex). If the user keeps talking while
# Auralis is speaking (or writing its reply), Auralis stops at once and
# answers the NEW question instead.
#   BARGE_IN=0                     turn it off (old behaviour: ignore as echo)
#   BARGE_IN_MIN_VOICED_FRAMES=20  real-voice frames needed (20 x 20 ms = 400 ms)
#                                  higher = fewer false stops from speaker echo
# Headphones make barge-in most reliable (no Auralis voice in the mic).
BARGE_IN = os.getenv("BARGE_IN", "1") == "1"
BARGE_IN_MIN_VOICED_FRAMES = int(os.getenv("BARGE_IN_MIN_VOICED_FRAMES", "20"))

# Week 4 (ML W4D1 finding, ml/results/week4_bargein): Silero VAD also calls
# Auralis's leftover speaker echo "speech", so counting frames alone cannot
# tell echo from the user. The echo is QUIETER. So, once Auralis has heard
# the user's normal speaking level, a frame only counts for barge-in when it
# is at most BARGE_IN_GATE_DB below that level (relative gate, works on any
# mic). With the gate, fewer frames are needed (10 x 20 ms = 200 ms of loud
# voice; ML eval: 98 % detection, 0 % false triggers on simulated echo).
#   BARGE_IN_GATE_DB=5         dB below the user's level still counted (0 = off)
#   BARGE_IN_GATED_FRAMES=10   loud voiced frames needed when the gate is on
# Until ~1 s of the user's speech has been heard, the plain rule above is used.
BARGE_IN_GATE_DB = float(os.getenv("BARGE_IN_GATE_DB", "5"))
BARGE_IN_GATED_FRAMES = int(os.getenv("BARGE_IN_GATED_FRAMES", "10"))
# Week 4: live tests showed the voice model says "disgust" far too often on
# our voices (e.g. "Tell me a bank joke" -> disgust 84 %). When ONLY the voice
# says disgust, it must be at least this sure, otherwise it is not used.
VOICE_DISGUST_MIN = float(os.getenv("VOICE_DISGUST_MIN", "0.9"))

USER_LEVEL_MIN_FRAMES = 50          # voiced frames needed to trust the level
USER_LEVEL_HISTORY = 1500           # ~30 s of the user's voiced frames

# Week 3 Day 4: every turn's latency breakdown is appended to this CSV
# (for p50/p95 numbers in the report). Set LATENCY_LOG=0 to turn off.
LATENCY_LOG = os.getenv("LATENCY_LOG", "1") == "1"
LATENCY_LOG_PATH = Path(__file__).resolve().parents[2] / "logs" / "latency.csv"
LATENCY_FIELDS = [
    "time", "transcript", "emotion", "voice_label", "text_label", "source",
    "voice", "text_ms",
    "stt_ms", "emotion_ms", "ttft_ms", "first_piece_ms", "ttfa_ms",
    "first_synth_ms", "total_ms", "audio_ms", "pieces", "tokens",
]


def _log_latency(row):
    if not LATENCY_LOG:
        return
    try:
        LATENCY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        new_file = not LATENCY_LOG_PATH.exists()

        # Week 4 fix: a log started by an older version has fewer columns,
        # and new rows would land under the wrong headers. Keep the old file
        # aside and start a fresh one with the current header.
        if not new_file:
            with open(LATENCY_LOG_PATH, newline="", encoding="utf-8") as f:
                header = next(csv.reader(f), [])
            if header != LATENCY_FIELDS:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                old = LATENCY_LOG_PATH.with_name(f"latency_old_{stamp}.csv")
                LATENCY_LOG_PATH.rename(old)
                print(f"[LATENCY] Old log format -> moved to {old.name}, new log started")
                new_file = True
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
        emotion       {label, confidence, group, confident, ms, ready_ms,
                       source, voice_label, voice_confidence,
                       text_label, text_confidence, text_ms}
                      label/confidence/group = FINAL (voice + words fused,
                      Week 3 Day 5); source = "voice" or "words" (who decided)
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
        interrupted   {reason, stop_ms, spoken_ms, dropped_ms,
                       gate_db, user_level_db}                  Week 4
                      reason "barge_in" = user talked over Auralis,
                             "manual"   = STOP AURALIS button
                      stop_ms = user speech start (or click) -> Auralis stopped

    DataChannel messages received from the frontend (server.py):
        interrupt     {}   STOP AURALIS button -> request_interrupt("manual")
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

        # ---- Week 4 Day 2: barge-in ----
        self.reply_active = False      # LLM reply (text and/or voice) in progress
        self.ai_started_at = 0.0       # when Auralis's current audio started
        self.in_segment = False
        self.segment_started_at = 0.0
        self.segment_barged = False

        # ---- Week 4: relative loudness gate ----
        self.user_levels = deque(maxlen=USER_LEVEL_HISTORY)  # dBFS of user voiced frames
        self.user_level_db = None        # median of user_levels (once trusted)
        self.segment_loud_frames = 0     # voiced frames above the gate, this segment

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
            self.ai_started_at = time.perf_counter()
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
    # Week 4 Day 2: interruption (barge-in)
    # ==========================================================

    def _interrupt(self, reason, since=None):
        """
        Stop the current reply right now: cancel the LLM + TTS task, drop the
        audio that is still queued, and tell the frontend.
        Returns False when there was nothing to stop.
        """
        if not (self.ai_speaking or self.reply_active):
            return False

        now = time.perf_counter()
        spoken_ms = round((now - self.ai_started_at) * 1000) if self.ai_speaking else 0

        # Reply already fully written but only partly spoken: the history
        # should show the user did not hear all of it
        if (not self.reply_active and self.history
                and self.history[-1]["role"] == "assistant"
                and not self.history[-1]["content"].endswith("[interrupted by the user]")):
            self.history[-1] = {
                "role": "assistant",
                "content": self.history[-1]["content"] + " ... [interrupted by the user]",
            }

        # Cancels the running turn (Ollama stream + Piper) and any stale
        # turns still waiting for the lock
        self._cancel_turns()

        dropped_ms = self.tts_track.clear() if self.tts_track is not None else 0

        self.reply_active = False
        self.ai_speaking = False
        self.ai_finished_at = now

        stop_ms = round((now - since) * 1000) if since else 0

        print(
            f"[BARGE-IN] Auralis interrupted ({reason}): stopped {stop_ms} ms after "
            f"{'speech start' if reason == 'barge_in' else 'click'}, "
            f"{spoken_ms} ms spoken, {dropped_ms} ms of queued audio dropped"
        )
        self.send_event({
            "type": "interrupted",
            "reason": reason,
            "stop_ms": stop_ms,
            "spoken_ms": spoken_ms,
            "dropped_ms": dropped_ms,
            "gate_db": self._gate_db(),
            "user_level_db": self.user_level_db,
        })
        return True

    def request_interrupt(self, reason="manual"):
        """Called from server.py when the browser sends {"type": "interrupt"}."""
        if not self._interrupt(reason, since=time.perf_counter()):
            print(f"[BARGE-IN] Interrupt ({reason}) ignored - Auralis was not replying")
            return
        if self.running:
            self._idle_or_speaking()

    def _gate_db(self):
        """Level a frame must reach to count for barge-in, or None (gate off)."""
        if BARGE_IN_GATE_DB <= 0 or self.user_level_db is None:
            return None
        return self.user_level_db - BARGE_IN_GATE_DB

    def _track_levels(self, metadata):
        """
        Every frame of an open segment:
          - learn the user's normal level from their own speech (only while
            Auralis is silent, so no echo gets mixed in)
          - count loud voiced frames for the barge-in gate
        """
        if not self.in_segment or not metadata.get("voiced"):
            return
        level = metadata.get("level_db")
        if level is None:
            return

        if not self.segment_is_echo and not self.ai_speaking:
            self.user_levels.append(level)
            if len(self.user_levels) >= USER_LEVEL_MIN_FRAMES and len(self.user_levels) % 25 == 0:
                self.user_level_db = round(statistics.median(self.user_levels), 1)

        gate = self._gate_db()
        if gate is not None and level >= gate:
            self.segment_loud_frames += 1

    def _check_barge_in(self, metadata):
        """
        Runs on every frame while a speech segment is open. The segment is a
        barge-in candidate when it started while Auralis was talking (echo
        guard) or while a reply was being written. Once enough REAL voice
        frames arrive, it is the user talking over Auralis, not echo.
        """
        if not BARGE_IN or self.segment_barged or not self.in_segment:
            return

        if not (self.segment_is_echo or self.reply_active):
            return

        gate = self._gate_db()
        if gate is not None:
            # Week 4: loud voiced frames only (echo is quieter than the user)
            if self.segment_loud_frames < BARGE_IN_GATED_FRAMES:
                return
            rule = (f"{self.segment_loud_frames} frames >= {gate:.1f} dBFS "
                    f"(user level {self.user_level_db:.1f} dBFS)")
        else:
            if metadata.get("voiced_frames", 0) < BARGE_IN_MIN_VOICED_FRAMES:
                return
            rule = f"{metadata.get('voiced_frames', 0)} voiced frames (no gate yet)"

        self.segment_barged = True
        print(f"[BARGE-IN] Trigger: {rule}")
        self._interrupt("barge_in", since=self.segment_started_at)

        # The user is really talking: this segment becomes a normal turn
        # (transcribed + answered when it ends), even if Auralis had
        # already finished on its own during the echo tail.
        if self.segment_is_echo:
            self.segment_is_echo = False
            print(f"[BARGE-IN] Speech while Auralis was talking is the user "
                  f"(frame={self.frame_count}) - will be answered")
        self.user_speaking = True
        self._send_vad("speaking")

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
                    self.in_segment = True
                    self.segment_started_at = time.perf_counter()
                    self.segment_barged = False
                    self.segment_loud_frames = 0

                # Week 4: learn the user's level + count loud frames
                self._track_levels(metadata)

                # Week 4 Day 2: user talking over Auralis -> stop Auralis
                self._check_barge_in(metadata)

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
                    self.in_segment = False

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

    def _fuse_emotion(self, transcript, emotion, speech_end_time):
        """
        Week 3 Day 5: combine the voice emotion with the words.
        The words decide only when the text model is clearly sure of a
        non-neutral emotion; otherwise the voice result is kept.
        """
        emotion["voice_label"] = emotion["label"]
        emotion["voice_confidence"] = emotion["confidence"]
        emotion["text_label"] = None
        emotion["text_confidence"] = None
        emotion["text_ms"] = None
        emotion["source"] = "voice"

        try:
            text = text_emotion.predict(transcript)
        except Exception as exc:
            print(f"[TEXT ERROR] {type(exc).__name__}: {exc}")
            text = None

        if text is None or not emotion.get("probs"):
            return self._damp_voice_disgust(emotion)

        label, confidence, source = text_emotion.fuse(emotion["probs"], text["probs"])

        if source == "words":
            confident = True          # text model was >= 0.6 sure
        else:
            confident = emotion["confident"]   # same voice gate as before

        group = emotion_engine.GROUPS.get(label, "okay") if confident else "okay"

        emotion.update(
            label=label,
            confidence=confidence,
            confident=confident,
            group=group,
            hint=emotion_engine.LLM_HINTS[group],
            source=source,
            text_label=text["label"],
            text_confidence=text["confidence"],
            text_ms=text["ms"],
            ready_ms=round((time.perf_counter() - speech_end_time) * 1000),
        )
        return self._damp_voice_disgust(emotion)

    def _damp_voice_disgust(self, emotion):
        """Week 4: a voice-only 'disgust' below VOICE_DISGUST_MIN is not trusted."""
        if (emotion.get("source") == "voice" and emotion.get("label") == "disgust"
                and emotion.get("voice_confidence", emotion["confidence"]) < VOICE_DISGUST_MIN
                and emotion.get("confident")):
            emotion.update(confident=False, group="okay",
                           hint=emotion_engine.LLM_HINTS["okay"])
            print(f"[EMOTION] voice-only disgust ({emotion.get('voice_confidence', 0):.2f}) "
                  f"< {VOICE_DISGUST_MIN} -> not used")
        return emotion

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
                    # ---- Week 3 Day 5: voice + words ----
                    emotion = await asyncio.to_thread(
                        self._fuse_emotion, transcript, emotion, speech_end_time
                    )
                    words = (f"{emotion['text_label']} ({emotion['text_confidence']:.2f}, "
                             f"{emotion['text_ms']} ms)" if emotion["text_label"] else "-")
                    print(
                        f"[EMOTION] {emotion['label']} ({emotion['confidence']:.2f}) "
                        f"-> {emotion['group']} | by {emotion['source']} | "
                        f"voice {emotion['voice_label']} ({emotion['voice_confidence']:.2f}), "
                        f"words {words} | ready {emotion['ready_ms']} ms after speech end"
                    )
                    self.send_event({
                        "type": "emotion",
                        "label": emotion["label"],
                        "confidence": emotion["confidence"],
                        "group": emotion["group"],
                        "confident": emotion["confident"],
                        "source": emotion["source"],
                        "voice_label": emotion["voice_label"],
                        "voice_confidence": emotion["voice_confidence"],
                        "text_label": emotion["text_label"],
                        "text_confidence": emotion["text_confidence"],
                        "ms": emotion["ms"],
                        "text_ms": emotion["text_ms"],
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
                self.reply_active = False
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

        self.reply_active = True
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
            # Week 4 Day 2: keep the cut-off exchange so Auralis knows what
            # it was saying when the user interrupted
            partial = "".join(reply_parts).strip()
            if partial:
                self.history.append({"role": "user", "content": transcript})
                self.history.append({
                    "role": "assistant",
                    "content": partial + " ... [interrupted by the user]",
                })
                self.history = self.history[-HISTORY_MESSAGES:]
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
            "voice_label": emotion.get("voice_label", "") if emotion else "",
            "text_label": (emotion.get("text_label") or "") if emotion else "",
            "source": emotion.get("source", "") if emotion else "",
            "text_ms": emotion.get("text_ms") if emotion else None,
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