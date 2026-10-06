import time
from collections import deque

from app.audio.metrics.latency import LatencyTracker
from app.audio.segments.speech_segment import SpeechSegment
from app.audio.vad.detector import VoiceActivityDetector


# All counts below are WebRTC frames (20 ms each).

# Extra frames kept AFTER the VAD says speech ended.
# The VAD already waits ~700 ms of silence (SILENCE_CHUNKS_TO_END=22),
# so this only needs to be short. 5 frames = 100 ms.
# (Was 25 frames = 500 ms, which added to the VAD's 700 ms -> 1.2 s wait.)
SILENCE_HANGOVER_FRAMES = 5

# Frames kept BEFORE speech starts, so the first syllable is not cut.
# 15 frames = 300 ms.
PRE_ROLL_FRAMES = 15

# Minimum amount of REAL voice in a segment (frames where the VAD
# probability was above threshold). 10 frames = 200 ms.
# Shorter segments (coughs, clicks, desk taps) are skipped so Whisper
# does not hallucinate text like "Thank you."
MIN_VOICED_FRAMES = 10

# Force-finish very long segments so Whisper never gets a huge job.
# 750 frames = 15 s.
MAX_SEGMENT_FRAMES = 750


class AudioProcessor:
    """
    Processes incoming WebRTC audio.

    Pipeline:
        WebRTC audio frame
                ↓
             Silero VAD
                ↓
          SpeechSegment
     (pre-roll + short silence hangover)
                ↓
          Speech ends
                ↓
        completed audio segment

    STT is intentionally NOT performed here.

    STT and later ML/LLM processing happen asynchronously
    outside the frame-processing path so incoming audio keeps
    flowing without being blocked by Whisper.
    """

    def __init__(self):
        self.frame_count = 0

        self.vad = VoiceActivityDetector()
        self.latency_tracker = LatencyTracker()
        self.speech_segment = SpeechSegment()

        self.pre_roll = deque(maxlen=PRE_ROLL_FRAMES)

        self._reset_segment_counters()

        print(
            "[AUDIO] AudioProcessor initialized "
            f"(hangover={SILENCE_HANGOVER_FRAMES} frames, "
            f"pre_roll={PRE_ROLL_FRAMES} frames, "
            f"min_voiced={MIN_VOICED_FRAMES} frames, "
            f"max_segment={MAX_SEGMENT_FRAMES} frames)"
        )

    # ==========================================================
    # Helpers
    # ==========================================================

    def _reset_segment_counters(self):
        # Frames since the VAD said speech ended (hangover counter)
        self.silence_frames = 0

        # All frames in the current segment
        self.segment_frames = 0

        # Frames that actually contained voice
        # (VAD probability >= threshold). The VAD keeps is_speech=True
        # for ~700 ms of trailing silence, so counting is_speech frames
        # would always pass the minimum-length check.
        self.voiced_frames = 0

        # time.perf_counter() when the VAD reported end of speech.
        # Used downstream to measure STT latency and TTFT.
        self.speech_end_time = None

    def _is_voiced(self):
        return self.vad.last_probability >= self.vad.SPEECH_THRESHOLD

    def _finish_segment(self, reason):
        """Close the current segment and decide if it is worth transcribing."""
        segment = self.speech_segment.finish()

        if self.speech_end_time is None:
            # Forced finish (max length): speech is still going on
            self.speech_end_time = time.perf_counter()

        speech_end_time = self.speech_end_time
        voiced_frames = self.voiced_frames
        skipped = False

        print(
            "[AUDIO] Speech segment finished "
            f"(reason={reason}, frames={len(segment)}, "
            f"voiced_frames={voiced_frames})"
        )

        if voiced_frames < MIN_VOICED_FRAMES:
            print(
                "[AUDIO] Segment too short - "
                "skipping downstream processing"
            )
            segment = []
            skipped = True

        self._reset_segment_counters()

        return segment, skipped, speech_end_time

    # ==========================================================
    # Process one WebRTC frame
    # ==========================================================

    def process_frame(self, frame):
        """
        Process one incoming audio frame.

        This function must stay fast.

        It performs:
            - VAD
            - speech segment tracking
            - speech start detection
            - speech end detection

        It does NOT perform:
            - Whisper
            - emotion inference
            - LLM inference
        """

        self.frame_count += 1

        start_time = self.latency_tracker.start()

        is_speech = self.vad.process(frame)
        voiced = self._is_voiced()

        segment_started = False
        segment_finished = False
        segment_skipped = False
        speech_end_time = None
        segment = []

        if is_speech:

            # Speech (re)started: cancel any running hangover
            self.silence_frames = 0
            self.speech_end_time = None

            if not self.speech_segment.active:

                buffered = list(self.pre_roll)
                self.pre_roll.clear()

                if buffered:
                    self.speech_segment.start(buffered[0])

                    for buffered_frame in buffered[1:]:
                        self.speech_segment.add(buffered_frame)

                    self.speech_segment.add(frame)

                else:
                    self.speech_segment.start(frame)

                self.segment_frames = len(buffered) + 1
                self.voiced_frames = 1 if voiced else 0
                segment_started = True

                print(
                    "[AUDIO] Speech segment started "
                    f"(pre-roll frames={len(buffered)})"
                )

            else:

                self.speech_segment.add(frame)
                self.segment_frames += 1

                if voiced:
                    self.voiced_frames += 1

                # Safety: never let one segment grow forever
                if self.segment_frames >= MAX_SEGMENT_FRAMES:
                    segment, segment_skipped, speech_end_time = (
                        self._finish_segment("max_length")
                    )
                    segment_finished = True

        elif self.speech_segment.active:

            # The VAD has just said speech ended (or we are in the
            # hangover). Remember WHEN speech ended, once.
            if self.speech_end_time is None:
                self.speech_end_time = time.perf_counter()

            # Keep a few hangover frames so the last word is not cut off
            self.speech_segment.add(frame)
            self.segment_frames += 1
            self.silence_frames += 1

            if self.silence_frames >= SILENCE_HANGOVER_FRAMES:
                segment, segment_skipped, speech_end_time = (
                    self._finish_segment("silence")
                )
                segment_finished = True

        else:

            self.pre_roll.append(frame)

        latency_ms = self.latency_tracker.stop(start_time)

        return {
            "frame_count": self.frame_count,
            "sample_rate": getattr(frame, "sample_rate", None),
            "samples": getattr(frame, "samples", None),
            "pts": getattr(frame, "pts", None),
            "is_speech": is_speech,
            "segment_started": segment_started,
            "segment_finished": segment_finished,
            # True when the segment was too short and was dropped.
            # audio_track.py should send VAD "idle"/"listening" in this case
            # instead of "processing", because no transcript will come.
            "segment_skipped": segment_skipped,
            "segment": segment,
            "segment_frame_count": len(segment),
            # time.perf_counter() at end of speech (None unless finished).
            # Use it to measure STT latency / TTFT:
            #   (time.perf_counter() - speech_end_time) * 1000
            "speech_end_time": speech_end_time,
            "latency_ms": latency_ms,
        }

    def reset(self):

        print("[AUDIO] Resetting AudioProcessor")

        self.frame_count = 0

        self.vad.reset()

        self.speech_segment = SpeechSegment()

        self.pre_roll.clear()

        self._reset_segment_counters()

        print("[AUDIO] AudioProcessor reset")