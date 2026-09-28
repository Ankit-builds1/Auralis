from collections import deque

from app.audio.metrics.latency import LatencyTracker
from app.audio.segments.speech_segment import SpeechSegment
from app.audio.vad.detector import VoiceActivityDetector
from app.stt.transcriber import SpeechToText


# =========================================================
# SEGMENTATION SETTINGS (each WebRTC frame is 20 ms)
# =========================================================

# 25 x 20 ms = 500 ms of continuous silence ends a segment.
# Short pauses between words stay inside the same segment.
SILENCE_HANGOVER_FRAMES = 25

# Keep the last 15 x 20 ms = 300 ms of audio from before speech
# is detected, so the first syllable is not clipped.
PRE_ROLL_FRAMES = 15

# Segments with less than 10 x 20 ms = 200 ms of real speech
# (coughs, clicks) are not sent to Whisper.
MIN_SPEECH_FRAMES = 10


class AudioProcessor:
    """
    Processes incoming WebRTC audio.

    Pipeline:

        WebRTC audio frame
                ↓
             Silero VAD
                ↓
          SpeechSegment
     (pre-roll + silence hangover)
                ↓
          Speech ends
                ↓
              Whisper
    """

    def __init__(self):
        self.frame_count = 0

        self.vad = VoiceActivityDetector()

        self.latency_tracker = LatencyTracker()

        self.speech_segment = SpeechSegment()

        self.stt = SpeechToText(
            model_size="base",
            device="cpu",
            compute_type="int8",
        )

        # Rolling buffer of recent non-speech frames (pre-roll)
        self.pre_roll = deque(maxlen=PRE_ROLL_FRAMES)

        # Consecutive silent frames inside an active segment
        self.silence_frames = 0

        # Frames actually detected as speech in the current segment
        self.speech_frames = 0

        print("[AUDIO] AudioProcessor initialized")

    def process_frame(self, frame):

        self.frame_count += 1

        start_time = self.latency_tracker.start()

        # =====================================================
        # VAD
        # =====================================================

        is_speech = self.vad.process(frame)

        segment_started = False
        segment_finished = False

        segment = []
        transcript = ""

        # =====================================================
        # DEBUG
        # =====================================================

        print(
            f"[AUDIO] frame={self.frame_count} "
            f"speech={is_speech} "
            f"segment_active={self.speech_segment.active} "
            f"silence_frames={self.silence_frames}"
        )

        # =====================================================
        # SPEECH
        # =====================================================

        if is_speech:

            self.silence_frames = 0

            if not self.speech_segment.active:

                # Prepend the buffered audio so the start
                # of the first word is not lost.
                buffered = list(self.pre_roll)
                self.pre_roll.clear()

                if buffered:
                    self.speech_segment.start(buffered[0])

                    for buffered_frame in buffered[1:]:
                        self.speech_segment.add(buffered_frame)

                    self.speech_segment.add(frame)

                else:
                    self.speech_segment.start(frame)

                self.speech_frames = 1

                segment_started = True

                print(
                    f"[AUDIO] Speech segment started "
                    f"(pre-roll frames={len(buffered)})"
                )

            else:

                self.speech_segment.add(frame)

                self.speech_frames += 1

        # =====================================================
        # SILENCE INSIDE AN ACTIVE SEGMENT (HANGOVER)
        # =====================================================

        elif self.speech_segment.active:

            # Keep short pauses inside the segment.
            # Only end it after SILENCE_HANGOVER_FRAMES of silence.
            self.speech_segment.add(frame)

            self.silence_frames += 1

            if self.silence_frames >= SILENCE_HANGOVER_FRAMES:

                segment = self.speech_segment.finish()

                segment_finished = True

                self.silence_frames = 0

                print(
                    f"[AUDIO] Speech segment finished "
                    f"frames={len(segment)} "
                    f"speech_frames={self.speech_frames}"
                )

                # =============================================
                # TRANSCRIPTION
                # =============================================

                if self.speech_frames >= MIN_SPEECH_FRAMES:

                    try:

                        print(
                            "[STT] Starting transcription..."
                        )

                        transcript = self.stt.transcribe(
                            segment
                        )

                        print(
                            f"[STT] Transcript: {transcript}"
                        )

                    except Exception as exc:

                        print(
                            "[STT ERROR] "
                            f"{type(exc).__name__}: {exc}"
                        )

                else:

                    print(
                        "[STT] Segment too short - skipping"
                    )

                self.speech_frames = 0

        # =====================================================
        # SILENCE, NO ACTIVE SEGMENT
        # =====================================================

        else:

            # Keep a rolling buffer for the next segment's pre-roll
            self.pre_roll.append(frame)

        # =====================================================
        # LATENCY
        # =====================================================

        latency_ms = self.latency_tracker.stop(
            start_time
        )

        # =====================================================
        # RESULT
        # =====================================================

        return {
            "frame_count": self.frame_count,

            "sample_rate": getattr(
                frame,
                "sample_rate",
                None,
            ),

            "samples": getattr(
                frame,
                "samples",
                None,
            ),

            "pts": getattr(
                frame,
                "pts",
                None,
            ),

            "is_speech": is_speech,

            "segment_started": segment_started,

            "segment_finished": segment_finished,

            "segment": segment,

            "segment_frame_count": len(segment),

            "transcript": transcript,

            "latency_ms": latency_ms,
        }

    def reset(self):

        print("[AUDIO] Resetting AudioProcessor")

        self.frame_count = 0

        self.vad.reset()

        self.speech_segment = SpeechSegment()

        self.pre_roll.clear()

        self.silence_frames = 0

        self.speech_frames = 0

        print("[AUDIO] AudioProcessor reset")