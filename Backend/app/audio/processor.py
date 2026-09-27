from app.audio.metrics.latency import LatencyTracker
from app.audio.segments.speech_segment import SpeechSegment
from app.audio.vad.detector import VoiceActivityDetector
from app.stt.transcriber import SpeechToText


class AudioProcessor:
    """
    Processes incoming WebRTC audio.

    Pipeline:

        WebRTC audio frame
                ↓
             Silero VAD
                ↓
          SpeechSegment
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
            f"segment_active={self.speech_segment.active}"
        )

        # =====================================================
        # SPEECH
        # =====================================================

        if is_speech:

            if not self.speech_segment.active:

                self.speech_segment.start(frame)

                segment_started = True

                print(
                    "[AUDIO] Speech segment started"
                )

            else:

                self.speech_segment.add(frame)

        # =====================================================
        # SILENCE / SPEECH ENDED
        # =====================================================

        elif self.speech_segment.active:

            segment = self.speech_segment.finish()

            segment_finished = True

            print(
                f"[AUDIO] Speech segment finished "
                f"frames={len(segment)}"
            )

            # =================================================
            # TRANSCRIPTION
            # =================================================

            if len(segment) > 0:

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
                    "[STT] Empty segment - skipping"
                )

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

        print("[AUDIO] AudioProcessor reset")