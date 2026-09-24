from app.audio.metrics.latency import LatencyTracker
from app.audio.segments.speech_segment import SpeechSegment
from app.audio.vad.detector import VoiceActivityDetector


class AudioProcessor:
    """
    Handles incoming audio frames.

    Day 8:
    - Receive audio frames
    - Run voice activity detection
    - Track processing latency
    - Collect consecutive speech frames
    - Detect speech segment start and end
    """

    def __init__(self):
        self.frame_count = 0
        self.vad = VoiceActivityDetector()
        self.latency_tracker = LatencyTracker()
        self.speech_segment = SpeechSegment()

    def process_frame(self, frame):
        self.frame_count += 1

        start_time = self.latency_tracker.start()

        is_speech = self.vad.process(frame)

        segment_started = False
        segment_finished = False
        segment = []

        if is_speech:
            if not self.speech_segment.active:
                self.speech_segment.start(frame)
                segment_started = True
            else:
                self.speech_segment.add(frame)

        elif self.speech_segment.active:
            segment = self.speech_segment.finish()
            segment_finished = True

        latency_ms = self.latency_tracker.stop(start_time)

        return {
            "frame_count": self.frame_count,
            "sample_rate": getattr(frame, "sample_rate", None),
            "samples": getattr(frame, "samples", None),
            "pts": getattr(frame, "pts", None),
            "is_speech": is_speech,
            "segment_started": segment_started,
            "segment_finished": segment_finished,
            "segment": segment,
            "segment_frame_count": len(segment),
            "latency_ms": latency_ms,
        }