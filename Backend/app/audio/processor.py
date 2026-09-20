from app.audio.metrics.latency import LatencyTracker
from app.audio.vad.detector import VoiceActivityDetector


class AudioProcessor:
    """
    Handles incoming audio frames.

    Day 5:
    - Receive audio frames
    - Inspect basic metadata
    - Run voice activity detection
    - Measure processing latency

    Later:
    - Convert frames to PCM
    - Speech-to-Text
    - Emotion analysis
    """

    def __init__(self):
        self.frame_count = 0
        self.vad = VoiceActivityDetector()
        self.latency_tracker = LatencyTracker()

    def process_frame(self, frame):
        self.frame_count += 1

        start_time = self.latency_tracker.start()

        is_speech = self.vad.process(frame)

        latency_ms = self.latency_tracker.stop(start_time)

        return {
            "frame_count": self.frame_count,
            "sample_rate": getattr(frame, "sample_rate", None),
            "samples": getattr(frame, "samples", None),
            "pts": getattr(frame, "pts", None),
            "is_speech": is_speech,
            "latency_ms": latency_ms,
        }