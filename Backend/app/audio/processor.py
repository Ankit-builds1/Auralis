class AudioProcessor:
    """
    Handles incoming audio frames.

    Day 3:
    - Receive audio frames
    - Inspect basic metadata

    Later:
    - Convert frames to PCM
    - Voice Activity Detection
    - Speech-to-Text
    - Emotion analysis
    """

    def __init__(self):
        self.frame_count = 0

    def process_frame(self, frame):
        self.frame_count += 1

        return {
            "frame_count": self.frame_count,
            "sample_rate": getattr(frame, "sample_rate", None),
            "samples": getattr(frame, "samples", None),
            "pts": getattr(frame, "pts", None),
        }