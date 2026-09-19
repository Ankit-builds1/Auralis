from app.audio.vad.detector import VoiceActivityDetector


class AudioProcessor:
    """
    Handles incoming audio frames.

    Day 4:
    - Receive audio frames
    - Inspect basic metadata
    - Run voice activity detection

    Later:
    - Convert frames to PCM
    - Speech-to-Text
    - Emotion analysis
    """

    def __init__(self):
        self.frame_count = 0
        self.vad = VoiceActivityDetector()

    def process_frame(self, frame):
        self.frame_count += 1

        is_speech = self.vad.process(frame)

        return {
            "frame_count": self.frame_count,
            "sample_rate": getattr(frame, "sample_rate", None),
            "samples": getattr(frame, "samples", None),
            "pts": getattr(frame, "pts", None),
            "is_speech": is_speech,
        }