class VoiceActivityDetector:
    """
    Basic voice activity detector foundation.

    Day 4:
    - Accept processed audio frames
    - Provide a speech/silence decision interface

    Later:
    - Replace the simple detector with Silero VAD
    - Support streaming speech detection
    - Detect speech start/stop events
    """

    def __init__(self):
        self.speech_frames = 0
        self.silence_frames = 0

    def process(self, frame):
        """
        Process one audio frame.

        Returns:
            bool: True when speech is detected.
        """

        samples = getattr(frame, "samples", 0)

        if samples:
            self.speech_frames += 1
            self.silence_frames = 0
            return True

        self.silence_frames += 1
        return False