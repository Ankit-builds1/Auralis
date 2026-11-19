class PCMConverter:
    """
    Converts incoming aiortc audio frames into PCM audio data.

    Day 6:
    - Extract PCM samples from audio frames
    - Expose sample rate and channel information
    - Prepare audio data for STT and emotion analysis
    """

    def convert(self, frame):
        """
        Convert an audio frame into PCM bytes.

        Returns:
            dict: PCM audio information.
        """

        pcm = frame.to_ndarray()

        sample_rate = getattr(frame, "sample_rate", None)
        samples = getattr(frame, "samples", None)

        channels = "unknown"

        if frame.layout:
            channels = len(frame.layout.channels)

        return {
            "pcm": pcm,
            "sample_rate": sample_rate,
            "samples": samples,
            "channels": channels,
        }