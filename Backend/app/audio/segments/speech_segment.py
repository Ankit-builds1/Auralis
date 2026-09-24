class SpeechSegment:
    """
    Collects consecutive speech audio frames into one segment.
    """

    def __init__(self):
        self.frames = []
        self.active = False

    def start(self, frame):
        """
        Start a new speech segment.
        """
        self.frames = [frame]
        self.active = True

    def add(self, frame):
        """
        Add another frame to the active speech segment.
        """
        if not self.active:
            self.start(frame)
            return

        self.frames.append(frame)

    def finish(self):
        """
        Finish the current speech segment.

        Returns:
            list: collected audio frames.
        """
        if not self.active:
            return []

        segment = self.frames

        self.frames = []
        self.active = False

        return segment

    @property
    def frame_count(self):
        """
        Number of frames currently collected.
        """
        return len(self.frames)