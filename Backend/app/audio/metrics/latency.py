import time


class LatencyTracker:
    """
    Tracks processing latency for the audio pipeline.
    """

    def __init__(self):
        self.samples = []

    def start(self):
        return time.perf_counter()

    def stop(self, start_time):
        latency_ms = (time.perf_counter() - start_time) * 1000

        self.samples.append(latency_ms)

        return latency_ms

    @property
    def last_latency_ms(self):
        if not self.samples:
            return None

        return self.samples[-1]

    @property
    def average_latency_ms(self):
        if not self.samples:
            return None

        return sum(self.samples) / len(self.samples)