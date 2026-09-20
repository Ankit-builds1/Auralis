import time

from app.audio.metrics.latency import LatencyTracker


def test_latency_tracker_records_latency():
    tracker = LatencyTracker()

    start = tracker.start()

    time.sleep(0.01)

    latency = tracker.stop(start)

    assert latency >= 0
    assert tracker.last_latency_ms == latency
    assert tracker.average_latency_ms == latency