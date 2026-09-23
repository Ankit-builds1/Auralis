import numpy as np
from av import AudioFrame

from app.audio.processor import AudioProcessor


def test_audio_processor_processes_frame():
    sample_rate = 16000
    samples = 512

    audio = np.zeros(samples, dtype=np.int16)

    frame = AudioFrame.from_ndarray(
        audio.reshape(1, -1),
        format="s16",
        layout="mono",
    )

    frame.sample_rate = sample_rate
    frame.pts = 123

    processor = AudioProcessor()

    result = processor.process_frame(frame)

    assert result["frame_count"] == 1
    assert result["sample_rate"] == 16000
    assert result["samples"] == samples
    assert result["pts"] == 123
    assert isinstance(result["is_speech"], bool)
    assert isinstance(result["latency_ms"], float)
    assert result["latency_ms"] >= 0