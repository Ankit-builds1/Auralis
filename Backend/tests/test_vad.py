import numpy as np
from av import AudioFrame

from app.audio.vad.detector import VoiceActivityDetector


def test_vad_detects_audio_frame():
    sample_rate = 16000
    samples = 512

    audio = np.zeros(samples, dtype=np.int16)

    frame = AudioFrame.from_ndarray(
        audio.reshape(1, -1),
        format="s16",
        layout="mono",
    )

    frame.sample_rate = sample_rate

    detector = VoiceActivityDetector()

    result = detector.process(frame)

    assert isinstance(result, bool)
    assert detector.speech_frames + detector.silence_frames == 1