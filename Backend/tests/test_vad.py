from av import AudioFrame

from app.audio.vad.detector import VoiceActivityDetector


def test_vad_detects_audio_frame():
    frame = AudioFrame(
        format="s16",
        layout="mono",
        samples=160
    )

    detector = VoiceActivityDetector()

    result = detector.process(frame)

    assert result is True
    assert detector.speech_frames == 1
    assert detector.silence_frames == 0