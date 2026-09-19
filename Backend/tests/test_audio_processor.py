from av import AudioFrame

from app.audio.processor import AudioProcessor


def test_audio_processor_processes_frame():
    frame = AudioFrame(
        format="s16",
        layout="mono",
        samples=160
    )

    frame.sample_rate = 16000
    frame.pts = 123

    processor = AudioProcessor()

    result = processor.process_frame(frame)

    assert result["frame_count"] == 1
    assert result["sample_rate"] == 16000
    assert result["samples"] == 160
    assert result["pts"] == 123
    assert result["is_speech"] is True