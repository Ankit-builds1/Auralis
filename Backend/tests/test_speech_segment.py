from app.audio.segments.speech_segment import SpeechSegment


def test_speech_segment_collects_frames():
    segment = SpeechSegment()

    frame1 = object()
    frame2 = object()
    frame3 = object()

    assert segment.active is False
    assert segment.frame_count == 0

    segment.start(frame1)

    assert segment.active is True
    assert segment.frame_count == 1

    segment.add(frame2)
    segment.add(frame3)

    assert segment.frame_count == 3

    result = segment.finish()

    assert result == [frame1, frame2, frame3]
    assert segment.active is False
    assert segment.frame_count == 0


def test_speech_segment_add_starts_when_inactive():
    segment = SpeechSegment()

    frame = object()

    segment.add(frame)

    assert segment.active is True
    assert segment.frame_count == 1


def test_speech_segment_finish_when_empty():
    segment = SpeechSegment()

    result = segment.finish()

    assert result == []
    assert segment.active is False
    assert segment.frame_count == 0