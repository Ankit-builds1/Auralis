import numpy as np

from app.audio.pcm.converter import PCMConverter


class FakeLayout:
    channels = ["mono"]


class FakeFrame:
    sample_rate = 48000
    samples = 960
    pts = 0
    layout = FakeLayout()

    def to_ndarray(self):
        return np.zeros((1, 960), dtype=np.int16)


def test_pcm_converter_extracts_audio():
    converter = PCMConverter()

    result = converter.convert(FakeFrame())

    assert result["sample_rate"] == 48000
    assert result["samples"] == 960
    assert result["channels"] == 1
    assert result["pcm"].shape == (1, 960)