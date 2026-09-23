import numpy as np
import torch
from silero_vad import load_silero_vad


class VoiceActivityDetector:
    """
    Real-time voice activity detector using Silero VAD.

    Day 7:
    - Convert incoming audio frames to mono float audio
    - Resample audio to 16 kHz
    - Run Silero VAD
    - Track speech/silence frames
    """

    TARGET_SAMPLE_RATE = 16000

    def __init__(self):
        self.speech_frames = 0
        self.silence_frames = 0
        self.model = load_silero_vad(onnx=True)

    def _frame_to_tensor(self, frame):
        """
        Convert an aiortc AudioFrame into a mono 16 kHz float tensor.
        """

        if not hasattr(frame, "to_ndarray"):
            return None

        audio = frame.to_ndarray()

        if audio is None or audio.size == 0:
            return None

        audio = np.asarray(audio)

        channels = 1

        if getattr(frame, "layout", None):
            try:
                channels = len(frame.layout.channels)
            except TypeError:
                channels = 1

        # Convert multi-channel audio to mono.
        if audio.ndim == 2:
            if channels > 1 and audio.shape[0] == channels:
                audio = audio.mean(axis=0)
            elif channels > 1 and audio.shape[1] == channels:
                audio = audio.mean(axis=1)
            else:
                audio = audio.reshape(-1)

        audio = audio.reshape(-1)

        # Convert integer PCM to float32 [-1, 1].
        if np.issubdtype(audio.dtype, np.integer):
            info = np.iinfo(audio.dtype)
            max_value = max(abs(info.min), info.max)
            audio = audio.astype(np.float32) / max_value
        else:
            audio = audio.astype(np.float32)

        sample_rate = getattr(
            frame,
            "sample_rate",
            self.TARGET_SAMPLE_RATE,
        )

        # Some synthetic test frames may not specify a sample rate.
        if not sample_rate or sample_rate <= 0:
            sample_rate = self.TARGET_SAMPLE_RATE

        # Resample to Silero's required 16 kHz.
        if sample_rate != self.TARGET_SAMPLE_RATE:
            old_length = len(audio)

            if old_length == 0:
                return None

            new_length = max(
                1,
                int(
                    old_length
                    * self.TARGET_SAMPLE_RATE
                    / sample_rate
                ),
            )

            old_positions = np.linspace(
                0,
                1,
                old_length,
                endpoint=False,
            )

            new_positions = np.linspace(
                0,
                1,
                new_length,
                endpoint=False,
            )

            audio = np.interp(
                new_positions,
                old_positions,
                audio,
            ).astype(np.float32)

        return torch.from_numpy(audio)

    def process(self, frame):
        """
        Process one audio frame.

        Returns:
            bool: True when Silero detects speech.
        """

        audio = self._frame_to_tensor(frame)

        if audio is None or audio.numel() == 0:
            self.silence_frames += 1
            return False

        with torch.no_grad():
            speech_probability = self.model(
                audio,
                self.TARGET_SAMPLE_RATE,
            )

        if hasattr(speech_probability, "item"):
            speech_probability = speech_probability.item()

        is_speech = float(speech_probability) >= 0.5

        if is_speech:
            self.speech_frames += 1
            self.silence_frames = 0
        else:
            self.silence_frames += 1

        return is_speech