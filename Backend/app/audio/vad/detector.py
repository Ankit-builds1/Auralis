import numpy as np
import torch
from silero_vad import load_silero_vad


class VoiceActivityDetector:
    """
    Real-time voice activity detector using Silero VAD.

    Converts incoming aiortc audio frames to mono 16 kHz
    and feeds Silero VAD exactly 512 samples at a time.
    """

    TARGET_SAMPLE_RATE = 16000
    VAD_CHUNK_SIZE = 512
    SPEECH_THRESHOLD = 0.5

    def __init__(self):
        self.speech_frames = 0
        self.silence_frames = 0

        # Buffer for audio that does not make a complete
        # 512-sample Silero VAD chunk.
        self.audio_buffer = np.empty(
            0,
            dtype=np.float32
        )

        self.model = load_silero_vad(onnx=True)

        print(
            "[VAD] Silero VAD initialized "
            f"(sample_rate={self.TARGET_SAMPLE_RATE}, "
            f"chunk_size={self.VAD_CHUNK_SIZE})"
        )

    def _frame_to_tensor(self, frame):
        """
        Convert an aiortc AudioFrame into mono 16 kHz
        float32 audio.
        """

        if not hasattr(frame, "to_ndarray"):
            return None

        audio = frame.to_ndarray()

        if audio is None or audio.size == 0:
            return None

        audio = np.asarray(audio)

        # --------------------------------------------------
        # Convert multi-channel audio to mono
        # --------------------------------------------------

        channels = 1

        if getattr(frame, "layout", None):
            try:
                channels = len(frame.layout.channels)
            except (TypeError, AttributeError):
                channels = 1

        if audio.ndim == 2:
            if channels > 1 and audio.shape[0] == channels:
                audio = audio.mean(axis=0)

            elif channels > 1 and audio.shape[1] == channels:
                audio = audio.mean(axis=1)

            else:
                audio = audio.reshape(-1)

        audio = audio.reshape(-1)

        # --------------------------------------------------
        # Convert PCM integer -> float32 [-1, 1]
        # --------------------------------------------------

        if np.issubdtype(audio.dtype, np.integer):
            info = np.iinfo(audio.dtype)

            max_value = max(
                abs(info.min),
                info.max
            )

            audio = (
                audio.astype(np.float32)
                / max_value
            )

        else:
            audio = audio.astype(
                np.float32,
                copy=False
            )

        # --------------------------------------------------
        # Get sample rate
        # --------------------------------------------------

        sample_rate = getattr(
            frame,
            "sample_rate",
            self.TARGET_SAMPLE_RATE
        )

        if not sample_rate or sample_rate <= 0:
            sample_rate = self.TARGET_SAMPLE_RATE

        # --------------------------------------------------
        # Resample to 16 kHz
        # --------------------------------------------------

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
                )
            )

            old_positions = np.linspace(
                0,
                1,
                old_length,
                endpoint=False
            )

            new_positions = np.linspace(
                0,
                1,
                new_length,
                endpoint=False
            )

            audio = np.interp(
                new_positions,
                old_positions,
                audio
            ).astype(np.float32)

        return torch.from_numpy(audio)

    def process(self, frame):
        """
        Process one incoming WebRTC audio frame.

        The browser currently sends 640 samples per frame.
        Silero VAD requires exactly 512 samples at 16 kHz.

        Therefore audio is buffered and processed in
        512-sample chunks.

        Returns:
            bool: True if speech is detected.
        """

        audio = self._frame_to_tensor(frame)

        if audio is None or audio.numel() == 0:
            self.silence_frames += 1
            return False

        # --------------------------------------------------
        # Add incoming samples to buffer
        # --------------------------------------------------

        incoming = audio.numpy()

        self.audio_buffer = np.concatenate(
            (
                self.audio_buffer,
                incoming
            )
        )

        speech_detected = False
        chunks_processed = 0

        # --------------------------------------------------
        # Process complete 512-sample chunks
        # --------------------------------------------------

        while (
            len(self.audio_buffer)
            >= self.VAD_CHUNK_SIZE
        ):

            chunk = self.audio_buffer[
                :self.VAD_CHUNK_SIZE
            ]

            self.audio_buffer = self.audio_buffer[
                self.VAD_CHUNK_SIZE:
            ]

            tensor = torch.from_numpy(
                chunk.astype(
                    np.float32,
                    copy=False
                )
            )

            # --------------------------------------------------
            # Silero VAD
            # --------------------------------------------------

            with torch.no_grad():
                speech_probability = self.model(
                    tensor,
                    self.TARGET_SAMPLE_RATE
                )

            if hasattr(
                speech_probability,
                "item"
            ):
                speech_probability = (
                    speech_probability.item()
                )

            speech_probability = float(
                speech_probability
            )

            is_speech = (
                speech_probability
                >= self.SPEECH_THRESHOLD
            )

            chunks_processed += 1

            # --------------------------------------------------
            # Track speech / silence
            # --------------------------------------------------

            if is_speech:
                speech_detected = True

                self.speech_frames += 1
                self.silence_frames = 0

            else:
                self.silence_frames += 1

        # --------------------------------------------------
        # If no complete 512-sample chunk was available yet,
        # keep waiting for the next WebRTC frame.
        # --------------------------------------------------

        if chunks_processed == 0:
            return False

        return speech_detected