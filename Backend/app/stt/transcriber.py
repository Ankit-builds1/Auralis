import av
import numpy as np
from faster_whisper import WhisperModel


class SpeechToText:
    """
    Converts a completed speech segment into text.

    Input:
        List of aiortc AudioFrame objects.

    WebRTC audio is usually 48 kHz.
    Whisper expects 16 kHz mono float32 audio.
    """

    def __init__(
        self,
        model_size="base",
        device="cpu",
        compute_type="int8",
    ):
        print(
            f"[STT] Loading Whisper model "
            f"(model={model_size}, device={device}, compute_type={compute_type})"
        )

        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )

        print("[STT] Whisper model loaded")

        # Convert incoming WebRTC audio to:
        # mono + 16 kHz + signed 16-bit PCM
        self.resampler = av.AudioResampler(
            format="s16",
            layout="mono",
            rate=16000,
        )

    def _frames_to_audio(self, frames):
        """
        Convert aiortc AudioFrames to a 16 kHz mono
        float32 NumPy waveform.
        """

        if not frames:
            print("[STT] No frames received")
            return None

        audio_chunks = []

        for frame in frames:

            try:
                # Resample 48 kHz -> 16 kHz
                resampled_frames = self.resampler.resample(frame)

                # Depending on PyAV version, this can be a frame
                # or a list of frames.
                if not isinstance(resampled_frames, list):
                    resampled_frames = [resampled_frames]

                for resampled_frame in resampled_frames:

                    array = resampled_frame.to_ndarray()

                    if array is None or array.size == 0:
                        continue

                    array = np.asarray(array)

                    # Mono audio should normally be:
                    # (1, samples)
                    if array.ndim == 2:
                        array = array[0]

                    array = array.reshape(-1)

                    # s16 -> float32 [-1, 1]
                    if np.issubdtype(array.dtype, np.integer):
                        array = (
                            array.astype(np.float32) / 32768.0
                        )
                    else:
                        array = array.astype(np.float32)

                    audio_chunks.append(array)

            except Exception as exc:
                print(
                    f"[STT] Error converting frame: {exc}"
                )

        if not audio_chunks:
            print("[STT] No usable audio after resampling")
            return None

        audio = np.concatenate(audio_chunks)

        print(
            f"[STT] Prepared audio: "
            f"samples={len(audio)}, "
            f"duration={len(audio) / 16000:.2f}s, "
            f"sample_rate=16000"
        )

        return audio

    def transcribe(self, frames):
        """
        Transcribe one completed speech segment.

        Returns:
            str: recognized text
        """

        print(
            f"[STT] Starting transcription "
            f"for {len(frames)} frames"
        )

        audio = self._frames_to_audio(frames)

        if audio is None or len(audio) == 0:
            print("[STT] Empty audio segment")
            return ""

        try:

            segments, info = self.model.transcribe(
                audio,
                language="en",
                beam_size=1,
                vad_filter=False,
            )

            text_parts = []

            for segment in segments:
                text = segment.text.strip()

                if text:
                    text_parts.append(text)

            transcript = " ".join(text_parts).strip()

            print(
                f"[STT] Transcript: {transcript}"
            )

            return transcript

        except Exception as exc:
            print(
                f"[STT] Transcription error: {exc}"
            )
            return ""