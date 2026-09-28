import numpy as np
import torch
from silero_vad import load_silero_vad


class VoiceActivityDetector:
    """
    Real-time Silero VAD for WebRTC audio.

    Browser / WebRTC:
        48 kHz
        960 samples/frame

    VAD:
        16 kHz
        512 samples/chunk

    This version:
        - converts stereo -> mono
        - converts PCM -> float32
        - resamples 48 kHz -> 16 kHz
        - buffers audio into 512-sample chunks
        - uses speech/silence smoothing
        - exposes the latest VAD probability
    """

    TARGET_SAMPLE_RATE = 16000
    VAD_CHUNK_SIZE = 512

    # Lower than the previous 0.35 because your logs
    # show probabilities around 0.20-0.39 while speaking.
    SPEECH_THRESHOLD = 0.20

    # Require multiple speech chunks before starting.
    SPEECH_CHUNKS_TO_START = 2

    # Require sustained silence before ending.
    SILENCE_CHUNKS_TO_END = 12

    def __init__(self):

        self.model = load_silero_vad(onnx=True)

        self.audio_buffer = np.empty(
            0,
            dtype=np.float32
        )

        self.in_speech = False

        self.consecutive_speech = 0
        self.consecutive_silence = 0

        self.speech_frames = 0
        self.silence_frames = 0

        self.last_probability = 0.0

        print(
            "[VAD] Silero VAD initialized "
            f"(sample_rate={self.TARGET_SAMPLE_RATE}, "
            f"chunk_size={self.VAD_CHUNK_SIZE}, "
            f"threshold={self.SPEECH_THRESHOLD}, "
            f"start_chunks={self.SPEECH_CHUNKS_TO_START}, "
            f"silence_chunks={self.SILENCE_CHUNKS_TO_END})"
        )

    # ==========================================================
    # Convert WebRTC frame -> mono 16 kHz float32
    # ==========================================================

    def _frame_to_audio(self, frame):

        if not hasattr(frame, "to_ndarray"):
            return None

        try:
            audio = frame.to_ndarray()
        except Exception as exc:
            print(
                f"[VAD ERROR] to_ndarray failed: "
                f"{type(exc).__name__}: {exc}"
            )
            return None

        if audio is None or audio.size == 0:
            return None

        audio = np.asarray(audio)

        # ------------------------------------------------------
        # Convert channels to mono
        # ------------------------------------------------------

        if audio.ndim == 2:

            try:
                channels = len(frame.layout.channels)
            except Exception:
                channels = 1

            if channels > 1:

                if audio.shape[0] == channels:
                    audio = audio.mean(axis=0)

                elif audio.shape[1] == channels:
                    audio = audio.mean(axis=1)

                else:
                    audio = audio.reshape(-1)

            else:
                audio = audio.reshape(-1)

        else:
            audio = audio.reshape(-1)

        # ------------------------------------------------------
        # Convert PCM -> float32
        # ------------------------------------------------------

        if np.issubdtype(audio.dtype, np.integer):

            info = np.iinfo(audio.dtype)

            max_value = max(
                abs(info.min),
                info.max
            )

            audio = (
                audio.astype(np.float32)
                / float(max_value)
            )

        else:

            audio = audio.astype(
                np.float32,
                copy=False
            )

        # ------------------------------------------------------
        # Get original sample rate
        # ------------------------------------------------------

        sample_rate = getattr(
            frame,
            "sample_rate",
            48000
        )

        if not sample_rate or sample_rate <= 0:
            sample_rate = 48000

        # ------------------------------------------------------
        # Resample -> 16 kHz
        # ------------------------------------------------------

        if sample_rate != self.TARGET_SAMPLE_RATE:

            old_length = len(audio)

            if old_length == 0:
                return None

            new_length = int(
                round(
                    old_length
                    * self.TARGET_SAMPLE_RATE
                    / sample_rate
                )
            )

            new_length = max(
                1,
                new_length
            )

            old_positions = np.linspace(
                0.0,
                1.0,
                old_length,
                endpoint=False
            )

            new_positions = np.linspace(
                0.0,
                1.0,
                new_length,
                endpoint=False
            )

            audio = np.interp(
                new_positions,
                old_positions,
                audio
            ).astype(
                np.float32
            )

        # Disabled: printed once per frame (~50/sec) and slowed
        # processing enough to cause a growing backlog.
        # print(
        #     "[AUDIO LEVEL] "
        #     f"min={audio.min():.6f} "
        #     f"max={audio.max():.6f} "
        #     f"mean={audio.mean():.6f} "
        #     f"rms={np.sqrt(np.mean(audio ** 2)):.6f}"
        # )

        return audio

    # ==========================================================
    # Process one WebRTC frame
    # ==========================================================

    def process(self, frame):

        audio = self._frame_to_audio(frame)

        if audio is None or len(audio) == 0:

            self.silence_frames += 1

            return self.in_speech

        # ------------------------------------------------------
        # Add audio to buffer
        # ------------------------------------------------------

        self.audio_buffer = np.concatenate(
            (
                self.audio_buffer,
                audio
            )
        )

        chunks_processed = 0

        # ------------------------------------------------------
        # Process every complete 512-sample chunk
        # ------------------------------------------------------

        while len(self.audio_buffer) >= self.VAD_CHUNK_SIZE:

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
            # Silero
            # --------------------------------------------------

            try:

                with torch.no_grad():

                    probability = self.model(
                        tensor,
                        self.TARGET_SAMPLE_RATE
                    )

                if hasattr(
                    probability,
                    "item"
                ):
                    probability = probability.item()

                probability = float(
                    probability
                )

            except Exception as exc:

                print(
                    "[VAD ERROR] Model failed: "
                    f"{type(exc).__name__}: {exc}"
                )

                continue

            chunks_processed += 1

            self.last_probability = probability

            is_speech = (
                probability >= self.SPEECH_THRESHOLD
            )

            # Disabled: printed once per 32 ms chunk and slowed
            # processing enough to cause a growing backlog.
            # print(
            #     "[VAD DEBUG] "
            #     f"probability={probability:.4f} "
            #     f"speech={is_speech} "
            #     f"in_speech={self.in_speech} "
            #     f"speech_count={self.consecutive_speech} "
            #     f"silence_count={self.consecutive_silence}"
            # )

            # ==================================================
            # SPEECH
            # ==================================================

            if is_speech:

                self.consecutive_speech += 1

                self.consecutive_silence = 0

                self.speech_frames += 1

                self.silence_frames = 0

                # ----------------------------------------------
                # Start speech after consecutive speech chunks
                # ----------------------------------------------

                if (
                    not self.in_speech
                    and
                    self.consecutive_speech
                    >= self.SPEECH_CHUNKS_TO_START
                ):

                    self.in_speech = True

                    print(
                        "[VAD] ============================="
                    )

                    print(
                        "[VAD] SPEECH STARTED"
                    )

                    print(
                        f"[VAD] probability={probability:.4f}"
                    )

                    print(
                        "[VAD] ============================="
                    )

            # ==================================================
            # SILENCE
            # ==================================================

            else:

                self.consecutive_speech = 0

                self.silence_frames += 1

                if self.in_speech:

                    self.consecutive_silence += 1

                    # Disabled: printed on every silent chunk.
                    # print(
                    #     "[VAD] silence "
                    #     f"{self.consecutive_silence}/"
                    #     f"{self.SILENCE_CHUNKS_TO_END}"
                    # )

                    # ------------------------------------------
                    # End speech after sustained silence
                    # ------------------------------------------

                    if (
                        self.consecutive_silence
                        >= self.SILENCE_CHUNKS_TO_END
                    ):

                        self.in_speech = False

                        self.consecutive_silence = 0

                        print(
                            "[VAD] ============================="
                        )

                        print(
                            "[VAD] SPEECH ENDED"
                        )

                        print(
                            "[VAD] ============================="
                        )

        # ------------------------------------------------------
        # No complete chunk yet
        # ------------------------------------------------------

        if chunks_processed == 0:
            return self.in_speech

        return self.in_speech

    # ==========================================================
    # Optional helper
    # ==========================================================

    def get_state(self):

        return {
            "is_speech": self.in_speech,
            "probability": self.last_probability,
            "speech_frames": self.speech_frames,
            "silence_frames": self.silence_frames,
            "consecutive_speech": self.consecutive_speech,
            "consecutive_silence": self.consecutive_silence,
        }

    # ==========================================================
    # Reset
    # ==========================================================

    def reset(self):

        self.audio_buffer = np.empty(
            0,
            dtype=np.float32
        )

        self.in_speech = False

        self.consecutive_speech = 0
        self.consecutive_silence = 0

        self.speech_frames = 0
        self.silence_frames = 0

        self.last_probability = 0.0

        print("[VAD] Reset")