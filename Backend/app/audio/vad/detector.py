import math

import numpy as np
import torch
from silero_vad import load_silero_vad

try:
    import av
    HAS_AV = True
except Exception:
    HAS_AV = False

try:
    from scipy.signal import resample_poly
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


# Keep the VAD on one CPU thread so it does not compete with Whisper
# for CPU time (Whisper runs at the same moment in the background).
torch.set_num_threads(1)


class VoiceActivityDetector:
    """
    Real-time Silero VAD for WebRTC audio.

    Browser / WebRTC: 48 kHz, 960 samples/frame
    VAD:              16 kHz, 512 samples/chunk (32 ms)

    Steps:
        - WebRTC frame -> mono 16 kHz float32
          (stateful av.AudioResampler, so there are no clicks at frame
           edges; falls back to resample_poly / np.interp)
        - buffer into 512-sample chunks
        - speech/silence smoothing
        - expose latest VAD probability
    """

    TARGET_SAMPLE_RATE = 16000
    VAD_CHUNK_SIZE = 512

    # Silero's default value. If speech is never detected, try 0.35-0.4.
    SPEECH_THRESHOLD = 0.5

    # Consecutive speech chunks needed to start (3 x 32 ms = ~96 ms).
    SPEECH_CHUNKS_TO_START = 3

    # Consecutive silence chunks needed to end (22 x 32 ms = ~700 ms).
    # NOTE: this is the ONLY silence wait. Keep the hangover in
    # processor.py small (0-200 ms) or the two waits add up.
    SILENCE_CHUNKS_TO_END = 22

    def __init__(self):
        self.model = load_silero_vad(onnx=True)

        self.resampler = None
        self._create_resampler()

        self.audio_buffer = np.empty(0, dtype=np.float32)

        self.in_speech = False

        self.consecutive_speech = 0
        self.consecutive_silence = 0

        # Counted in 32 ms VAD chunks (not WebRTC frames)
        self.speech_chunks = 0
        self.silence_chunks = 0

        self.last_probability = 0.0

        print(
            "[VAD] Silero VAD initialized "
            f"(sample_rate={self.TARGET_SAMPLE_RATE}, "
            f"chunk_size={self.VAD_CHUNK_SIZE}, "
            f"threshold={self.SPEECH_THRESHOLD}, "
            f"start_chunks={self.SPEECH_CHUNKS_TO_START}, "
            f"silence_chunks={self.SILENCE_CHUNKS_TO_END}, "
            f"resampler={self._resampler_name()})"
        )

    # ==========================================================
    # Resampling
    # ==========================================================

    def _create_resampler(self):
        """Stateful resampler: keeps filter history between frames."""
        if not HAS_AV:
            self.resampler = None
            return

        try:
            self.resampler = av.AudioResampler(
                format="s16",
                layout="mono",
                rate=self.TARGET_SAMPLE_RATE,
            )
        except Exception as exc:
            print(f"[VAD] AudioResampler unavailable: {type(exc).__name__}: {exc}")
            self.resampler = None

    def _resampler_name(self):
        if self.resampler is not None:
            return "av.AudioResampler"
        if HAS_SCIPY:
            return "resample_poly"
        return "np.interp"

    def _resample_with_av(self, frame):
        """Return mono 16 kHz float32 audio, or None if it fails."""
        if self.resampler is None:
            return None

        try:
            result = self.resampler.resample(frame)
        except Exception as exc:
            print(f"[VAD] AudioResampler failed, using fallback: {type(exc).__name__}: {exc}")
            self.resampler = None
            return None

        # PyAV >= 9 returns a list of frames; older versions return one frame
        frames = result if isinstance(result, list) else [result]

        parts = []
        for out_frame in frames:
            if out_frame is None:
                continue
            pcm = out_frame.to_ndarray().reshape(-1)
            if pcm.size > 0:
                parts.append(pcm.astype(np.float32) / 32768.0)

        if not parts:
            # The resampler may buffer a few samples; that is normal
            return np.empty(0, dtype=np.float32)

        return np.concatenate(parts)

    def _resample_array(self, audio, sample_rate):
        """Fallback for one frame at a time (no state between frames)."""
        if sample_rate == self.TARGET_SAMPLE_RATE:
            return audio

        if HAS_SCIPY:
            g = math.gcd(int(sample_rate), self.TARGET_SAMPLE_RATE)
            up = self.TARGET_SAMPLE_RATE // g
            down = int(sample_rate) // g
            return resample_poly(audio, up, down).astype(np.float32)

        old_length = len(audio)
        if old_length == 0:
            return None

        new_length = max(
            1,
            int(round(old_length * self.TARGET_SAMPLE_RATE / sample_rate)),
        )
        old_positions = np.linspace(0.0, 1.0, old_length, endpoint=False)
        new_positions = np.linspace(0.0, 1.0, new_length, endpoint=False)
        return np.interp(new_positions, old_positions, audio).astype(np.float32)

    # ==========================================================
    # Convert WebRTC frame -> mono 16 kHz float32
    # ==========================================================

    def _frame_to_audio(self, frame):
        # ---- preferred path: stateful av resampler ----
        if HAS_AV and isinstance(frame, av.AudioFrame):
            audio = self._resample_with_av(frame)
            if audio is not None:
                return audio

        # ---- fallback path ----
        if not hasattr(frame, "to_ndarray"):
            return None

        try:
            audio = frame.to_ndarray()
        except Exception as exc:
            print(f"[VAD ERROR] to_ndarray failed: {type(exc).__name__}: {exc}")
            return None

        if audio is None or audio.size == 0:
            return None

        audio = np.asarray(audio)

        # ---- channels -> mono ----
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

        # ---- PCM -> float32 ----
        if np.issubdtype(audio.dtype, np.integer):
            info = np.iinfo(audio.dtype)
            max_value = max(abs(info.min), info.max)
            audio = audio.astype(np.float32) / float(max_value)
        else:
            audio = audio.astype(np.float32, copy=False)

        # ---- sample rate ----
        sample_rate = getattr(frame, "sample_rate", 48000)
        if not sample_rate or sample_rate <= 0:
            sample_rate = 48000

        audio = self._resample_array(audio, sample_rate)

        if audio is None or len(audio) == 0:
            return None

        return audio

    # ==========================================================
    # Process one WebRTC frame
    # ==========================================================

    def process(self, frame):
        audio = self._frame_to_audio(frame)

        if audio is None or len(audio) == 0:
            return self.in_speech

        self.audio_buffer = np.concatenate((self.audio_buffer, audio))

        # Process every complete 512-sample chunk
        while len(self.audio_buffer) >= self.VAD_CHUNK_SIZE:
            chunk = self.audio_buffer[: self.VAD_CHUNK_SIZE]
            self.audio_buffer = self.audio_buffer[self.VAD_CHUNK_SIZE:]

            tensor = torch.from_numpy(chunk.astype(np.float32, copy=False))

            try:
                with torch.no_grad():
                    probability = self.model(tensor, self.TARGET_SAMPLE_RATE)

                if hasattr(probability, "item"):
                    probability = probability.item()

                probability = float(probability)

            except Exception as exc:
                print(f"[VAD ERROR] Model failed: {type(exc).__name__}: {exc}")
                continue

            self.last_probability = probability
            is_speech = probability >= self.SPEECH_THRESHOLD

            # ---------------- SPEECH ----------------
            if is_speech:
                self.consecutive_speech += 1
                self.consecutive_silence = 0

                self.speech_chunks += 1
                self.silence_chunks = 0

                if (
                    not self.in_speech
                    and self.consecutive_speech >= self.SPEECH_CHUNKS_TO_START
                ):
                    self.in_speech = True
                    print(f"[VAD] SPEECH STARTED (probability={probability:.2f})")

            # ---------------- SILENCE ----------------
            else:
                self.consecutive_speech = 0
                self.silence_chunks += 1

                if self.in_speech:
                    self.consecutive_silence += 1

                    if self.consecutive_silence >= self.SILENCE_CHUNKS_TO_END:
                        self.in_speech = False
                        self.consecutive_silence = 0
                        print("[VAD] SPEECH ENDED")

        return self.in_speech

    # ==========================================================
    # Helpers
    # ==========================================================

    def get_state(self):
        return {
            "is_speech": self.in_speech,
            "probability": self.last_probability,
            "speech_chunks": self.speech_chunks,
            "silence_chunks": self.silence_chunks,
            "consecutive_speech": self.consecutive_speech,
            "consecutive_silence": self.consecutive_silence,
            # Old key names kept so existing callers do not break
            "speech_frames": self.speech_chunks,
            "silence_frames": self.silence_chunks,
        }

    def reset(self):
        self.audio_buffer = np.empty(0, dtype=np.float32)

        self.in_speech = False

        self.consecutive_speech = 0
        self.consecutive_silence = 0

        self.speech_chunks = 0
        self.silence_chunks = 0

        self.last_probability = 0.0

        # Fresh resampler so old filter history is not carried over
        self._create_resampler()

        # Clear Silero's internal state too
        try:
            self.model.reset_states()
        except Exception as exc:
            print(f"[VAD] reset_states failed: {type(exc).__name__}: {exc}")

        print("[VAD] Reset")