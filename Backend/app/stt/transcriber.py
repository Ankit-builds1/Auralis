import glob
import os
import sys
import threading
import time

import av
import numpy as np
from faster_whisper import WhisperModel


TARGET_SAMPLE_RATE = 16000

# ---------------------------------------------------------------------------
# Week 3 Day 6: Whisper settings from the environment / .env
#   WHISPER_MODEL         base.en (default), small.en, ...
#   WHISPER_DEVICE        auto (default) | cpu | cuda
#                         auto = GPU if one is usable, otherwise CPU
#   WHISPER_COMPUTE_TYPE  default float16 on GPU, int8 on CPU
#   WHISPER_BEAM_SIZE     5 (default, most accurate) | 1 (fastest, greedy)
# If the GPU cannot be used (no CUDA libraries, out of memory, ...), the
# server says why and falls back to CPU instead of crashing.
# ---------------------------------------------------------------------------
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base.en")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto").strip().lower()
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "").strip().lower()
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "5"))


def _add_cuda_dll_dirs():
    """
    Windows: faster-whisper needs cuBLAS + cuDNN DLLs for the GPU. The
    easiest way to get them is `python -m pip install nvidia-cublas-cu12
    nvidia-cudnn-cu12==9.*`, which puts them in site-packages/nvidia/*/bin.
    Windows does not search there by default, so add those folders.
    """
    if sys.platform != "win32":
        return
    for base in sys.path:
        for folder in glob.glob(os.path.join(base, "nvidia", "*", "bin")):
            try:
                os.add_dll_directory(folder)
            except (OSError, AttributeError):
                pass
            if folder not in os.environ.get("PATH", ""):
                os.environ["PATH"] = folder + os.pathsep + os.environ.get("PATH", "")


def _gpu_available():
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False

# Words Whisper should expect. Helps it spell names and project terms
# correctly ("Ankit Dash" instead of "Ankida", "BTech" instead of "BTEK").
DEFAULT_INITIAL_PROMPT = (
    "Ankit Dash, Priya, Siddhant, BTech, CSE, Auralis, "
    "Centurion University, Bhubaneswar, Odisha."
)

# Whisper's own "this is not speech" check. A segment is dropped when
# Whisper is fairly sure there is no speech AND is unsure about the text.
NO_SPEECH_PROB_THRESHOLD = 0.6
LOW_LOGPROB_THRESHOLD = -1.0

# Phrases Whisper commonly invents for noise or silence. They are only
# dropped when Whisper also had some doubt that there was speech, so a
# real "Thank you." from the user still gets through.
HALLUCINATION_PHRASES = {
    "thank you",
    "thank you.",
    "thanks for watching",
    "thanks for watching!",
    "thank you for watching",
    "thank you for watching.",
    "you",
    "bye",
    "bye.",
}
HALLUCINATION_NO_SPEECH_PROB = 0.3


class SpeechToText:
    """
    Converts a completed speech segment into text.

    Input:
        List of aiortc AudioFrame objects.

    WebRTC audio is usually 48 kHz.
    Whisper expects 16 kHz mono float32 audio.

    Usage:
        stt = SpeechToText()
        stt.warmup()            # call once at server start
        text = stt.transcribe(frames)
    """

    def __init__(
        self,
        model_size=None,
        device=None,
        compute_type=None,
        beam_size=None,
        initial_prompt=DEFAULT_INITIAL_PROMPT,
    ):
        model_size = model_size or WHISPER_MODEL
        device = (device or WHISPER_DEVICE).lower()
        beam_size = beam_size or WHISPER_BEAM_SIZE
        self.fallback_reason = None

        if device in ("auto", "cuda"):
            _add_cuda_dll_dirs()
        if device == "auto":
            device = "cuda" if _gpu_available() else "cpu"

        self.model = None
        if device == "cuda":
            gpu_type = compute_type or WHISPER_COMPUTE_TYPE or "float16"
            try:
                self._load(model_size, "cuda", gpu_type, beam_size)
                self._self_test()     # cuBLAS / cuDNN errors show up here
            except Exception as exc:
                self.fallback_reason = f"{type(exc).__name__}: {exc}"
                print(f"[STT] GPU not usable ({self.fallback_reason}) -> falling back to CPU")
                self.model = None

        if self.model is None:
            cpu_type = (compute_type or WHISPER_COMPUTE_TYPE or "int8")
            if cpu_type == "float16":       # float16 is GPU-only
                cpu_type = "int8"
            self._load(model_size, "cpu", cpu_type, beam_size)

        self.model_size = model_size
        self.beam_size = beam_size
        self.initial_prompt = initial_prompt

        # Only one transcription at a time. Two sentences can finish
        # close together, and both run in background threads.
        self._lock = threading.Lock()

        print(f"[STT] Whisper model loaded on {self.device.upper()}")

    def _load(self, model_size, device, compute_type, beam_size):
        print(
            f"[STT] Loading Whisper model "
            f"(model={model_size}, device={device}, "
            f"compute_type={compute_type}, beam_size={beam_size})"
        )
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.device = device
        self.compute_type = compute_type

    def _self_test(self):
        """Tiny transcription: makes missing GPU libraries fail now, not mid-call."""
        segments, _ = self.model.transcribe(
            np.zeros(TARGET_SAMPLE_RATE // 2, dtype=np.float32),
            language="en", beam_size=1, vad_filter=False,
        )
        list(segments)

    def info(self):
        """Settings shown on /health and in the frontend chips (Week 3 Day 6)."""
        return {
            "model": self.model_size,
            "device": self.device,
            "compute_type": self.compute_type,
            "beam_size": self.beam_size,
            "gpu_fallback": self.fallback_reason,
        }

    # ==========================================================
    # Warm-up
    # ==========================================================

    def warmup(self):
        """
        Run one tiny transcription so the first real sentence is fast.
        Call this once when the server starts (not on the first segment).
        """
        start = time.perf_counter()

        silence = np.zeros(TARGET_SAMPLE_RATE, dtype=np.float32)  # 1 s

        with self._lock:
            segments, _ = self.model.transcribe(
                silence,
                language="en",
                beam_size=1,
                vad_filter=False,
            )
            list(segments)  # transcribe() is lazy; this makes it run

        print(f"[STT] Warm-up done in {(time.perf_counter() - start) * 1000:.0f} ms")

    # ==========================================================
    # Frames -> audio
    # ==========================================================

    def _frames_to_audio(self, frames):
        """
        Convert aiortc AudioFrames to a 16 kHz mono
        float32 NumPy waveform.

        A NEW resampler is created for every segment, and it is flushed at
        the end. With one shared resampler, a few leftover samples from the
        previous sentence were added to the start of the next one, and the
        last few milliseconds of every sentence were lost.
        """

        if not frames:
            print("[STT] No frames received")
            return None

        resampler = av.AudioResampler(
            format="s16",
            layout="mono",
            rate=TARGET_SAMPLE_RATE,
        )

        audio_chunks = []

        def collect(resampled):
            if resampled is None:
                return

            if not isinstance(resampled, list):
                resampled = [resampled]

            for resampled_frame in resampled:
                if resampled_frame is None:
                    continue

                array = resampled_frame.to_ndarray()

                if array is None or array.size == 0:
                    continue

                array = np.asarray(array).reshape(-1)

                # s16 -> float32 [-1, 1]
                if np.issubdtype(array.dtype, np.integer):
                    array = array.astype(np.float32) / 32768.0
                else:
                    array = array.astype(np.float32)

                audio_chunks.append(array)

        for frame in frames:
            try:
                collect(resampler.resample(frame))
            except Exception as exc:
                print(f"[STT] Error converting frame: {type(exc).__name__}: {exc}")

        # Flush: get the samples still held inside the resampler
        try:
            collect(resampler.resample(None))
        except Exception:
            pass

        if not audio_chunks:
            print("[STT] No usable audio after resampling")
            return None

        audio = np.concatenate(audio_chunks)

        print(
            f"[STT] Prepared audio: "
            f"samples={len(audio)}, "
            f"duration={len(audio) / TARGET_SAMPLE_RATE:.2f}s, "
            f"sample_rate={TARGET_SAMPLE_RATE}"
        )

        return audio

    # ==========================================================
    # Filtering
    # ==========================================================

    def _keep_segment(self, segment):
        """Return False for text Whisper probably invented from noise."""
        text = segment.text.strip()

        if not text:
            return False

        no_speech_prob = getattr(segment, "no_speech_prob", 0.0) or 0.0
        avg_logprob = getattr(segment, "avg_logprob", 0.0) or 0.0

        # Whisper's standard rule: likely silence AND low confidence
        if (
            no_speech_prob > NO_SPEECH_PROB_THRESHOLD
            and avg_logprob < LOW_LOGPROB_THRESHOLD
        ):
            print(
                f"[STT] Dropped (no speech): '{text}' "
                f"(no_speech_prob={no_speech_prob:.2f}, avg_logprob={avg_logprob:.2f})"
            )
            return False

        # Typical hallucinations, only when Whisper also had some doubt
        if (
            text.lower() in HALLUCINATION_PHRASES
            and no_speech_prob > HALLUCINATION_NO_SPEECH_PROB
        ):
            print(
                f"[STT] Dropped (likely hallucination): '{text}' "
                f"(no_speech_prob={no_speech_prob:.2f})"
            )
            return False

        return True

    # ==========================================================
    # Transcribe
    # ==========================================================

    def transcribe(self, frames):
        """
        Transcribe one completed speech segment.

        Returns:
            str: recognized text ("" if nothing usable)
        """

        print(f"[STT] Starting transcription for {len(frames)} frames")

        audio = self._frames_to_audio(frames)

        if audio is None or len(audio) == 0:
            print("[STT] Empty audio segment")
            return ""

        start = time.perf_counter()

        try:
            with self._lock:
                segments, _ = self.model.transcribe(
                    audio,
                    language="en",
                    beam_size=self.beam_size,
                    initial_prompt=self.initial_prompt,
                    # Each sentence is independent; do not let a wrong
                    # earlier guess push later text in the wrong direction
                    condition_on_previous_text=False,
                    # One pass only. The default retries at higher
                    # temperatures, which can triple the time on noise.
                    temperature=0.0,
                    vad_filter=False,  # our Silero VAD already did this
                )

                # transcribe() is lazy: the real work happens here,
                # so keep it inside the lock
                text_parts = [
                    segment.text.strip()
                    for segment in segments
                    if self._keep_segment(segment)
                ]

            transcript = " ".join(text_parts).strip()

            elapsed_ms = (time.perf_counter() - start) * 1000
            print(f"[STT] Transcript ({elapsed_ms:.0f} ms): {transcript}")

            return transcript

        except Exception as exc:
            print(f"[STT] Transcription error: {type(exc).__name__}: {exc}")
            return ""