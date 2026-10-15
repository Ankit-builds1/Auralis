"""
Live emotion recognition for Auralis (Week 3 Day 2).

Model: WavLM-Large (frozen) + logistic-regression classifier trained on
CREMA-D in the ML branch (test acc 74.6%, 3-group 80.7%, ~426 ms on CPU).

Needs:  Backend/models/ssl_wavlm-large_emotion.joblib   (not in git)
WavLM-Large weights load from the HuggingFace cache (HF_HOME).

If the model file is missing or fails to load, emotion is disabled and
the rest of the pipeline (STT -> LLM) keeps working.

Live test finding (Week 3 Day 2): the audio pipeline is correct
(16 kHz float32, normal levels), but the CREMA-D model is often
confidently wrong on our voices (domain shift). So only very confident
predictions (>= 0.80) are passed to the LLM; everything else is "okay".
The real fix (speaker adaptation with our own clips) is an ML task.

DEBUG mode (off by default; turn on with EMOTION_DEBUG=1):
  - saves the exact audio the model hears to Backend/debug_emotion/*.wav
  - prints audio stats (length, min/max, rms)
  - prints the probability of every emotion
"""

import os
import threading
import time
from pathlib import Path

import numpy as np


MODEL_KEY = os.getenv("EMOTION_MODEL", "wavlm-large")
BACKEND_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(os.getenv("EMOTION_MODEL_DIR", BACKEND_DIR / "models"))

# CPU threads for WavLM. Half the cores, so Whisper (running at the
# same time) keeps the other half.
EMOTION_THREADS = int(os.getenv("EMOTION_THREADS", max(1, (os.cpu_count() or 2) // 2)))

SAMPLE_RATE = 16000
MAX_SECONDS = 6.0
MIN_SAMPLES = SAMPLE_RATE // 4          # < 0.25 s of speech: too short to judge

# Below this -> treat as neutral ("okay"). Raised from 0.45 after the live
# test showed confident mistakes on our voices.
MIN_CONFIDENCE = float(os.getenv("EMOTION_MIN_CONFIDENCE", "0.80"))

# Debug: save audio + print all probabilities (EMOTION_DEBUG=1 to turn on)
DEBUG = os.getenv("EMOTION_DEBUG", "0") == "1"
DEBUG_DIR = BACKEND_DIR / "debug_emotion"

GROUPS = {
    "angry": "upset", "disgust": "upset", "fear": "upset",
    "sad": "low",
    "happy": "okay", "neutral": "okay",
}

LLM_HINTS = {
    "upset": "The user sounds upset or stressed. Stay calm and reassuring. "
             "Use short, slow, simple sentences.",
    "low": "The user sounds sad or low. Be gentle, warm and supportive.",
    "okay": "The user sounds fine. Be friendly and natural.",
}


# ---------------------------------------------------------
# DEBUG HELPERS
# ---------------------------------------------------------

def _audio_stats(x):
    if x.size == 0:
        return "empty"
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    return (f"sec={x.size / SAMPLE_RATE:.2f} min={x.min():.3f} "
            f"max={x.max():.3f} rms={rms:.4f}")


def _debug_save(raw, tag):
    """Save the audio exactly as the emotion model receives it."""
    try:
        import soundfile as sf

        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        name = DEBUG_DIR / f"turn_{time.strftime('%H%M%S')}_{tag}.wav"
        sf.write(str(name), np.clip(raw, -1.0, 1.0), SAMPLE_RATE)
        print(f"[EMOTION-DEBUG] saved {name}")
    except Exception as exc:
        print(f"[EMOTION-DEBUG] could not save wav ({type(exc).__name__}: {exc})")


# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------

class _EmotionModel:
    def __init__(self, path):
        import joblib
        import torch
        from transformers import AutoFeatureExtractor, AutoModel

        torch.set_num_threads(EMOTION_THREADS)

        bundle = joblib.load(path)
        self.torch = torch
        self.layer = bundle["layer"]
        self.scaler = bundle["scaler"]
        self.classifier = bundle["classifier"]
        self.labels = list(self.classifier.classes_)
        self.ssl_name = bundle["ssl_model"]

        self.extractor = AutoFeatureExtractor.from_pretrained(self.ssl_name)
        self.model = AutoModel.from_pretrained(self.ssl_name).eval()

        # Keep only the layers up to the chosen one (faster, same features)
        if self.layer != "avg":
            keep = int(self.layer) + 1
            self.model.encoder.layers = self.model.encoder.layers[:keep]

    def predict(self, audio, debug=False):
        """Returns (label, confidence, probs_dict) or None if too short."""
        import librosa

        raw = np.asarray(audio)
        input_dtype = raw.dtype

        audio = raw.astype(np.float32).reshape(-1)
        if input_dtype == np.int16:
            # Safety: model expects -1..1 floats, not int16 numbers
            audio = audio / 32768.0

        if debug:
            print(f"[EMOTION-DEBUG] input dtype={input_dtype} shape={raw.shape} "
                  f"{_audio_stats(audio)}")
            _debug_save(audio, "raw")

        audio, _ = librosa.effects.trim(audio, top_db=30)
        audio = audio[: int(MAX_SECONDS * SAMPLE_RATE)]

        if debug:
            print(f"[EMOTION-DEBUG] after trim {_audio_stats(audio)}")

        if len(audio) < MIN_SAMPLES:
            if debug:
                print("[EMOTION-DEBUG] too short after trim -> skipped")
            return None

        with self.torch.inference_mode():
            inputs = self.extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            outputs = self.model(**inputs, output_hidden_states=True)

            if self.layer == "avg":
                layers = self.torch.stack(outputs.hidden_states, dim=0)[:, 0]
                vector = layers.mean(dim=1).mean(dim=0)
            else:
                vector = outputs.hidden_states[int(self.layer)][0].mean(dim=0)

        features = vector.float().numpy().astype(np.float16).astype(np.float32)
        probs = self.classifier.predict_proba(
            self.scaler.transform(features.reshape(1, -1))
        )[0]

        prob_dict = {label: float(p) for label, p in zip(self.labels, probs)}
        best = int(np.argmax(probs))

        if debug:
            ranked = sorted(prob_dict.items(), key=lambda kv: kv[1], reverse=True)
            text = "  ".join(f"{k}={v * 100:.0f}%" for k, v in ranked)
            print(f"[EMOTION-DEBUG] probs: {text}")

        return self.labels[best], float(probs[best]), prob_dict


_model = None
_lock = threading.Lock()


def load_and_warmup():
    """Blocking. Call once from a thread at server start."""
    global _model

    path = MODEL_DIR / f"ssl_{MODEL_KEY}_emotion.joblib"
    if not path.exists():
        print(f"[EMOTION] Disabled: model file not found ({path})")
        return

    try:
        start = time.perf_counter()
        model = _EmotionModel(path)
        print(f"[EMOTION] Loaded {model.ssl_name} (layer {model.layer}, "
              f"{EMOTION_THREADS} threads, min confidence {MIN_CONFIDENCE}) "
              f"in {time.perf_counter() - start:.1f} s")

        warm = np.random.default_rng(0).normal(0, 0.05, SAMPLE_RATE).astype(np.float32)
        start = time.perf_counter()
        model.predict(warm, debug=False)
        print(f"[EMOTION] Warm-up done in {(time.perf_counter() - start) * 1000:.0f} ms")

        if DEBUG:
            print(f"[EMOTION] DEBUG on: wav files -> {DEBUG_DIR}")

        _model = model
    except Exception as exc:
        print(f"[EMOTION] Disabled: failed to load ({type(exc).__name__}: {exc})")


def is_ready():
    return _model is not None


def predict(audio):
    """
    audio: 16 kHz mono float32 waveform.
    Returns a dict, or None if emotion is disabled / audio too short.
    """
    if _model is None or audio is None or len(audio) == 0:
        return None

    start = time.perf_counter()
    with _lock:
        result = _model.predict(audio, debug=DEBUG)
    ms = round((time.perf_counter() - start) * 1000)

    if result is None:
        return None

    label, confidence, probs = result
    confident = confidence >= MIN_CONFIDENCE
    group = GROUPS.get(label, "okay") if confident else "okay"

    if DEBUG:
        print(f"[EMOTION-DEBUG] result: {label} {confidence * 100:.0f}% "
              f"-> group {group} (confident={confident}) in {ms} ms")

    return {
        "label": label,
        "confidence": round(confidence, 3),
        "group": group,
        "confident": confident,
        "hint": LLM_HINTS[group],
        "ms": ms,
        "probs": {k: round(v, 3) for k, v in probs.items()},
    }