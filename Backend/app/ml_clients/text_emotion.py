
"""
Week 3 Day 5: text emotion model + voice/text fusion (live backend).

From Priya's ML Day 5 (ml/src/models/text_emotion.py):
  - j-hartmann/emotion-english-distilroberta-base, ~30 ms on CPU
  - 60-sentence test: 90% accuracy, 9/10 neutral recording sentences stay neutral
  - fusion rule: 5/6 live cases correct

Why: the voice model alone tops out around 55% on our voices and misses
obvious cases ("I'm very sad today" -> neutral 93%). The words often show
the emotion when the tone doesn't.

Fusion rule:
  text top label is NOT neutral and >= TEXT_STRONG -> words decide
      final = 0.7 x text + 0.3 x voice
  otherwise -> voice decides (text only nudges)
      final = 0.2 x text + 0.8 x voice

Env:
  TEXT_EMOTION=0          turn the text model off (voice only, as before)
  TEXT_EMOTION_MODEL=...  other Hugging Face model id
"""

import os
import threading
import time


MODEL_NAME = os.getenv("TEXT_EMOTION_MODEL", "j-hartmann/emotion-english-distilroberta-base")
ENABLED = os.getenv("TEXT_EMOTION", "1") == "1"

LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad"]
MODEL_TO_OURS = {
    "anger": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "joy": "happy",
    "neutral": "neutral",
    "sadness": "sad",
    # "surprise" is dropped on purpose
}

TEXT_STRONG = 0.6          # text confidence needed for the words to decide
WORDS_WEIGHTS = (0.7, 0.3)   # (text, voice) when the words decide
VOICE_WEIGHTS = (0.2, 0.8)   # (text, voice) otherwise
MIN_WORDS = 3                # shorter transcripts ("okay", "yes") -> voice only

_pipe = None
_lock = threading.Lock()


def load_and_warmup():
    """Blocking. Call once from a thread at server start."""
    global _pipe

    if not ENABLED:
        print("[TEXT] Disabled (TEXT_EMOTION=0) - voice-only emotion")
        return

    try:
        from transformers import pipeline

        start = time.perf_counter()
        pipe = pipeline("text-classification", model=MODEL_NAME, top_k=None, device=-1)
        print(f"[TEXT] Loaded {MODEL_NAME} in {time.perf_counter() - start:.1f} s")

        start = time.perf_counter()
        for _ in range(3):
            pipe("I am fine, thank you.")
        print(f"[TEXT] Warm-up done in {(time.perf_counter() - start) * 1000:.0f} ms")

        _pipe = pipe
    except Exception as exc:
        print(f"[TEXT] Disabled: failed to load ({type(exc).__name__}: {exc})")


def is_ready():
    return _pipe is not None


def predict(text):
    """
    Returns {"label", "confidence", "probs", "ms"} over our 6 labels,
    or None if the model is off or the text is too short.
    """
    if _pipe is None or not text or len(text.split()) < MIN_WORDS:
        return None

    start = time.perf_counter()
    with _lock:
        out = _pipe(text)
    ms = round((time.perf_counter() - start) * 1000)

    if out and isinstance(out[0], list):
        out = out[0]

    scores = {MODEL_TO_OURS[d["label"]]: float(d["score"])
              for d in out if d["label"] in MODEL_TO_OURS}
    total = sum(scores.values()) or 1.0
    probs = {k: scores.get(k, 0.0) / total for k in LABELS}
    label = max(probs, key=probs.get)

    return {
        "label": label,
        "confidence": round(probs[label], 3),
        "probs": {k: round(v, 3) for k, v in probs.items()},
        "ms": ms,
    }


def fuse(voice_probs, text_probs):
    """Returns (final_label, confidence, source) where source is 'voice' or 'words'."""
    t_label = max(text_probs, key=text_probs.get)
    t_conf = text_probs[t_label]

    if t_label != "neutral" and t_conf >= TEXT_STRONG:
        (w_text, w_voice), source = WORDS_WEIGHTS, "words"
    else:
        (w_text, w_voice), source = VOICE_WEIGHTS, "voice"

    final = {k: w_text * text_probs.get(k, 0.0) + w_voice * voice_probs.get(k, 0.0)
             for k in LABELS}
    label = max(final, key=final.get)
    return label, round(final[label], 3), source
