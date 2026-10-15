"""
Piper text-to-speech for Auralis (Week 3 Day 3, faster first audio Day 4).

Why Piper: the ML branch benchmark (Week 3 Day 1) measured Piper
en_US-lessac-medium at ~31 ms (GPU laptop) / ~200 ms (CPU laptop) per short
sentence, far faster than real time, so replies can be spoken sentence by
sentence while Llama is still writing.

Emotion -> voice style (from ml/src/tts/emotion_style.py, ML Day 2):
    upset -> calm : slower, flatter, a little quieter, pitch -1 semitone
    low   -> soft : slowest, flatter, quieter,         pitch -2 semitones
    okay  -> normal

Needs (NOT in git, copy from the ML folder):
    Backend/models/tts/en_US-lessac-medium.onnx
    Backend/models/tts/en_US-lessac-medium.onnx.json

If the voice files are missing, TTS is disabled and the rest of the
pipeline (STT -> emotion -> LLM text) keeps working.
"""

import os
import re
import threading
import time
from pathlib import Path

import numpy as np


BACKEND_DIR = Path(__file__).resolve().parents[2]
VOICE_PATH = Path(os.getenv(
    "TTS_VOICE",
    BACKEND_DIR / "models" / "tts" / "en_US-lessac-medium.onnx",
))

STYLES = {
    "upset": {
        "name": "calm",
        "length_scale": 1.3,
        "noise_scale": 0.4,
        "noise_w_scale": 0.5,
        "volume": 0.85,
        "pitch_steps": -1.0,
    },
    "low": {
        "name": "soft",
        "length_scale": 1.4,
        "noise_scale": 0.4,
        "noise_w_scale": 0.5,
        "volume": 0.7,
        "pitch_steps": -2.0,
    },
    "okay": {
        "name": "normal",
        "length_scale": 1.0,
        "noise_scale": 0.667,
        "noise_w_scale": 0.8,
        "volume": 1.0,
        "pitch_steps": 0.0,
    },
}

WARMUP_TEXT = "Hello, I am here with you."

# Week 3 Day 4: the first spoken piece may end at a comma once it has at
# least this many characters (0 = always wait for a full sentence).
FIRST_CLAUSE_CHARS = int(os.getenv("TTS_FIRST_CLAUSE_CHARS", "20"))

_voice = None
_sample_rate = 22050
_lock = threading.Lock()   # one Piper (ONNX) session, one sentence at a time


# ---------------------------------------------------------
# LOAD
# ---------------------------------------------------------

def load_and_warmup():
    """Blocking. Call once from a thread at server start."""
    global _voice, _sample_rate

    if not VOICE_PATH.exists():
        print(f"[TTS] Disabled: voice file not found ({VOICE_PATH})")
        return

    try:
        from piper import PiperVoice

        start = time.perf_counter()
        voice = PiperVoice.load(str(VOICE_PATH))
        _sample_rate = int(voice.config.sample_rate)
        _voice = voice
        print(f"[TTS] Loaded Piper {VOICE_PATH.stem} ({_sample_rate} Hz) "
              f"in {time.perf_counter() - start:.1f} s")

        # Warm up every style. The first librosa pitch shift compiles
        # code (~10 s once); doing it here keeps it out of the first reply.
        for group, style in STYLES.items():
            start = time.perf_counter()
            synthesize(WARMUP_TEXT, group)
            print(f"[TTS] Warm-up '{style['name']}' voice: "
                  f"{(time.perf_counter() - start) * 1000:.0f} ms")

    except Exception as exc:
        _voice = None
        print(f"[TTS] Disabled: failed to load ({type(exc).__name__}: {exc})")


def is_ready():
    return _voice is not None


def sample_rate():
    return _sample_rate


def style_name(group):
    return STYLES.get(group, STYLES["okay"])["name"]


# ---------------------------------------------------------
# TEXT -> AUDIO
# ---------------------------------------------------------

def clean_text(text):
    """Remove things a voice should not read out (markdown, emojis)."""
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"[*_#`~>\[\]{}|]", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)      # emojis, symbols
    return re.sub(r"\s+", " ", text).strip()


def synthesize(text, group="okay"):
    """
    Blocking (run it in a thread). Returns float32 mono audio in -1..1 at
    sample_rate(), or an empty array if TTS is off / nothing to say.
    """
    if _voice is None:
        return np.zeros(0, dtype=np.float32)

    text = clean_text(text)
    if not re.search(r"[A-Za-z0-9]", text):
        return np.zeros(0, dtype=np.float32)

    from piper import SynthesisConfig

    style = STYLES.get(group, STYLES["okay"])
    config = SynthesisConfig(
        length_scale=style["length_scale"],
        noise_scale=style["noise_scale"],
        noise_w_scale=style["noise_w_scale"],
        volume=style["volume"],
    )

    with _lock:
        chunks = [c.audio_float_array for c in _voice.synthesize(text, syn_config=config)]

    if not chunks:
        return np.zeros(0, dtype=np.float32)

    audio = np.concatenate(chunks).astype(np.float32)

    steps = style["pitch_steps"]
    if steps and audio.size:
        import librosa

        audio = librosa.effects.pitch_shift(audio, sr=_sample_rate, n_steps=steps)

    return np.clip(audio, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------
# STREAMING: LLM tokens -> whole sentences
# ---------------------------------------------------------

class SentenceSplitter:
    """
    Collects streamed LLM tokens and returns pieces to speak as soon as
    they are complete, so Auralis can start talking while Llama is still
    writing the rest.

    Week 3 Day 4 (faster first audio): the FIRST piece may end at a comma
    (, ; :) once it is at least `first_clause_chars` long. A long first
    sentence like "It's lovely to meet you, Ankit, I'm here to listen ..."
    is then spoken from the first comma instead of waiting for the full
    stop. Every later piece is a whole sentence (more natural rhythm).

        feed(" I hear you.")  -> []              (end not confirmed yet)
        feed(" Take")         -> ["I hear you."]
        flush()               -> ["Take ..."]    (whatever is left)
    """

    # . ! ? (optionally followed by a quote/bracket) and then whitespace
    _END = re.compile(r"[.!?]+[\"')\]]*\s+")
    # , ; : and then whitespace  (only used for the first piece)
    _CLAUSE = re.compile(r"[,;:]\s+")

    def __init__(self, min_chars=2, first_clause_chars=FIRST_CLAUSE_CHARS):
        self.min_chars = min_chars
        self.first_clause_chars = first_clause_chars
        self.buffer = ""
        self.emitted = 0

    def _take(self, end):
        piece = self.buffer[:end].strip()
        self.buffer = self.buffer[end:]
        if len(piece) >= self.min_chars:
            self.emitted += 1
            return piece
        return None

    def feed(self, token):
        self.buffer += token
        pieces = []

        while True:
            match = self._END.search(self.buffer)

            if match:
                piece = self._take(match.end())
                if piece:
                    pieces.append(piece)
                continue

            # No full sentence yet: the first piece may end at a comma
            if self.emitted == 0 and self.first_clause_chars > 0:
                clause = next(
                    (m for m in self._CLAUSE.finditer(self.buffer)
                     if m.start() >= self.first_clause_chars),
                    None,
                )
                if clause:
                    piece = self._take(clause.end())
                    if piece:
                        pieces.append(piece)
                    continue

            break

        return pieces

    def flush(self):
        rest = self.buffer.strip()
        self.buffer = ""
        if len(rest) >= self.min_chars:
            self.emitted += 1
            return [rest]
        return []