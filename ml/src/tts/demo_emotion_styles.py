"""
ML Day 2 - demo: speak the same reply in each emotion style.

Run from the ml/ folder:
    python -m src.tts.demo_emotion_styles
Listen to results/week3_tts/style_*.wav to hear the difference.
"""

import time
from pathlib import Path

import numpy as np
import soundfile as sf
from piper import PiperVoice

from src.tts.emotion_style import apply_voice_effects, piper_config, voice_style


PIPER_MODEL = Path("models/tts/en_US-lessac-medium.onnx")
OUT_DIR = Path("results/week3_tts")

SENTENCE = "I hear you. Take a slow breath, we will sort this out together."

EXAMPLES = [
    ("angry", 0.82),
    ("sad", 0.71),
    ("happy", 0.77),
    ("fear", 0.30),
]


def main():
    voice = PiperVoice.load(str(PIPER_MODEL))
    sample_rate = voice.config.sample_rate
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    list(voice.synthesize("Warming up."))

    for label, confidence in EXAMPLES:
        group, style = voice_style(label, confidence)

        t0 = time.perf_counter()
        chunks = voice.synthesize(SENTENCE, syn_config=piper_config(style))
        audio = np.concatenate([c.audio_float_array for c in chunks])
        synth_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        audio = apply_voice_effects(audio, sample_rate, style)
        fx_ms = (time.perf_counter() - t1) * 1000

        path = OUT_DIR / f"style_{label}_{group}.wav"
        sf.write(path, audio, sample_rate)

        print(f"{label:8s} conf {confidence:.2f} -> {group:5s} | "
              f"synth {synth_ms:4.0f} ms + pitch {fx_ms:4.0f} ms | "
              f"audio {len(audio) / sample_rate * 1000:5.0f} ms | {path}")

    print("\nLLM hints:")
    for group in ("upset", "low", "okay"):
        from src.tts.emotion_style import STYLES
        print(f"  {group:5s}: {STYLES[group]['llm_hint']}")


if __name__ == "__main__":
    main()
