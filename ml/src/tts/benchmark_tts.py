"""
Week 3 Day 1 (ML): TTS benchmark on CPU.

Compares two fast, local, open-source TTS engines:
    1. Piper   (VITS, ONNX)          - en_US-lessac-medium
    2. Kokoro  (82M params, ONNX int8) - voice af_heart

For each engine and each test sentence we measure:
    synth_ms   : time to generate the audio for ONE sentence
                 (= "time to first audio" when the backend speaks
                  the LLM reply sentence by sentence)
    audio_ms   : length of the generated speech
    rtf        : real-time factor = synth_ms / audio_ms
                 (< 1.0 means faster than real time -> streamable)

Outputs:
    results/week3_tts/tts_benchmark.csv
    results/week3_tts/<engine>_<n>.wav   (listen to judge quality)

Run from the ml/ folder:
    python -m src.tts.benchmark_tts

Model files (NOT in git, download once into models/tts/):
    Piper : en_US-lessac-medium.onnx + en_US-lessac-medium.onnx.json
    Kokoro: kokoro-v1.0.int8.onnx + voices-v1.0.bin
"""

import csv
import time
from pathlib import Path

import numpy as np
import soundfile as sf


MODEL_DIR = Path("models/tts")
RESULTS_DIR = Path("results/week3_tts")

PIPER_MODEL = MODEL_DIR / "en_US-lessac-medium.onnx"
KOKORO_MODEL = MODEL_DIR / "kokoro-v1.0.int8.onnx"
KOKORO_VOICES = MODEL_DIR / "voices-v1.0.bin"
KOKORO_VOICE = "af_heart"

# Short, medium and long sentences, like real LLM reply sentences
SENTENCES = [
    "I hear you.",
    "Take a slow breath, I am right here with you.",
    "Let's go through this together, one step at a time, "
    "and I promise we will find a way to make it better.",
]

RUNS_PER_SENTENCE = 3


# ----------------------------------------------------------
# ENGINES
# ----------------------------------------------------------

class PiperEngine:
    name = "piper"

    def __init__(self):
        from piper import PiperVoice

        self.voice = PiperVoice.load(str(PIPER_MODEL))
        self.sample_rate = self.voice.config.sample_rate

    def synthesize(self, text):
        chunks = [c.audio_float_array for c in self.voice.synthesize(text)]
        return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)


class KokoroEngine:
    name = "kokoro"

    def __init__(self):
        from kokoro_onnx import Kokoro

        self.model = Kokoro(str(KOKORO_MODEL), str(KOKORO_VOICES))
        self.sample_rate = 24000

    def synthesize(self, text):
        audio, sample_rate = self.model.create(
            text, voice=KOKORO_VOICE, speed=1.0, lang="en-us"
        )
        self.sample_rate = sample_rate
        return audio


ENGINES = [
    (PiperEngine, [PIPER_MODEL]),
    (KokoroEngine, [KOKORO_MODEL, KOKORO_VOICES]),
]


# ----------------------------------------------------------
# BENCHMARK
# ----------------------------------------------------------

def benchmark(engine_cls):
    t0 = time.perf_counter()
    engine = engine_cls()
    load_ms = (time.perf_counter() - t0) * 1000

    # Warm-up: the first call is always slower (ONNX graph setup)
    t0 = time.perf_counter()
    engine.synthesize("Warming up.")
    warmup_ms = (time.perf_counter() - t0) * 1000

    print(f"\n[{engine.name}] load {load_ms:.0f} ms, warm-up {warmup_ms:.0f} ms")

    rows = []

    for index, text in enumerate(SENTENCES, start=1):
        times = []
        audio = None

        for _ in range(RUNS_PER_SENTENCE):
            t0 = time.perf_counter()
            audio = engine.synthesize(text)
            times.append((time.perf_counter() - t0) * 1000)

        synth_ms = float(np.median(times))
        audio_ms = len(audio) / engine.sample_rate * 1000
        rtf = synth_ms / audio_ms if audio_ms else float("nan")

        wav_path = RESULTS_DIR / f"{engine.name}_{index}.wav"
        sf.write(wav_path, audio, engine.sample_rate)

        print(
            f"  sentence {index} ({len(text.split())} words): "
            f"synth {synth_ms:.0f} ms, audio {audio_ms:.0f} ms, RTF {rtf:.2f}"
        )

        rows.append({
            "engine": engine.name,
            "sentence": index,
            "words": len(text.split()),
            "load_ms": round(load_ms),
            "warmup_ms": round(warmup_ms),
            "synth_ms": round(synth_ms),
            "audio_ms": round(audio_ms),
            "rtf": round(rtf, 3),
            "sample_rate": engine.sample_rate,
            "wav": wav_path.as_posix(),
        })

    return rows


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for engine_cls, files in ENGINES:
        missing = [f.name for f in files if not f.exists()]
        if missing:
            print(f"\n[{engine_cls.name}] SKIPPED, missing in {MODEL_DIR}: {', '.join(missing)}")
            continue

        try:
            all_rows.extend(benchmark(engine_cls))
        except ImportError as exc:
            print(f"\n[{engine_cls.name}] SKIPPED, library not installed: {exc}")

    if not all_rows:
        print("\nNothing was benchmarked.")
        return

    csv_path = RESULTS_DIR / "tts_benchmark.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    print("\nSummary (median over sentences)")
    print("=" * 52)
    for name in sorted({r["engine"] for r in all_rows}):
        rows = [r for r in all_rows if r["engine"] == name]
        first = rows[0]["synth_ms"]
        rtf = np.median([r["rtf"] for r in rows])
        print(f"  {name:8s} first short sentence {first:5d} ms | median RTF {rtf:.2f}")

    print(f"\nSaved {csv_path} and WAV samples in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
