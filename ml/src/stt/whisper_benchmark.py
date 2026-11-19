"""
ML Week 4 - Whisper speech-to-text benchmark on OUR voices.

PDF Week 1 asks for "a fast, local STT engine ... to transcribe microphone
input in under 200 ms". The live backend uses faster-whisper base.en and
sometimes mishears us ("AI" -> "A.A."). This script measures, on our own
recordings, which Whisper model gives the best accuracy / speed trade-off.

Data: ml/data/my_voice/<speaker>/<emotion>/<speaker>_<emotion>_<NN>.wav
      (Week 3 recordings; clip NN = sentence NN of record_my_voice.SENTENCES,
       so the correct transcript of every clip is known).

For every model x beam size it reports:
    WER          word error rate vs the true sentence (lower = better)
    exact %      clips transcribed with zero word errors
    median ms    transcription time per clip (warm model, GPU or CPU)
    p95 ms, under-200-ms %, real-time factor, load time

Outputs:
    results/week4_stt/whisper_benchmark.csv     one row per model x beam
    results/week4_stt/whisper_per_clip.csv      every clip (for error analysis)

Run from the ml/ folder with the BACKEND venv (it already has faster-whisper
and the CUDA libraries):
    "..\\..\\Auralis-Backend\\Backend\\venv\\Scripts\\python.exe" -m src.stt.whisper_benchmark
Options:
    --models tiny.en base.en small.en distil-small.en
    --beams 1 5
    --device auto|cuda|cpu
"""

import argparse
import csv
import glob
import os
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

# Same 10 sentences as src/data/record_my_voice.py (copied, because that
# script imports sounddevice, which the backend venv does not have)
SENTENCES = [
    "I am going to the shop now.",
    "The meeting is at five o'clock.",
    "I just got the message from him.",
    "We have to finish this today.",
    "That is what I was talking about.",
    "Someone is standing at the door.",
    "I did not expect this at all.",
    "It is already very late now.",
    "Tell me what happened yesterday.",
    "I think we should go home.",
]

DATA_DIR = Path("data/my_voice")
OUT_DIR = Path("results/week4_stt")

# Same prompt as the live backend (Backend/app/stt/transcriber.py)
INITIAL_PROMPT = (
    "Ankit Dash, Priya, Siddhant, BTech, CSE, Auralis, "
    "Centurion University, Bhubaneswar, Odisha."
)

DEFAULT_MODELS = ["tiny.en", "base.en", "small.en", "distil-small.en"]


# ---------------------------------------------------------
# GPU on Windows: faster-whisper needs the pip CUDA DLLs on the path
# (same trick as the backend transcriber)
# ---------------------------------------------------------
def add_cuda_dll_dirs():
    if os.name != "nt":
        return
    for base in {sys.prefix, sys.base_prefix}:
        for bin_dir in glob.glob(os.path.join(base, "Lib", "site-packages", "nvidia", "*", "bin")):
            try:
                os.add_dll_directory(bin_dir)
            except (OSError, AttributeError):
                pass
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


# ---------------------------------------------------------
# WORD ERROR RATE
# ---------------------------------------------------------
NUMBERS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
           "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten"}


def normalize(text):
    """Lowercase, no punctuation, digits as words ("5 o'clock" == "five o'clock")."""
    text = text.lower().replace("o'clock", " oclock").replace("o clock", " oclock")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    text = text.replace("'", "")
    return [NUMBERS.get(w, w) for w in text.split()]


def word_errors(ref, hyp):
    """Levenshtein distance on words -> (errors, reference length)."""
    r, h = normalize(ref), normalize(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)], len(r)


# ---------------------------------------------------------
# DATA
# ---------------------------------------------------------
def load_clips():
    clips = []
    for path in sorted(DATA_DIR.glob("*/*/*.wav")):
        m = re.search(r"_(\d+)\.wav$", path.name)
        if not m:
            continue
        idx = int(m.group(1)) - 1
        if not 0 <= idx < len(SENTENCES):
            continue
        audio, sr = sf.read(str(path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            raise SystemExit(f"{path} is {sr} Hz, expected 16000")
        clips.append({
            "speaker": path.parent.parent.name,
            "emotion": path.parent.name,
            "file": path.name,
            "audio": audio,
            "ref": SENTENCES[idx],
        })
    return clips


def pct(values, p):
    values = sorted(values)
    k = max(0, min(len(values) - 1, int(np.ceil(p / 100 * len(values))) - 1))
    return values[k]


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--beams", nargs="*", type=int, default=[1, 5])
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    add_cuda_dll_dirs()
    from faster_whisper import WhisperModel
    import ctranslate2

    device = args.device
    if device == "auto":
        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    clips = load_clips()
    if not clips:
        raise SystemExit(f"No clips in {DATA_DIR}. Run from the ml/ folder.")
    speakers = sorted({c["speaker"] for c in clips})
    print(f"[STT] {len(clips)} clips ({', '.join(speakers)}) | device {device} ({compute_type})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary, per_clip = [], []

    for name in args.models:
        start = time.perf_counter()
        try:
            model = WhisperModel(name, device=device, compute_type=compute_type)
        except Exception as exc:
            print(f"[STT] {name}: could not load ({type(exc).__name__}: {exc}) - skipped")
            continue
        load_s = time.perf_counter() - start

        for beam in args.beams:
            # warm-up (first call compiles kernels)
            list(model.transcribe(clips[0]["audio"], language="en", beam_size=beam)[0])

            errors = words = exact = 0
            times = []
            for c in clips:
                t0 = time.perf_counter()
                segments, _ = model.transcribe(
                    c["audio"], language="en", beam_size=beam,
                    initial_prompt=INITIAL_PROMPT, condition_on_previous_text=False,
                    vad_filter=False,
                )
                hyp = " ".join(s.text.strip() for s in segments).strip()
                ms = (time.perf_counter() - t0) * 1000
                e, n = word_errors(c["ref"], hyp)
                errors += e
                words += n
                exact += e == 0
                times.append(ms)
                per_clip.append({
                    "model": name, "beam": beam, "speaker": c["speaker"],
                    "emotion": c["emotion"], "file": c["file"],
                    "reference": c["ref"], "hypothesis": hyp,
                    "word_errors": e, "ms": round(ms),
                })

            audio_ms = statistics.mean(len(c["audio"]) / 16 for c in clips)
            row = {
                "model": name, "beam": beam, "device": device,
                "wer_pct": round(100 * errors / max(words, 1), 2),
                "exact_pct": round(100 * exact / len(clips), 1),
                "median_ms": round(statistics.median(times)),
                "p95_ms": round(pct(times, 95)),
                "under_200ms_pct": round(100 * sum(t < 200 for t in times) / len(times), 1),
                "rtf": round(statistics.median(times) / audio_ms, 3),
                "load_s": round(load_s, 1),
                "clips": len(clips),
            }
            summary.append(row)
            print(f"  {name:16s} beam {beam}: WER {row['wer_pct']:5.1f}% | exact {row['exact_pct']:5.1f}% "
                  f"| median {row['median_ms']:4d} ms | p95 {row['p95_ms']:4d} ms "
                  f"| <200 ms {row['under_200ms_pct']:5.1f}%")

        del model

    if not summary:
        raise SystemExit("No model could be loaded.")

    with open(OUT_DIR / "whisper_benchmark.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    with open(OUT_DIR / "whisper_per_clip.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_clip[0]))
        w.writeheader()
        w.writerows(per_clip)

    # ---- recommendation: lowest WER among configs with median <= 300 ms ----
    fast = [r for r in summary if r["median_ms"] <= 300] or summary
    best = min(fast, key=lambda r: (r["wer_pct"], r["median_ms"]))
    live = next((r for r in summary if r["model"] == "base.en" and r["beam"] == 5), None)

    print("\n" + "=" * 78)
    print(f"Whisper on our voices ({len(clips)} clips, {device})")
    print("=" * 78)
    print(f"{'model':16s} {'beam':>4s} {'WER %':>7s} {'exact %':>8s} {'median':>7s} {'p95':>6s} {'<200ms':>7s}")
    for r in sorted(summary, key=lambda r: (r["wer_pct"], r["median_ms"])):
        print(f"{r['model']:16s} {r['beam']:>4d} {r['wer_pct']:>7.1f} {r['exact_pct']:>8.1f} "
              f"{r['median_ms']:>6d}  {r['p95_ms']:>5d}  {r['under_200ms_pct']:>6.1f}")
    if live:
        print(f"\nlive backend now (base.en, beam 5): WER {live['wer_pct']}%, median {live['median_ms']} ms")
    print(f"recommended (best WER with median <= 300 ms): {best['model']}, beam {best['beam']} "
          f"-> WER {best['wer_pct']}%, median {best['median_ms']} ms")
    print(f"  backend .env: WHISPER_MODEL={best['model']}  WHISPER_BEAM_SIZE={best['beam']}")
    print(f"\nSaved {OUT_DIR / 'whisper_benchmark.csv'} and whisper_per_clip.csv")


if __name__ == "__main__":
    main()
