"""
ML Week 3 Day 6 - Step 1: zero-shot EMOTIONAL TTS with XTTS-v2 (GPU).

Project plan, Week 3: "Generative TTS: integrate a zero-shot TTS model.
Condition the generation on an emotional baseline (e.g. a calm vs a
panicked voice)."  Until now Auralis speaks with Piper and only changes
speed / pitch. XTTS-v2 clones a voice AND its tone from a few seconds of
reference audio, so we condition it on our own recorded emotion clips:

    reference clips of  neutral -> "calm"      voice
                        sad     -> "soft"      voice
                        fear    -> "panicked"  voice
                        angry   -> "angry"     voice
                        happy   -> "happy"     voice

The SAME reply sentences are spoken in every style, so any difference
comes from the reference emotion only.

For every sentence we measure (streaming, like the live backend would):
    first_chunk_ms : text in -> first audio chunk out  (what the user waits)
    total_ms       : time to generate the whole sentence
    audio_ms       : length of the speech
    rtf            : total_ms / audio_ms  (< 1 = faster than real time)

Outputs:
    results/week3_tts/xtts_benchmark.csv
    results/week3_tts/xtts/<emotion>_<n>.wav   (cloned voice: NOT pushed to git)

Next: score_tts_emotion.py (ML venv) checks if our emotion model actually
hears the intended emotion in these files, compared with Piper's styles.

Run from the ml/ folder, in the SEPARATE tts venv (see README / chat):
    python -m src.tts.xtts_emotion_test --speaker ankit
"""

import argparse
import csv
import os
import statistics
import time
from pathlib import Path

import numpy as np
import soundfile as sf

# XTTS-v2 (~1.9 GB) is downloaded from Hugging Face -> set HF_HOME to a D:
# folder before running so it doesn't fill C:. TTS_HOME covers older
# coqui versions. The XTTS-v2 licence (Coqui Public Model License,
# non-commercial) is accepted for this internship project.
os.environ.setdefault("TTS_HOME", str(Path("models/tts/coqui").resolve()))
os.environ.setdefault("COQUI_TOS_AGREED", "1")

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
VOICE_DIR = Path("data/my_voice")
OUT_DIR = Path("results/week3_tts/xtts")
CSV_PATH = Path("results/week3_tts/xtts_benchmark.csv")

# Auralis voice style -> which recorded emotion is used as the reference
STYLES = {
    "calm": "neutral",
    "soft": "sad",
    "panicked": "fear",
    "angry": "angry",
    "happy": "happy",
}

# Typical Auralis replies (same text in every style)
SENTENCES = [
    "I'm here with you. Take a slow breath, we will sort this out together.",
    "That sounds really hard. Do you want to tell me what happened?",
    "That's wonderful news, I'm so happy for you!",
]

REF_CLIPS = 3          # how many 4 s clips are joined as the voice reference
PIPER_FIRST_MS = "about 100-400 ms per sentence (backend latency log)"


def reference_clips(speaker, emotion):
    clips = sorted((VOICE_DIR / speaker / emotion).glob("*.wav"))
    if not clips:
        raise SystemExit(f"No clips in {VOICE_DIR / speaker / emotion} - record them first.")
    # the loudest clips make the clearest reference
    clips.sort(key=lambda p: float(np.abs(sf.read(p)[0]).max()), reverse=True)
    return [str(p) for p in clips[:REF_CLIPS]]


def synth_stream(model, text, latents):
    """Streams one sentence. Returns (audio, first_chunk_ms, total_ms)."""
    gpt_cond_latent, speaker_embedding = latents
    start = time.perf_counter()
    first_ms = None
    chunks = []
    for chunk in model.inference_stream(text, "en", gpt_cond_latent, speaker_embedding):
        if first_ms is None:
            first_ms = (time.perf_counter() - start) * 1000
        chunks.append(chunk.detach().cpu().numpy().reshape(-1))
    total_ms = (time.perf_counter() - start) * 1000
    return np.concatenate(chunks), first_ms, total_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speaker", default="ankit", help="whose recorded clips are the reference")
    parser.add_argument("--styles", nargs="*", default=list(STYLES), choices=list(STYLES))
    args = parser.parse_args()

    import torch
    from TTS.api import TTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[XTTS] device: {device}" + ("" if device == "cuda" else "  (WARNING: CPU is very slow)"))
    print(f"[XTTS] model files in {os.environ['TTS_HOME']}")

    start = time.perf_counter()
    model = TTS(MODEL_NAME).to(device).synthesizer.tts_model
    load_s = time.perf_counter() - start
    sr = model.config.audio.output_sample_rate
    print(f"[XTTS] loaded in {load_s:.1f} s (output {sr} Hz)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    warmed_up = False

    for style in args.styles:
        emotion = STYLES[style]
        refs = reference_clips(args.speaker, emotion)
        latents = model.get_conditioning_latents(audio_path=refs)

        if not warmed_up:   # first call compiles CUDA kernels; don't time it
            synth_stream(model, "Hello, this is a warm up.", latents)
            warmed_up = True

        for n, text in enumerate(SENTENCES, start=1):
            audio, first_ms, total_ms = synth_stream(model, text, latents)
            audio_ms = len(audio) / sr * 1000
            out = OUT_DIR / f"{emotion}_{n}.wav"
            sf.write(out, audio, sr)

            rows.append({
                "style": style, "reference_emotion": emotion, "sentence": n,
                "first_chunk_ms": round(first_ms), "total_ms": round(total_ms),
                "audio_ms": round(audio_ms), "rtf": round(total_ms / audio_ms, 2),
                "file": out.name, "text": text,
            })
            print(f"  {style:8s} #{n}: first chunk {first_ms:6.0f} ms | total {total_ms:6.0f} ms"
                  f" | audio {audio_ms:6.0f} ms | rtf {total_ms / audio_ms:.2f}")

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    firsts = [r["first_chunk_ms"] for r in rows]
    rtfs = [r["rtf"] for r in rows]
    vram = torch.cuda.max_memory_allocated() / 1024 ** 3 if device == "cuda" else 0.0

    print("\n" + "=" * 70)
    print(f"XTTS-v2 on {device}  ({len(rows)} sentences, reference speaker '{args.speaker}')")
    print("=" * 70)
    print(f"  first audio chunk : median {statistics.median(firsts):.0f} ms, max {max(firsts)} ms")
    print(f"  real-time factor  : median {statistics.median(rtfs):.2f}  (< 1 = streamable)")
    print(f"  model load        : {load_s:.1f} s")
    if device == "cuda":
        print(f"  peak GPU memory   : {vram:.2f} GB")
    print(f"  Piper (current)   : {PIPER_FIRST_MS}")
    print("=" * 70)
    print(f"Saved {CSV_PATH} and {len(rows)} wav files in {OUT_DIR}/  -> LISTEN to them")


if __name__ == "__main__":
    main()
