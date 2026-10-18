"""
ML Week 3 Day 3 - Step 2: WavLM embeddings for our own recorded clips.

Uses exactly the same audio loading (16 kHz, trim, max 6 s) and the same
embedding (mean over time of every hidden layer) as the CREMA-D features
in extract_ssl_embeddings.py, so the two can be trained together.

Run from the ml/ folder:
    python -m src.features.extract_my_voice --model wavlm-large

Input : data/my_voice/<speaker>/<emotion>/*.wav
Output: data/features/ssl/<model>_myvoice.npz   (X, y, speaker, files)
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModel

from src.features.extract_ssl_embeddings import MODELS, OUT_DIR, embed, load_audio


VOICE_DIR = Path("data/my_voice")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, default="wavlm-large")
    args = parser.parse_args()

    files = sorted(VOICE_DIR.glob("*/*/*.wav"))
    if not files:
        raise SystemExit(f"No clips found in {VOICE_DIR}/<speaker>/<emotion>/ - record first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = MODELS[args.model]
    print(f"[MYVOICE] {len(files)} clips, {name} on {device}")

    extractor = AutoFeatureExtractor.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(device).eval()

    features, labels, speakers, names = [], [], [], []
    skipped = 0

    for i, path in enumerate(files, start=1):
        audio = load_audio(str(path))
        if len(audio) < 16000 // 4:        # < 0.25 s of sound after trimming
            skipped += 1
            print(f"  skipped (almost silent): {path}")
            continue

        features.append(embed(model, extractor, audio, device).astype(np.float16))
        labels.append(path.parent.name)            # emotion folder
        speakers.append(path.parent.parent.name)   # speaker folder
        names.append(path.name)

        if i % 20 == 0 or i == len(files):
            print(f"  {i}/{len(files)} clips")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.model}_myvoice.npz"
    np.savez_compressed(
        out,
        X=np.stack(features),
        y=np.array(labels),
        speaker=np.array(speakers),
        files=np.array(names),
    )

    counts = {}
    for s, e in zip(speakers, labels):
        counts.setdefault(s, {}).setdefault(e, 0)
        counts[s][e] += 1
    for s, per in counts.items():
        print(f"  {s}: " + ", ".join(f"{e}={n}" for e, n in sorted(per.items())))
    print(f"[MYVOICE] saved {out}  ({len(features)} clips, {skipped} skipped)")


if __name__ == "__main__":
    main()
