"""
Week 3 (ML): extract self-supervised speech embeddings for emotion.

Instead of hand-made MFCC/prosody features (Day 7 SVM, 49% test accuracy),
we use a pretrained speech model (WavLM / Wav2Vec2) as a feature extractor.
For every clip we average each hidden layer over time -> one 768-number
vector per layer (13 layers). train_ssl_classifier.py picks the best layer.

Run from the ml/ folder:
    python -m src.features.extract_ssl_embeddings --model wavlm --limit 30   (quick test)
    python -m src.features.extract_ssl_embeddings --model wavlm             (full run)

Output: data/features/ssl/<model>_<split>.npz
"""

import argparse
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from transformers import AutoFeatureExtractor, AutoModel


MODELS = {
    "wav2vec2": "facebook/wav2vec2-base",
    "wavlm": "microsoft/wavlm-base-plus",
    "wavlm-large": "microsoft/wavlm-large",
}

SPLIT_DIR = Path("data/metadata/splits")
OUT_DIR = Path("data/features/ssl")

SAMPLE_RATE = 16000
MAX_SECONDS = 6.0


def load_audio(path):
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    audio, _ = librosa.effects.trim(audio, top_db=30)
    return audio[: int(MAX_SECONDS * SAMPLE_RATE)]


@torch.inference_mode()
def embed(model, extractor, audio, device):
    inputs = extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs, output_hidden_states=True)
    layers = torch.stack(outputs.hidden_states, dim=0)[:, 0]
    return layers.mean(dim=1).float().cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, default="wavlm")
    parser.add_argument("--limit", type=int, default=0,
                        help="only process about N clips per split (quick test)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = MODELS[args.model]
    print(f"[SSL] {name} on {device}")

    extractor = AutoFeatureExtractor.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(device).eval()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in ("train", "validation", "test"):
        df = pd.read_csv(SPLIT_DIR / f"{split}.csv")
        if args.limit:
            df = df.groupby("emotion").head(max(1, args.limit // 6))

        features = []
        start = time.perf_counter()

        for i, row in enumerate(df.itertuples(index=False), start=1):
            audio = load_audio(row.audio_path)
            features.append(embed(model, extractor, audio, device).astype(np.float16))

            if i % 250 == 0 or i == len(df):
                rate = (time.perf_counter() - start) / i
                left = rate * (len(df) - i) / 60
                print(f"  {split}: {i}/{len(df)} clips  (~{left:.1f} min left)")

        X = np.stack(features)
        out = OUT_DIR / f"{args.model}_{split}.npz"
        np.savez_compressed(
            out,
            X=X,
            y=df["emotion"].to_numpy().astype(str),
            actor=df["actor_id"].to_numpy().astype(int),
            files=df["filename"].to_numpy().astype(str),
        )
        print(f"[SSL] saved {out}  shape={X.shape}")


if __name__ == "__main__":
    main()
