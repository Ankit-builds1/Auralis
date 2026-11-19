"""
ML Day 2 - Task 1: live emotion inference + CPU speed test.

EmotionModel loads a trained classifier (models/ssl_<model>_emotion.joblib)
plus its WavLM backbone and predicts emotion from raw audio:
    {"label": "angry", "confidence": 0.82, "probs": {...}, "ms": 240}

Only the encoder layers up to the chosen layer are kept, so inference is
faster while hidden_states[layer] stays exactly the same as in training.

Speed test on real test clips (CPU, like the live backend):
    python -m src.models.ssl_inference --model wavlm --n 30
    python -m src.models.ssl_inference --model wavlm-large --n 30
"""

import argparse
import time
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
import torch
from transformers import AutoFeatureExtractor, AutoModel


SAMPLE_RATE = 16000
MAX_SECONDS = 6.0
MIN_SAMPLES = SAMPLE_RATE // 4   # less than 0.25 s of speech -> not enough to judge

MODEL_DIR = Path("models")
SPLIT_DIR = Path("data/metadata/splits")
RESULTS_DIR = Path("results/week3_ssl")


class EmotionModel:
    def __init__(self, model_key="wavlm", device="cpu"):
        bundle = joblib.load(MODEL_DIR / f"ssl_{model_key}_emotion.joblib")

        self.model_key = model_key
        self.layer = bundle["layer"]
        self.scaler = bundle["scaler"]
        self.classifier = bundle["classifier"]
        self.labels = list(self.classifier.classes_)
        self.device = device

        self.extractor = AutoFeatureExtractor.from_pretrained(bundle["ssl_model"])
        self.model = AutoModel.from_pretrained(bundle["ssl_model"]).to(device).eval()

        if self.layer != "avg":
            keep = int(self.layer) + 1
            self.model.encoder.layers = self.model.encoder.layers[:keep]

    @torch.inference_mode()
    def predict(self, audio, sample_rate=SAMPLE_RATE):
        start = time.perf_counter()

        audio = np.asarray(audio, dtype=np.float32)
        if sample_rate != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
        audio, _ = librosa.effects.trim(audio, top_db=30)
        audio = audio[: int(MAX_SECONDS * SAMPLE_RATE)]

        if len(audio) < MIN_SAMPLES:
            return {"label": "neutral", "confidence": 0.0, "probs": {},
                    "ms": round((time.perf_counter() - start) * 1000)}

        inputs = self.extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs, output_hidden_states=True)

        if self.layer == "avg":
            layers = torch.stack(outputs.hidden_states, dim=0)[:, 0]
            vector = layers.mean(dim=1).mean(dim=0)
        else:
            vector = outputs.hidden_states[int(self.layer)][0].mean(dim=0)

        features = vector.float().cpu().numpy().astype(np.float16).astype(np.float32)
        probs = self.classifier.predict_proba(self.scaler.transform(features.reshape(1, -1)))[0]
        best = int(np.argmax(probs))

        return {
            "label": self.labels[best],
            "confidence": round(float(probs[best]), 3),
            "probs": {label: round(float(p), 3) for label, p in zip(self.labels, probs)},
            "ms": round((time.perf_counter() - start) * 1000),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="wavlm")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print(f"[INFER] loading {args.model} on {args.device} ...")
    t0 = time.perf_counter()
    em = EmotionModel(args.model, args.device)
    print(f"[INFER] loaded in {time.perf_counter() - t0:.1f} s (layer {em.layer})")

    df = pd.read_csv(SPLIT_DIR / "test.csv").sample(n=args.n, random_state=0)

    warm, _ = librosa.load(df.iloc[0]["audio_path"], sr=SAMPLE_RATE, mono=True)
    em.predict(warm)

    rows = []
    for row in df.itertuples(index=False):
        audio, _ = librosa.load(row.audio_path, sr=SAMPLE_RATE, mono=True)
        result = em.predict(audio)
        rows.append({
            "file": row.filename,
            "true": row.emotion,
            "pred": result["label"],
            "confidence": result["confidence"],
            "ms": result["ms"],
            "audio_s": round(len(audio) / SAMPLE_RATE, 2),
        })
        mark = "OK " if result["label"] == row.emotion else "x  "
        print(f"  {mark}{row.emotion:8s} -> {result['label']:8s} "
              f"conf {result['confidence']:.2f}  {result['ms']} ms")

    res = pd.DataFrame(rows)
    print("\n" + "=" * 50)
    print(f"Model          : {args.model} on {args.device}")
    print(f"Clips          : {len(res)} (avg {res['audio_s'].mean():.1f} s each)")
    print(f"Accuracy (n)   : {(res['true'] == res['pred']).mean():.3f}")
    print(f"Avg time       : {res['ms'].mean():.0f} ms")
    print(f"90% under      : {res['ms'].quantile(0.9):.0f} ms")
    print(f"Slowest        : {res['ms'].max():.0f} ms")
    print("=" * 50)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{args.model}_{args.device}_speed.csv"
    res.to_csv(out, index=False)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
