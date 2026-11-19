"""
ML Week 3 Day 6 - Step 2: does the generated voice SOUND emotional?

Listens to the TTS outputs with our own voice emotion model
(WavLM-Large + adapted classifier) and checks if it hears the intended
emotion:

    XTTS-v2 : results/week3_tts/xtts/<emotion>_<n>.wav   (from xtts_emotion_test.py)
    Piper   : results/week3_tts/style_<emotion>_<group>.wav  (Week 3 Day 2 styles:
              same voice, only speed / pitch / volume changed)

Report per engine:
    hit rate          : predicted emotion == intended emotion
    3-group hit rate  : upset / low / okay matches
    mean p(intended)  : how much probability the model gives the intended emotion

This is an objective check next to listening: if the classifier hears
"fear" in the panicked XTTS voice but "neutral" in Piper's, XTTS really
carries the emotion.

Run from the ml/ folder, in the normal ML venv:
    python -m src.tts.score_tts_emotion
"""

import argparse
import csv
from pathlib import Path

import joblib
import numpy as np

from src.features.extract_ssl_embeddings import MODELS, embed, load_audio
from src.models.train_ssl_classifier import GROUPS, pick


TTS_DIR = Path("results/week3_tts")
OUT_CSV = TTS_DIR / "tts_emotion_scores.csv"


def collect():
    """[(engine, intended_emotion, path)]"""
    items = []
    for path in sorted((TTS_DIR / "xtts").glob("*.wav")):
        items.append(("xtts", path.stem.split("_")[0], path))
    for path in sorted(TTS_DIR.glob("style_*.wav")):
        items.append(("piper", path.stem.split("_")[1], path))
    return [it for it in items if it[1] in GROUPS]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="wavlm-large-adapted",
                        help="emotion model in models/ssl_<name>_emotion.joblib")
    args = parser.parse_args()

    import torch
    from transformers import AutoFeatureExtractor, AutoModel

    bundle = joblib.load(Path("models") / f"ssl_{args.model}_emotion.joblib")
    layer, scaler, clf = bundle["layer"], bundle["scaler"], bundle["classifier"]
    labels = list(clf.classes_)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = bundle["ssl_model"]
    extractor = AutoFeatureExtractor.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(device).eval()
    print(f"[SCORE] {args.model} ({name}, layer {layer}) on {device}")

    items = collect()
    if not items:
        raise SystemExit("No TTS wav files found - run xtts_emotion_test.py first.")

    rows = []
    for engine, intended, path in items:
        audio = load_audio(str(path))      # 16 kHz, trimmed, max 6 s (same as training)
        X = embed(model, extractor, audio, device).astype(np.float16).astype(np.float32)
        x = pick(X[None], layer)
        probs = clf.predict_proba(scaler.transform(x))[0]
        p = dict(zip(labels, probs))
        pred = labels[int(np.argmax(probs))]
        rows.append({
            "engine": engine, "file": path.name, "intended": intended,
            "predicted": pred, "p_predicted": round(float(p[pred]), 3),
            "p_intended": round(float(p.get(intended, 0.0)), 3),
            "hit": pred == intended, "group_hit": GROUPS[pred] == GROUPS[intended],
        })
        print(f"  {engine:5s} {path.name:28s} intended {intended:8s} -> heard {pred:8s} "
              f"({p[pred] * 100:.0f}%)  p(intended) {p.get(intended, 0) * 100:.0f}%")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 70)
    print("Does the TTS voice carry the emotion?  (judge: our voice emotion model)")
    print("=" * 70)
    for engine in ("xtts", "piper"):
        sel = [r for r in rows if r["engine"] == engine]
        if not sel:
            continue
        hit = sum(r["hit"] for r in sel) / len(sel)
        grp = sum(r["group_hit"] for r in sel) / len(sel)
        p_int = float(np.mean([r["p_intended"] for r in sel]))
        print(f"  {engine:5s}: {len(sel):2d} clips | hit {hit:.0%} | 3-group {grp:.0%} "
              f"| mean p(intended) {p_int:.0%}")
    print("  chance: 6 emotions = 17%, 3 groups = 33%")
    print("=" * 70)
    print(f"Saved {OUT_CSV}")


if __name__ == "__main__":
    main()
