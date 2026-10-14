"""
Week 3 (ML): train the emotion classifier on WavLM / Wav2Vec2 embeddings.

- Tries every layer (and the average of all layers) with a few C values
- Picks the best one on the VALIDATION set (test set is never used to choose)
- Retrains on train + validation, then reports TEST accuracy once
- Also reports 3-group accuracy: upset (angry/disgust/fear), low (sad),
  okay (happy/neutral)

Run from the ml/ folder:
    python -m src.models.train_ssl_classifier --model wavlm
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.preprocessing import StandardScaler

from src.features.extract_ssl_embeddings import MODELS


FEATURE_DIR = Path("data/features/ssl")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results/week3_ssl")

GROUPS = {
    "angry": "upset", "disgust": "upset", "fear": "upset",
    "sad": "low",
    "happy": "okay", "neutral": "okay",
}

C_VALUES = [0.01, 0.1, 1.0]


def load(model, split):
    data = np.load(FEATURE_DIR / f"{model}_{split}.npz")
    return data["X"].astype(np.float32), data["y"]


def pick(X, layer):
    return X.mean(axis=1) if layer == "avg" else X[:, layer]


def fit(X, y, C):
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
    clf.fit(scaler.transform(X), y)
    return scaler, clf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, default="wavlm")
    args = parser.parse_args()

    X_train, y_train = load(args.model, "train")
    X_val, y_val = load(args.model, "validation")
    X_test, y_test = load(args.model, "test")

    layers = list(range(X_train.shape[1])) + ["avg"]
    rows = []
    best = None

    print(f"[TRAIN] {args.model}: searching {len(layers)} layers x {len(C_VALUES)} C values")

    for layer in layers:
        for C in C_VALUES:
            scaler, clf = fit(pick(X_train, layer), y_train, C)
            pred = clf.predict(scaler.transform(pick(X_val, layer)))
            acc = accuracy_score(y_val, pred)
            rows.append({"layer": layer, "C": C, "val_accuracy": round(acc, 4)})
            if best is None or acc > best[0]:
                best = (acc, layer, C)
        print(f"  layer {layer}: best val so far {best[0]:.3f} (layer {best[1]}, C={best[2]})")

    best_val, best_layer, best_C = best

    X_final = np.concatenate([pick(X_train, best_layer), pick(X_val, best_layer)])
    y_final = np.concatenate([y_train, y_val])
    scaler, clf = fit(X_final, y_final, best_C)

    pred = clf.predict(scaler.transform(pick(X_test, best_layer)))
    test_acc = accuracy_score(y_test, pred)
    test_f1 = f1_score(y_test, pred, average="macro")

    group_true = [GROUPS[y] for y in y_test]
    group_pred = [GROUPS[p] for p in pred]
    group_acc = accuracy_score(group_true, group_pred)

    print("\n" + "=" * 50)
    print(f"Model           : {MODELS[args.model]}")
    print(f"Best layer / C  : {best_layer} / {best_C}")
    print(f"Validation acc  : {best_val:.3f}")
    print(f"TEST accuracy   : {test_acc:.3f}   (Day 7 SVM was 0.490)")
    print(f"TEST macro F1   : {test_f1:.3f}")
    print(f"3-group accuracy: {group_acc:.3f}   (upset / low / okay)")
    print("=" * 50)
    print(classification_report(y_test, pred, digits=3))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / f"{args.model}_layer_search.csv", index=False)
    pd.DataFrame(
        classification_report(y_test, pred, output_dict=True)
    ).T.to_csv(RESULTS_DIR / f"{args.model}_classification_report.csv")
    labels = list(clf.classes_)
    pd.DataFrame(
        confusion_matrix(y_test, pred, labels=labels), index=labels, columns=labels
    ).to_csv(RESULTS_DIR / f"{args.model}_confusion_matrix.csv")
    pd.DataFrame([{
        "model": MODELS[args.model], "layer": best_layer, "C": best_C,
        "val_accuracy": round(best_val, 4), "test_accuracy": round(test_acc, 4),
        "test_macro_f1": round(test_f1, 4), "test_3group_accuracy": round(group_acc, 4),
    }]).to_csv(RESULTS_DIR / f"{args.model}_summary.csv", index=False)

    MODEL_DIR.mkdir(exist_ok=True)
    out = MODEL_DIR / f"ssl_{args.model}_emotion.joblib"
    joblib.dump({
        "ssl_model": MODELS[args.model],
        "layer": best_layer,
        "scaler": scaler,
        "classifier": clf,
        "labels": labels,
    }, out)
    print(f"\nSaved model -> {out}")
    print(f"Saved results -> {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
