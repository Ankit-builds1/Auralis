"""
ML Week 3 Day 3 - Step 3: speaker-adapted emotion model.

Idea: keep the frozen WavLM features and the best layer / C already found
on CREMA-D, and retrain the classifier on
    CREMA-D (train + validation)  +  our own clips (weighted higher)
so it learns how OUR voices sound in each emotion without forgetting
CREMA-D.

Fair evaluation (nothing is graded on clips it trained on):
  - our clips are split per speaker x emotion: TEST_FRAC held out
  - BEFORE = the current CREMA-D-only model
  - AFTER  = the adapted model trained without the held-out clips
  - both are also checked on the CREMA-D test set (it should not drop much)

Then a DEPLOY model is trained on ALL our clips and saved as
    models/ssl_<model>-adapted_emotion.joblib
(same format as before, so the backend can load it with
 EMOTION_MODEL=<model>-adapted).

Run from the ml/ folder:
    python -m src.models.train_speaker_adapted --model wavlm-large
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.features.extract_ssl_embeddings import MODELS
from src.models.train_ssl_classifier import GROUPS, load, pick


FEATURE_DIR = Path("data/features/ssl")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results/week3_adapt")


def fit_weighted(X, y, weights, C):
    scaler = StandardScaler().fit(X, sample_weight=weights)
    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
    clf.fit(scaler.transform(X), y, sample_weight=weights)
    return scaler, clf


def predict(scaler, clf, X):
    return clf.predict(scaler.transform(X))


def scores(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    group = accuracy_score([GROUPS[y] for y in y_true], [GROUPS[p] for p in y_pred])
    return acc, group


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, default="wavlm-large")
    parser.add_argument("--weight", type=float, default=10.0,
                        help="how much more one of our clips counts than one CREMA-D clip")
    parser.add_argument("--test-frac", type=float, default=0.3,
                        help="share of our clips held out for the fair test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ---------- current CREMA-D model: its layer and C ----------
    base_path = MODEL_DIR / f"ssl_{args.model}_emotion.joblib"
    base = joblib.load(base_path)
    layer = base["layer"]
    C = float(base["classifier"].C)
    print(f"[ADAPT] base model {base_path.name}: layer {layer}, C={C}")

    # ---------- CREMA-D features ----------
    X_tr, y_tr = load(args.model, "train")
    X_va, y_va = load(args.model, "validation")
    X_te, y_te = load(args.model, "test")
    X_crema = np.concatenate([pick(X_tr, layer), pick(X_va, layer)])
    y_crema = np.concatenate([y_tr, y_va])
    X_crema_test = pick(X_te, layer)

    # ---------- our clips ----------
    mine = np.load(FEATURE_DIR / f"{args.model}_myvoice.npz")
    X_mine = pick(mine["X"].astype(np.float32), layer)
    y_mine = mine["y"].astype(str)
    spk_mine = mine["speaker"].astype(str)
    print(f"[ADAPT] our clips: {len(y_mine)} from speakers {sorted(set(spk_mine.tolist()))}")

    strata = np.char.add(np.char.add(spk_mine, "_"), y_mine)
    idx_train, idx_test = train_test_split(
        np.arange(len(y_mine)), test_size=args.test_frac,
        random_state=args.seed, stratify=strata,
    )

    # ---------- BEFORE: CREMA-D-only model ----------
    before_mine = predict(base["scaler"], base["classifier"], X_mine[idx_test])
    before_crema = predict(base["scaler"], base["classifier"], X_crema_test)

    # ---------- AFTER: adapted model (held-out clips not used) ----------
    X_fit = np.concatenate([X_crema, X_mine[idx_train]])
    y_fit = np.concatenate([y_crema, y_mine[idx_train]])
    w_fit = np.concatenate([np.ones(len(y_crema)), np.full(len(idx_train), args.weight)])
    scaler, clf = fit_weighted(X_fit, y_fit, w_fit, C)

    after_mine = predict(scaler, clf, X_mine[idx_test])
    after_crema = predict(scaler, clf, X_crema_test)

    # ---------- report ----------
    rows = []
    for name, y_true, b, a in [
        ("our voices (held out)", y_mine[idx_test], before_mine, after_mine),
        ("CREMA-D test", y_te, before_crema, after_crema),
    ]:
        b_acc, b_grp = scores(y_true, b)
        a_acc, a_grp = scores(y_true, a)
        rows.append({"test_set": name, "clips": len(y_true),
                     "before_acc": round(b_acc, 3), "after_acc": round(a_acc, 3),
                     "before_3group": round(b_grp, 3), "after_3group": round(a_grp, 3)})

    for spk in sorted(set(spk_mine.tolist())):
        sel = spk_mine[idx_test] == spk
        b_acc, b_grp = scores(y_mine[idx_test][sel], before_mine[sel])
        a_acc, a_grp = scores(y_mine[idx_test][sel], after_mine[sel])
        rows.append({"test_set": f"  speaker {spk}", "clips": int(sel.sum()),
                     "before_acc": round(b_acc, 3), "after_acc": round(a_acc, 3),
                     "before_3group": round(b_grp, 3), "after_3group": round(a_grp, 3)})

    table = pd.DataFrame(rows)
    print("\n" + "=" * 78)
    print(f"SPEAKER ADAPTATION  ({MODELS[args.model]}, layer {layer}, "
          f"our-clip weight {args.weight})")
    print("=" * 78)
    print(table.to_string(index=False))
    print("=" * 78)

    labels = list(clf.classes_)
    cm = pd.DataFrame(confusion_matrix(y_mine[idx_test], after_mine, labels=labels),
                      index=labels, columns=labels)
    print("\nAFTER - confusion matrix on our held-out clips (rows = true):")
    print(cm.to_string())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_DIR / f"{args.model}_adapt_summary.csv", index=False)
    cm.to_csv(RESULTS_DIR / f"{args.model}_adapt_confusion_ourvoices.csv")

    # ---------- DEPLOY model: all our clips ----------
    X_all = np.concatenate([X_crema, X_mine])
    y_all = np.concatenate([y_crema, y_mine])
    w_all = np.concatenate([np.ones(len(y_crema)), np.full(len(y_mine), args.weight)])
    scaler_all, clf_all = fit_weighted(X_all, y_all, w_all, C)

    MODEL_DIR.mkdir(exist_ok=True)
    out = MODEL_DIR / f"ssl_{args.model}-adapted_emotion.joblib"
    joblib.dump({
        "ssl_model": base["ssl_model"],
        "layer": layer,
        "scaler": scaler_all,
        "classifier": clf_all,
        "labels": list(clf_all.classes_),
        "adapted_speakers": sorted(set(spk_mine.tolist())),
        "our_clip_weight": args.weight,
    }, out)

    print(f"\nSaved deploy model -> {out}")
    print(f"Saved results      -> {RESULTS_DIR}/")
    print(f"Backend: copy it to Backend/models/ and start with "
          f"$env:EMOTION_MODEL = \"{args.model}-adapted\"")


if __name__ == "__main__":
    main()
