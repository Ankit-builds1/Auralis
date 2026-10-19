"""
ML Week 3 Day 4 - Step 3: multi-corpus emotion model + "will it work on a
new person?" test.

Trains on CREMA-D + RAVDESS + TESS + our own clips (same frozen WavLM
features, same layer and C as the CREMA-D model) and compares it with the
current CREMA-D + our-clips model.

Three reports:
  1. SPEAKER-HELD-OUT TEST - every test set has speakers the model never heard
       CREMA-D : official test split (its own actors)
       RAVDESS : actors 21-24 held out
       TESS    : 20% of clips held out (only 2 speakers, so this one is
                 optimistic - same voices as training)
       ours    : same 30% held-out clips as train_speaker_adapted.py
  2. LEAVE-ONE-CORPUS-OUT - train without a whole dataset, test on all of it.
     Different studio, mic and actors: the closest thing to "a new user".
  3. DEPLOY model on everything -> models/ssl_<model>-multi_emotion.joblib
     (backend: EMOTION_MODEL=<model>-multi)

Run from the ml/ folder (after extract_corpus.py for ravdess and tess):
    python -m src.models.train_multicorpus --model wavlm-large
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
from src.models.train_ssl_classifier import GROUPS, pick


FEATURE_DIR = Path("data/features/ssl")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results/week3_multi")

RAVDESS_TEST_ACTORS = {f"ravdess_{a:02d}" for a in (21, 22, 23, 24)}
PUBLIC = ["crema", "ravdess", "tess"]


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def npz(name):
    path = FEATURE_DIR / f"{name}.npz"
    return np.load(path) if path.exists() else None


def load_all(model, layer, test_frac, seed):
    """Returns {corpus: {"train": (X, y), "test": (X, y), "all": (X, y)}}."""
    data = {}

    tr, va, te = npz(f"{model}_train"), npz(f"{model}_validation"), npz(f"{model}_test")
    if tr is None or va is None or te is None:
        raise SystemExit(f"CREMA-D features missing in {FEATURE_DIR} - run extract_ssl_embeddings first.")
    Xtr = np.concatenate([pick(tr["X"].astype(np.float32), layer),
                          pick(va["X"].astype(np.float32), layer)])
    ytr = np.concatenate([tr["y"], va["y"]]).astype(str)
    Xte, yte = pick(te["X"].astype(np.float32), layer), te["y"].astype(str)
    data["crema"] = {"train": (Xtr, ytr), "test": (Xte, yte), "all": (Xtr, ytr)}
    # CREMA "all" = train + validation (the test actors stay unseen, as before)

    rav = npz(f"{model}_ravdess")
    if rav is not None:
        X, y, spk = pick(rav["X"].astype(np.float32), layer), rav["y"].astype(str), rav["speaker"].astype(str)
        test = np.isin(spk, list(RAVDESS_TEST_ACTORS))
        data["ravdess"] = {"train": (X[~test], y[~test]), "test": (X[test], y[test]), "all": (X, y)}

    tess = npz(f"{model}_tess")
    if tess is not None:
        X, y, spk = pick(tess["X"].astype(np.float32), layer), tess["y"].astype(str), tess["speaker"].astype(str)
        strata = np.char.add(np.char.add(spk, "_"), y)
        i_tr, i_te = train_test_split(np.arange(len(y)), test_size=0.2,
                                      random_state=seed, stratify=strata)
        data["tess"] = {"train": (X[i_tr], y[i_tr]), "test": (X[i_te], y[i_te]), "all": (X, y)}

    mine = npz(f"{model}_myvoice")
    if mine is not None:
        X, y, spk = pick(mine["X"].astype(np.float32), layer), mine["y"].astype(str), mine["speaker"].astype(str)
        strata = np.char.add(np.char.add(spk, "_"), y)
        i_tr, i_te = train_test_split(np.arange(len(y)), test_size=test_frac,
                                      random_state=seed, stratify=strata)
        data["ours"] = {"train": (X[i_tr], y[i_tr]), "test": (X[i_te], y[i_te]), "all": (X, y)}

    return data


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

def stack(parts, ours_weight):
    """parts: list of (corpus_name, (X, y)). Our clips get a higher weight."""
    X = np.concatenate([p[1][0] for p in parts])
    y = np.concatenate([p[1][1] for p in parts])
    w = np.concatenate([np.full(len(p[1][1]), ours_weight if p[0] == "ours" else 1.0)
                        for p in parts])
    return X, y, w


def fit(X, y, w, C):
    scaler = StandardScaler().fit(X, sample_weight=w)
    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
    clf.fit(scaler.transform(X), y, sample_weight=w)
    return scaler, clf


def scores(scaler, clf, X, y):
    pred = clf.predict(scaler.transform(X))
    acc = accuracy_score(y, pred)
    grp = accuracy_score([GROUPS[t] for t in y], [GROUPS[p] for p in pred])
    return round(acc, 3), round(grp, 3), pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, default="wavlm-large")
    parser.add_argument("--weight", type=float, default=30.0,
                        help="how much one of our clips counts vs one public clip")
    parser.add_argument("--test-frac", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base = joblib.load(MODEL_DIR / f"ssl_{args.model}_emotion.joblib")
    layer, C = base["layer"], float(base["classifier"].C)
    print(f"[MULTI] layer {layer}, C={C}, our-clip weight {args.weight}")

    data = load_all(args.model, layer, args.test_frac, args.seed)
    for name, d in data.items():
        print(f"  {name:8s} train {len(d['train'][1]):5d}   test {len(d['test'][1]):5d}")
    public = [c for c in PUBLIC if c in data]
    has_ours = "ours" in data

    # ---------- 1. speaker-held-out test: old vs new model ----------
    old_parts = [("crema", data["crema"]["train"])]
    new_parts = [(c, data[c]["train"]) for c in public]
    if has_ours:
        old_parts.append(("ours", data["ours"]["train"]))
        new_parts.append(("ours", data["ours"]["train"]))

    old = fit(*stack(old_parts, args.weight), C)
    new = fit(*stack(new_parts, args.weight), C)

    rows = []
    for name in public + (["ours"] if has_ours else []):
        X, y = data[name]["test"]
        o_acc, o_grp, _ = scores(*old, X, y)
        n_acc, n_grp, pred = scores(*new, X, y)
        label = {"crema": "CREMA-D test actors", "ravdess": "RAVDESS actors 21-24",
                 "tess": "TESS 20% (same 2 voices)", "ours": "our voices (held out)"}[name]
        rows.append({"test_set": label, "clips": len(y),
                     "crema+ours_acc": o_acc, "multi_acc": n_acc,
                     "crema+ours_3group": o_grp, "multi_3group": n_grp})
        if name == "ours":
            ours_pred, ours_true = pred, y

    held = pd.DataFrame(rows)
    print("\n" + "=" * 86)
    print("1) SPEAKER-HELD-OUT TEST   (old = CREMA-D + ours,  new = all corpora + ours)")
    print("=" * 86)
    print(held.to_string(index=False))

    # ---------- 2. leave-one-corpus-out ----------
    loco = []
    for out_name in public:
        parts = [(c, data[c]["all"]) for c in public if c != out_name]
        if has_ours:
            parts.append(("ours", data["ours"]["train"]))
        if not parts or (len(parts) == 1 and parts[0][0] == "ours"):
            continue
        model = fit(*stack(parts, args.weight), C)
        X, y = data[out_name]["all"]
        acc, grp, _ = scores(*model, X, y)
        trained_on = " + ".join(p[0] for p in parts)
        loco.append({"unseen_corpus": out_name, "clips": len(y),
                     "trained_on": trained_on, "accuracy": acc, "3group": grp})
        print(f"  [LOCO] {out_name} done")

    loco = pd.DataFrame(loco)
    print("\n" + "=" * 86)
    print("2) LEAVE-ONE-CORPUS-OUT   (a whole dataset never seen = closest to a new user)")
    print("=" * 86)
    print(loco.to_string(index=False) if len(loco) else "  needs at least 2 public corpora")
    print("   chance level: 6 emotions = 0.167, 3 groups = 0.333")

    if has_ours:
        labels = sorted(set(ours_true) | set(ours_pred))
        cm = pd.DataFrame(confusion_matrix(ours_true, ours_pred, labels=labels),
                          index=labels, columns=labels)
        print("\nNEW model - confusion matrix on our held-out clips (rows = true):")
        print(cm.to_string())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    held.to_csv(RESULTS_DIR / f"{args.model}_heldout_speakers.csv", index=False)
    loco.to_csv(RESULTS_DIR / f"{args.model}_leave_one_corpus_out.csv", index=False)
    if has_ours:
        cm.to_csv(RESULTS_DIR / f"{args.model}_confusion_ourvoices.csv")

    # ---------- 3. deploy model on everything ----------
    parts = [(c, data[c]["all"]) for c in public]
    if has_ours:
        parts.append(("ours", data["ours"]["all"]))
    scaler, clf = fit(*stack(parts, args.weight), C)

    out = MODEL_DIR / f"ssl_{args.model}-multi_emotion.joblib"
    joblib.dump({
        "ssl_model": base["ssl_model"],
        "layer": layer,
        "scaler": scaler,
        "classifier": clf,
        "labels": list(clf.classes_),
        "corpora": public + (["ours"] if has_ours else []),
        "our_clip_weight": args.weight,
    }, out)
    print(f"\nSaved deploy model -> {out}")
    print(f"Saved results      -> {RESULTS_DIR}/")
    print(f"Backend: copy it to Backend/models/ and start with "
          f"$env:EMOTION_MODEL = \"{args.model}-multi\"")


if __name__ == "__main__":
    main()
