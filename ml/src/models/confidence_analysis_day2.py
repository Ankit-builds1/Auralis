from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score


MODEL_DIR = Path("models")
FEATURE_PATH = Path("data/features/day2_combined/test_combined.npz")
RESULTS_PATH = Path("results/week3_day2_confidence_analysis.csv")


def main():
    print("=" * 60)
    print("Auralis - Week 3 Day 2 Confidence Analysis")
    print("=" * 60)

    scaler = joblib.load(
        MODEL_DIR / "day2_combined_scaler.joblib"
    )

    model = joblib.load(
        MODEL_DIR / "day2_combined_svm_rbf.joblib"
    )

    data = np.load(
        FEATURE_PATH,
        allow_pickle=True,
    )

    X_test = data["X"]
    y_test = data["y"]

    X_scaled = scaler.transform(X_test)

    predictions = model.predict(X_scaled)
    decision_scores = model.decision_function(X_scaled)

    sorted_scores = np.sort(
        decision_scores,
        axis=1
    )

    max_scores = sorted_scores[:, -1]
    second_scores = sorted_scores[:, -2]

    margins = max_scores - second_scores

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    correct = predictions == y_test

    print("\nDataset")
    print("-" * 40)
    print(f"Test samples: {len(y_test)}")
    print(f"Features:      {X_test.shape[1]}")

    print("\nOverall performance")
    print("-" * 40)
    print(f"Accuracy: {accuracy:.4f}")

    print("\nDecision margin statistics")
    print("-" * 40)
    print(f"Mean margin:   {np.mean(margins):.4f}")
    print(f"Median margin: {np.median(margins):.4f}")
    print(f"Min margin:    {np.min(margins):.4f}")
    print(f"Max margin:    {np.max(margins):.4f}")

    print("\nCorrect vs incorrect predictions")
    print("-" * 40)
    print(f"Correct mean margin:   {np.mean(margins[correct]):.4f}")
    print(f"Incorrect mean margin: {np.mean(margins[~correct]):.4f}")

    print("\nConfidence buckets")
    print("-" * 40)

    thresholds = [
        ("Very uncertain", 0.0, 0.5),
        ("Uncertain", 0.5, 1.0),
        ("Moderate", 1.0, 2.0),
        ("Confident", 2.0, np.inf),
    ]

    for name, low, high in thresholds:
        mask = (margins >= low) & (margins < high)

        if not np.any(mask):
            continue

        bucket_accuracy = np.mean(correct[mask])

        print(
            f"{name:16s}: "
            f"{np.sum(mask):4d} samples | "
            f"accuracy={bucket_accuracy:.4f} | "
            f"mean_margin={np.mean(margins[mask]):.4f}"
        )

    RESULTS_PATH.parent.mkdir(exist_ok=True)

    results = np.column_stack(
        (
            margins,
            correct.astype(int),
        )
    )

    np.savetxt(
        RESULTS_PATH,
        results,
        delimiter=",",
        header="decision_margin,correct",
        comments="",
    )

    print(f"\nAnalysis saved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
