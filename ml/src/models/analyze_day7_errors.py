from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


FEATURE_DIR = Path("data/features/temporal")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")
METADATA_PATH = Path("data/metadata/splits/test.csv")


def main():

    print("=" * 70)
    print("AURALIS ML — WEEK 2 DAY 1 ERROR ANALYSIS")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load test features
    # ---------------------------------------------------------

    print("\nLoading test features...")

    data = np.load(
        FEATURE_DIR / "test_temporal.npz"
    )

    X_test = data["X"]
    y_test = data["y"]

    print("Test features:", X_test.shape)

    # ---------------------------------------------------------
    # Load scaler
    # ---------------------------------------------------------

    scaler = joblib.load(
        MODEL_DIR / "day7_temporal_scaler.joblib"
    )

    X_test_scaled = scaler.transform(X_test)

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------

    model = joblib.load(
        MODEL_DIR / "day7_temporal_svm_rbf.joblib"
    )

    # ---------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------

    print("\nGenerating predictions...")

    predictions = model.predict(
        X_test_scaled
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(
        f"\nTest accuracy: {accuracy:.4f}"
    )

    # ---------------------------------------------------------
    # Classification report
    # ---------------------------------------------------------

    report = classification_report(
        y_test,
        predictions,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    print("\nClassification report:")
    print(
        report_df.to_string()
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    report_path = (
        RESULTS_DIR /
        "day7_classification_report.csv"
    )

    report_df.to_csv(
        report_path
    )

    print(
        "\nClassification report saved to:",
        report_path
    )

    # ---------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------

    labels = sorted(
        np.unique(y_test)
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )

    print("\nConfusion matrix:")
    print(cm)

    plt.figure(
        figsize=(8, 6)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=labels,
        yticklabels=labels
    )

    plt.xlabel(
        "Predicted Emotion"
    )

    plt.ylabel(
        "Actual Emotion"
    )

    plt.title(
        "Auralis Day 7 Temporal SVM"
    )

    plt.tight_layout()

    confusion_path = (
        RESULTS_DIR /
        "day7_confusion_matrix.png"
    )

    plt.savefig(
        confusion_path,
        dpi=150
    )

    plt.close()

    print(
        "\nConfusion matrix saved to:",
        confusion_path
    )

    # ---------------------------------------------------------
    # Error analysis
    # ---------------------------------------------------------

    metadata = pd.read_csv(
        METADATA_PATH
    )

    if len(metadata) != len(y_test):

        raise ValueError(
            "Test metadata length does not "
            "match test feature length."
        )

    metadata["actual"] = y_test
    metadata["predicted"] = predictions

    metadata["correct"] = (
        metadata["actual"]
        == metadata["predicted"]
    )

    errors = metadata[
        ~metadata["correct"]
    ].copy()

    errors_path = (
        RESULTS_DIR /
        "day7_errors.csv"
    )

    errors.to_csv(
        errors_path,
        index=False
    )

    print(
        "\nTotal test samples:",
        len(metadata)
    )

    print(
        "Correct predictions:",
        metadata["correct"].sum()
    )

    print(
        "Incorrect predictions:",
        len(errors)
    )

    print(
        "Error rate:",
        f"{len(errors) / len(metadata):.4f}"
    )

    # ---------------------------------------------------------
    # Most confused pairs
    # ---------------------------------------------------------

    confusion_pairs = (
        errors
        .groupby(
            ["actual", "predicted"]
        )
        .size()
        .reset_index(
            name="count"
        )
        .sort_values(
            "count",
            ascending=False
        )
    )

    print(
        "\nMost common confusion pairs:"
    )

    print(
        confusion_pairs
        .head(15)
        .to_string(index=False)
    )

    confusion_pairs_path = (
        RESULTS_DIR /
        "day7_confusion_pairs.csv"
    )

    confusion_pairs.to_csv(
        confusion_pairs_path,
        index=False
    )

    print(
        "\nConfusion pairs saved to:",
        confusion_pairs_path
    )

    # ---------------------------------------------------------
    # Per-emotion error rates
    # ---------------------------------------------------------

    emotion_stats = (
        metadata
        .groupby("actual")
        .agg(
            total=("correct", "size"),
            correct=("correct", "sum")
        )
    )

    emotion_stats["errors"] = (
        emotion_stats["total"]
        - emotion_stats["correct"]
    )

    emotion_stats["error_rate"] = (
        emotion_stats["errors"]
        / emotion_stats["total"]
    )

    emotion_stats = emotion_stats.sort_values(
        "error_rate",
        ascending=False
    )

    print(
        "\nPer-emotion error rates:"
    )

    print(
        emotion_stats.to_string()
    )

    emotion_stats_path = (
        RESULTS_DIR /
        "day7_emotion_error_rates.csv"
    )

    emotion_stats.to_csv(
        emotion_stats_path
    )

    print(
        "\nEmotion error rates saved to:",
        emotion_stats_path
    )

    print("\n" + "=" * 70)
    print("WEEK 2 DAY 1 ERROR ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()