from pathlib import Path

import numpy as np
from joblib import dump
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


CLEAN_TRAIN = Path(
    "data/features/day2_combined/train_combined.npz"
)

NOISY_TRAIN = Path(
    "data/features/day4_augmented/train_noisy_combined.npz"
)

CLEAN_TEST = Path(
    "data/features/day2_combined/test_combined.npz"
)

SCALER_OUTPUT = Path(
    "models/day4_noise_augmented_scaler.joblib"
)

MODEL_OUTPUT = Path(
    "models/day4_noise_augmented_svm_rbf.joblib"
)

RESULTS_OUTPUT = Path(
    "results/day4/noise_augmented_results.txt"
)


def load_features(path):
    data = np.load(
        path,
        allow_pickle=True
    )

    return data["X"], data["y"]


def main():

    print("=" * 70)
    print("AURALIS DAY 4 NOISE-AUGMENTED SVM")
    print("=" * 70)

    print("\nLoading clean training features...")
    X_clean, y_clean = load_features(CLEAN_TRAIN)

    print(f"Clean training shape: {X_clean.shape}")
    print(f"Clean labels shape:   {y_clean.shape}")

    print("\nLoading noisy training features...")
    X_noisy, y_noisy = load_features(NOISY_TRAIN)

    print(f"Noisy training shape: {X_noisy.shape}")
    print(f"Noisy labels shape:   {y_noisy.shape}")

    if X_clean.shape != X_noisy.shape:
        raise ValueError(
            "Clean and noisy feature shapes do not match."
        )

    if not np.array_equal(y_clean, y_noisy):
        raise ValueError(
            "Clean and noisy labels do not match."
        )

    print("\nCombining clean + noisy training data...")

    X_train = np.vstack([
        X_clean,
        X_noisy
    ])

    y_train = np.concatenate([
        y_clean,
        y_noisy
    ])

    print(f"Combined training shape: {X_train.shape}")
    print(f"Combined labels shape:   {y_train.shape}")

    print("\nScaling features...")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    print("Scaling complete.")

    print("\nTraining RBF SVM...")

    model = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        class_weight="balanced",
        random_state=42
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    print("Training complete.")

    print("\nSaving scaler and model...")

    SCALER_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    dump(
        scaler,
        SCALER_OUTPUT
    )

    dump(
        model,
        MODEL_OUTPUT
    )

    print(f"Scaler saved: {SCALER_OUTPUT}")
    print(f"Model saved:  {MODEL_OUTPUT}")

    print("\nEvaluating on clean test set...")

    X_test, y_test = load_features(
        CLEAN_TEST
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    predictions = model.predict(
        X_test_scaled
    )

    clean_accuracy = accuracy_score(
        y_test,
        predictions
    )

    clean_macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    clean_weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted"
    )

    print(
        f"Clean test accuracy:    {clean_accuracy:.4f}"
    )

    print(
        f"Clean test macro F1:    {clean_macro_f1:.4f}"
    )

    print(
        f"Clean test weighted F1: {clean_weighted_f1:.4f}"
    )

    print("\nClean test classification report:")
    print(
        classification_report(
            y_test,
            predictions
        )
    )

    results = f"""
AURALIS DAY 4 NOISE-AUGMENTED SVM

Training:
Clean samples: {len(X_clean)}
Noisy samples: {len(X_noisy)}
Total samples: {len(X_train)}
Features: {X_train.shape[1]}

Model:
SVC(kernel="rbf", C=10, gamma="scale",
    class_weight="balanced", random_state=42)

Clean Test:
Accuracy: {clean_accuracy:.6f}
Macro F1: {clean_macro_f1:.6f}
Weighted F1: {clean_weighted_f1:.6f}

Classification Report:
{classification_report(y_test, predictions)}
"""

    RESULTS_OUTPUT.write_text(
        results,
        encoding="utf-8"
    )

    print(
        f"\nResults saved: {RESULTS_OUTPUT}"
    )

    print("\nDay 4 augmented model training complete.")


if __name__ == "__main__":
    main()
