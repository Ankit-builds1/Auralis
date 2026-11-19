from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# Paths
# --------------------------------------------------

FEATURE_DIR = Path("data/features")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")


# --------------------------------------------------
# Load features
# --------------------------------------------------

def load_features(split_name):

    file_path = FEATURE_DIR / f"{split_name}_mfcc.npz"

    data = np.load(file_path)

    X = data["X"]
    y = data["y"]

    return X, y


# --------------------------------------------------
# Main training pipeline
# --------------------------------------------------

def main():

    print("=" * 60)
    print("AURALIS BASELINE EMOTION CLASSIFIER")
    print("=" * 60)

    # Load datasets
    X_train, y_train = load_features("train")
    X_validation, y_validation = load_features("validation")
    X_test, y_test = load_features("test")

    print("\nDataset shapes:")
    print("Training:", X_train.shape)
    print("Validation:", X_validation.shape)
    print("Test:", X_test.shape)

    # --------------------------------------------------
    # Feature scaling
    # --------------------------------------------------

    print("\nScaling features...")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)
    X_test_scaled = scaler.transform(X_test)

    # --------------------------------------------------
    # Random Forest
    # --------------------------------------------------

    print("\nTraining Random Forest...")

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    model.fit(X_train_scaled, y_train)

    print("Training complete.")

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    print("\nEvaluating validation set...")

    validation_predictions = model.predict(
        X_validation_scaled
    )

    validation_accuracy = accuracy_score(
        y_validation,
        validation_predictions,
    )

    print(
        f"Validation accuracy: "
        f"{validation_accuracy:.4f}"
    )

    print("\nValidation classification report:")
    print(
        classification_report(
            y_validation,
            validation_predictions,
        )
    )

    # --------------------------------------------------
    # Test
    # --------------------------------------------------

    print("\nEvaluating test set...")

    test_predictions = model.predict(
        X_test_scaled
    )

    test_accuracy = accuracy_score(
        y_test,
        test_predictions,
    )

    print(
        f"Test accuracy: "
        f"{test_accuracy:.4f}"
    )

    print("\nTest classification report:")
    print(
        classification_report(
            y_test,
            test_predictions,
        )
    )

    # --------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------

    labels = sorted(np.unique(y_test))

    cm = confusion_matrix(
        y_test,
        test_predictions,
        labels=labels,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=labels,
        yticklabels=labels,
    )

    plt.xlabel("Predicted Emotion")
    plt.ylabel("Actual Emotion")
    plt.title("Auralis Baseline Emotion Classifier")

    plt.tight_layout()

    confusion_matrix_path = (
        RESULTS_DIR / "baseline_confusion_matrix.png"
    )

    plt.savefig(confusion_matrix_path)

    plt.close()

    print(
        "\nConfusion matrix saved to:",
        confusion_matrix_path,
    )

    # --------------------------------------------------
    # Save model + scaler
    # --------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = MODEL_DIR / "baseline_random_forest.joblib"
    scaler_path = MODEL_DIR / "mfcc_scaler.joblib"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    print("\nModel saved to:", model_path)
    print("Scaler saved to:", scaler_path)

    print("\n" + "=" * 60)
    print("BASELINE TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()