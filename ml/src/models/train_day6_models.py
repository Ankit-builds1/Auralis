from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


FEATURE_DIR = Path("data/features/advanced")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")


def load_features(split_name):
    path = FEATURE_DIR / f"{split_name}_advanced.npz"
    data = np.load(path)
    return data["X"], data["y"]


def evaluate_model(model_name, model, X_train, y_train,
                   X_validation, y_validation, X_test, y_test):

    print("\n" + "=" * 70)
    print(f"TRAINING: {model_name}")
    print("=" * 70)

    model.fit(X_train, y_train)

    # Validation
    validation_predictions = model.predict(X_validation)

    validation_accuracy = accuracy_score(
        y_validation,
        validation_predictions
    )

    validation_macro_f1 = f1_score(
        y_validation,
        validation_predictions,
        average="macro"
    )

    print(f"\nValidation accuracy: {validation_accuracy:.4f}")
    print(f"Validation macro F1: {validation_macro_f1:.4f}")

    # Test
    test_predictions = model.predict(X_test)

    test_accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    test_macro_f1 = f1_score(
        y_test,
        test_predictions,
        average="macro"
    )

    test_weighted_f1 = f1_score(
        y_test,
        test_predictions,
        average="weighted"
    )

    print(f"\nTest accuracy: {test_accuracy:.4f}")
    print(f"Test macro F1: {test_macro_f1:.4f}")
    print(f"Test weighted F1: {test_weighted_f1:.4f}")

    print("\nTest classification report:")
    print(
        classification_report(
            y_test,
            test_predictions
        )
    )

    return {
        "model": model_name,
        "validation_accuracy": validation_accuracy,
        "validation_macro_f1": validation_macro_f1,
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_macro_f1,
        "test_weighted_f1": test_weighted_f1,
        "model_object": model,
    }


def main():

    print("=" * 70)
    print("AURALIS ML — DAY 6 MODEL COMPARISON")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load advanced features
    # ---------------------------------------------------------

    print("\nLoading advanced features...")

    X_train, y_train = load_features("train")
    X_validation, y_validation = load_features("validation")
    X_test, y_test = load_features("test")

    print("Training:", X_train.shape)
    print("Validation:", X_validation.shape)
    print("Test:", X_test.shape)

    # ---------------------------------------------------------
    # Scale
    # ---------------------------------------------------------

    print("\nScaling features...")

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_validation = scaler.transform(X_validation)
    X_test = scaler.transform(X_test)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    scaler_path = MODEL_DIR / "day6_feature_scaler.joblib"
    joblib.dump(scaler, scaler_path)

    print("Scaler saved to:", scaler_path)

    # ---------------------------------------------------------
    # Models
    # ---------------------------------------------------------

    models = {

        "SVM_RBF": SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            class_weight="balanced",
            random_state=42,
        ),

        "SVM_LINEAR": SVC(
            kernel="linear",
            C=1,
            class_weight="balanced",
            random_state=42,
        ),
    }

    results = []

    # ---------------------------------------------------------
    # Train models
    # ---------------------------------------------------------

    for model_name, model in models.items():

        result = evaluate_model(
            model_name,
            model,
            X_train,
            y_train,
            X_validation,
            y_validation,
            X_test,
            y_test,
        )

        results.append(result)

        model_path = MODEL_DIR / f"{model_name.lower()}.joblib"

        joblib.dump(
            model,
            model_path
        )

        print("\nModel saved to:", model_path)

    # ---------------------------------------------------------
    # Save comparison
    # ---------------------------------------------------------

    comparison_rows = []

    for result in results:

        comparison_rows.append({
            "model": result["model"],
            "validation_accuracy": result["validation_accuracy"],
            "validation_macro_f1": result["validation_macro_f1"],
            "test_accuracy": result["test_accuracy"],
            "test_macro_f1": result["test_macro_f1"],
            "test_weighted_f1": result["test_weighted_f1"],
        })

    comparison_df = pd.DataFrame(comparison_rows)

    output_path = RESULTS_DIR / "day6_model_comparison.csv"

    comparison_df.to_csv(
        output_path,
        index=False
    )

    print("\n" + "=" * 70)
    print("DAY 6 MODEL COMPARISON")
    print("=" * 70)

    print(
        comparison_df.to_string(index=False)
    )

    print("\nResults saved to:", output_path)

    print("\n" + "=" * 70)
    print("DAY 6 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()