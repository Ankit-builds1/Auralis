from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


TRAIN_FILE = Path(
    "data/features/day2_combined/train_combined.npz"
)

VALIDATION_FILE = Path(
    "data/features/day2_combined/validation_combined.npz"
)

TEST_FILE = Path(
    "data/features/day2_combined/test_combined.npz"
)

RESULTS_FILE = Path(
    "results/day5/class_weight_tuning.csv"
)

BEST_SCALER = Path(
    "models/day5_class_weight_scaler.joblib"
)

BEST_MODEL = Path(
    "models/day5_class_weight_svm_rbf.joblib"
)


def load_features(path):
    data = np.load(
        path,
        allow_pickle=True
    )

    return data["X"], data["y"]


def evaluate_model(model, scaler, X, y):

    X_scaled = scaler.transform(X)

    predictions = model.predict(
        X_scaled
    )

    accuracy = accuracy_score(
        y,
        predictions
    )

    macro_f1 = f1_score(
        y,
        predictions,
        average="macro"
    )

    weighted_f1 = f1_score(
        y,
        predictions,
        average="weighted"
    )

    return accuracy, macro_f1, weighted_f1


def main():

    print("=" * 70)
    print("AURALIS DAY 5 CLASS-WEIGHT OPTIMIZATION")
    print("=" * 70)

    print("\nLoading datasets...")

    X_train, y_train = load_features(
        TRAIN_FILE
    )

    X_validation, y_validation = load_features(
        VALIDATION_FILE
    )

    X_test, y_test = load_features(
        TEST_FILE
    )

    print(
        f"Train:      {X_train.shape}"
    )

    print(
        f"Validation: {X_validation.shape}"
    )

    print(
        f"Test:       {X_test.shape}"
    )

    print("\nScaling features...")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_validation_scaled = scaler.transform(
        X_validation
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    class_weights = {

        "balanced": "balanced",

        "custom_1": {
            "angry": 1.0,
            "disgust": 1.3,
            "fear": 1.3,
            "happy": 1.1,
            "neutral": 1.0,
            "sad": 1.2
        },

        "custom_2": {
            "angry": 1.0,
            "disgust": 1.5,
            "fear": 1.5,
            "happy": 1.2,
            "neutral": 1.0,
            "sad": 1.3
        },

        "custom_3": {
            "angry": 0.9,
            "disgust": 1.5,
            "fear": 1.5,
            "happy": 1.2,
            "neutral": 0.9,
            "sad": 1.3
        }
    }

    results = []

    best_name = None
    best_model = None
    best_macro_f1 = -1.0

    for name, weights in class_weights.items():

        print("\n" + "-" * 70)
        print(
            f"Training configuration: {name}"
        )
        print("-" * 70)

        model = SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            class_weight=weights,
            random_state=42
        )

        model.fit(
            X_train_scaled,
            y_train
        )

        validation_predictions = model.predict(
            X_validation_scaled
        )

        validation_accuracy = accuracy_score(
            y_validation,
            validation_predictions
        )

        validation_macro_f1 = f1_score(
            y_validation,
            validation_predictions,
            average="macro"
        )

        validation_weighted_f1 = f1_score(
            y_validation,
            validation_predictions,
            average="weighted"
        )

        test_predictions = model.predict(
            X_test_scaled
        )

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

        print(
            f"Validation accuracy: {validation_accuracy:.4f}"
        )

        print(
            f"Validation macro F1: {validation_macro_f1:.4f}"
        )

        print(
            f"Test accuracy:       {test_accuracy:.4f}"
        )

        print(
            f"Test macro F1:       {test_macro_f1:.4f}"
        )

        results.append({
            "configuration": name,
            "validation_accuracy": validation_accuracy,
            "validation_macro_f1": validation_macro_f1,
            "validation_weighted_f1": validation_weighted_f1,
            "test_accuracy": test_accuracy,
            "test_macro_f1": test_macro_f1,
            "test_weighted_f1": test_weighted_f1
        })

        if validation_macro_f1 > best_macro_f1:

            best_macro_f1 = validation_macro_f1
            best_name = name
            best_model = model

    results_df = pd.DataFrame(
        results
    )

    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        RESULTS_FILE,
        index=False
    )

    BEST_SCALER.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    dump(
        scaler,
        BEST_SCALER
    )

    dump(
        best_model,
        BEST_MODEL
    )

    print("\n" + "=" * 70)
    print("CLASS-WEIGHT RESULTS")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    print("\nSelected configuration:")
    print(
        f"{best_name}"
    )

    print(
        f"Validation macro F1: "
        f"{best_macro_f1:.4f}"
    )

    print(
        f"\nResults saved: {RESULTS_FILE}"
    )

    print(
        f"Scaler saved:  {BEST_SCALER}"
    )

    print(
        f"Model saved:   {BEST_MODEL}"
    )

    print("\nDay 5 class-weight experiment complete.")


if __name__ == "__main__":
    main()
