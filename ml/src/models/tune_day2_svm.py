from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


FEATURE_DIR = Path("data/features/day2_combined")
MODEL_DIR = Path("models")
RESULT_DIR = Path("results")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


C_VALUES = [1, 5, 10, 20, 50]


def load_split(name):
    data = np.load(
        FEATURE_DIR / f"{name}_combined.npz",
        allow_pickle=True
    )

    X = data["X"]
    y = data["y"]

    return X, y


print("Loading combined features...")

X_train, y_train = load_split("train")
X_val, y_val = load_split("validation")
X_test, y_test = load_split("test")

print(f"Train: {X_train.shape}")
print(f"Validation: {X_val.shape}")
print(f"Test: {X_test.shape}")


print("\nScaling features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


results = []

best_model = None
best_c = None
best_val_f1 = -1


for C in C_VALUES:

    print(f"\nTraining RBF SVM with C={C}...")

    model = SVC(
        kernel="rbf",
        C=C,
        gamma="scale",
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    val_pred = model.predict(X_val_scaled)
    test_pred = model.predict(X_test_scaled)

    val_accuracy = accuracy_score(y_val, val_pred)
    val_macro_f1 = f1_score(
        y_val,
        val_pred,
        average="macro"
    )

    test_accuracy = accuracy_score(y_test, test_pred)
    test_macro_f1 = f1_score(
        y_test,
        test_pred,
        average="macro"
    )

    print(f"Validation accuracy: {val_accuracy:.4f}")
    print(f"Validation macro F1: {val_macro_f1:.4f}")
    print(f"Test accuracy:       {test_accuracy:.4f}")
    print(f"Test macro F1:       {test_macro_f1:.4f}")

    results.append({
        "C": C,
        "validation_accuracy": val_accuracy,
        "validation_macro_f1": val_macro_f1,
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_macro_f1,
    })

    if val_macro_f1 > best_val_f1:
        best_val_f1 = val_macro_f1
        best_c = C
        best_model = model


results_df = pd.DataFrame(results)

results_path = RESULT_DIR / "day2_svm_tuning.csv"
results_df.to_csv(results_path, index=False)

dump(
    scaler,
    MODEL_DIR / "day2_tuned_scaler.joblib"
)

dump(
    best_model,
    MODEL_DIR / "day2_tuned_svm_rbf.joblib"
)

print("\n" + "=" * 60)
print("SVM TUNING COMPLETE")
print("=" * 60)

print("\nResults:")
print(results_df.to_string(index=False))

print(f"\nBest C based on validation macro F1: {best_c}")
print(f"Best validation macro F1: {best_val_f1:.4f}")

print(f"\nSaved results to: {results_path}")
print("Saved best scaler/model.")
