from pathlib import Path

import joblib
import numpy as np

from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


FEATURE_DIR = Path("data/features/day2_combined")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def load_features(split_name):
    path = FEATURE_DIR / f"{split_name}_combined.npz"
    data = np.load(path, allow_pickle=True)

    return data["X"], data["y"]


print("=" * 60)
print("Week 2 Day 2 - Combined Temporal + Prosodic SVM")
print("=" * 60)

print("\nLoading features...")

X_train, y_train = load_features("train")
X_val, y_val = load_features("validation")
X_test, y_test = load_features("test")

print(f"Train:      {X_train.shape}")
print(f"Validation: {X_val.shape}")
print(f"Test:       {X_test.shape}")


print("\nScaling features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


print("\nTraining RBF SVM...")

model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_scaled, y_train)


print("\nGenerating predictions...")

val_predictions = model.predict(X_val_scaled)
test_predictions = model.predict(X_test_scaled)


val_accuracy = accuracy_score(y_val, val_predictions)
val_macro_f1 = f1_score(
    y_val,
    val_predictions,
    average="macro"
)

test_accuracy = accuracy_score(y_test, test_predictions)
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


print("\nResults")
print("-" * 40)

print(f"Validation accuracy: {val_accuracy:.4f}")
print(f"Validation macro F1: {val_macro_f1:.4f}")
print(f"Test accuracy:       {test_accuracy:.4f}")
print(f"Test macro F1:       {test_macro_f1:.4f}")
print(f"Test weighted F1:    {test_weighted_f1:.4f}")


print("\nTest classification report:")
print(
    classification_report(
        y_test,
        test_predictions
    )
)


scaler_path = MODEL_DIR / "day2_combined_scaler.joblib"
model_path = MODEL_DIR / "day2_combined_svm_rbf.joblib"

joblib.dump(scaler, scaler_path)
joblib.dump(model, model_path)

print(f"\nScaler saved: {scaler_path}")
print(f"Model saved:  {model_path}")


results_path = RESULTS_DIR / "day2_combined_results.txt"

with open(results_path, "w") as f:
    f.write(
        "Week 2 Day 2 - Combined Temporal + Prosodic SVM\n"
    )
    f.write(f"Validation accuracy: {val_accuracy:.6f}\n")
    f.write(f"Validation macro F1: {val_macro_f1:.6f}\n")
    f.write(f"Test accuracy: {test_accuracy:.6f}\n")
    f.write(f"Test macro F1: {test_macro_f1:.6f}\n")
    f.write(f"Test weighted F1: {test_weighted_f1:.6f}\n")

    f.write("\nClassification report:\n")
    f.write(
        classification_report(
            y_test,
            test_predictions
        )
    )

print(f"Results saved: {results_path}")
