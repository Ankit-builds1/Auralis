from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


FEATURE_PATH = Path(
    "data/features/day2_combined/test_combined.npz"
)

SCALER_PATH = Path(
    "models/day2_combined_scaler.joblib"
)

MODEL_PATH = Path(
    "models/day2_combined_svm_rbf.joblib"
)

TEST_CSV = Path(
    "data/metadata/splits/test.csv"
)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


print("Loading test features...")

data = __import__("numpy").load(
    FEATURE_PATH,
    allow_pickle=True
)

X_test = data["X"]
y_test = data["y"]

print(f"Test features: {X_test.shape}")


print("\nLoading scaler and model...")

scaler = joblib.load(SCALER_PATH)
model = joblib.load(MODEL_PATH)

X_test_scaled = scaler.transform(X_test)


print("\nGenerating predictions...")

predictions = model.predict(X_test_scaled)

accuracy = accuracy_score(
    y_test,
    predictions
)

print(f"\nTest accuracy: {accuracy:.4f}")


print("\nClassification report:")

report = classification_report(
    y_test,
    predictions,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

print(report_df)


report_df.to_csv(
    RESULTS_DIR / "day2_classification_report.csv"
)


labels = sorted(set(y_test))

cm = confusion_matrix(
    y_test,
    predictions,
    labels=labels
)

print("\nConfusion matrix:")
print(cm)


plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=labels,
    yticklabels=labels
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Day 2 Combined SVM Confusion Matrix")

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "day2_confusion_matrix.png",
    dpi=200
)

plt.close()


test_df = pd.read_csv(TEST_CSV)

test_df["actual"] = y_test
test_df["predicted"] = predictions
test_df["correct"] = (
    test_df["actual"] ==
    test_df["predicted"]
)

errors_df = test_df[
    ~test_df["correct"]
]

errors_df.to_csv(
    RESULTS_DIR / "day2_errors.csv",
    index=False
)


confusion_pairs = (
    errors_df
    .groupby(["actual", "predicted"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

print("\nMost common confusion pairs:")
print(confusion_pairs.head(15).to_string(index=False))

confusion_pairs.to_csv(
    RESULTS_DIR / "day2_confusion_pairs.csv",
    index=False
)


emotion_stats = (
    test_df
    .groupby("actual")["correct"]
    .agg(
        total="count",
        correct="sum"
    )
)

emotion_stats["errors"] = (
    emotion_stats["total"] -
    emotion_stats["correct"]
)

emotion_stats["error_rate"] = (
    emotion_stats["errors"] /
    emotion_stats["total"]
)

print("\nPer-emotion error rates:")
print(emotion_stats)

emotion_stats.to_csv(
    RESULTS_DIR / "day2_emotion_error_rates.csv"
)


print("\nDay 2 error analysis complete.")
