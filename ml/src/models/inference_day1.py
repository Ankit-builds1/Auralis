from pathlib import Path

import joblib
import numpy as np


MODEL_DIR = Path("models")

SCALER_PATH = MODEL_DIR / "day2_combined_scaler.joblib"
MODEL_PATH = MODEL_DIR / "day2_combined_svm_rbf.joblib"


EMOTION_LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
]


def load_model():
    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_PATH)

    return scaler, model


def predict_emotion(features):
    scaler, model = load_model()

    features = np.asarray(features)

    if features.ndim == 1:
        features = features.reshape(1, -1)

    if features.shape[1] != 2640:
        raise ValueError(
            f"Expected 2640 features, got {features.shape[1]}"
        )

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)

    return prediction[0]


if __name__ == "__main__":
    print("=" * 60)
    print("Auralis - Week 3 Day 1 ML Inference")
    print("=" * 60)

    scaler, model = load_model()

    print("\nModel loaded successfully.")
    print(f"Expected features: {model.n_features_in_}")
    print(f"Classes: {list(model.classes_)}")

    # Test inference using one unseen test sample
    test_data = np.load(
        "data/features/day2_combined/test_combined.npz",
        allow_pickle=True,
    )

    X_test = test_data["X"]
    y_test = test_data["y"]

    sample = X_test[0]
    actual_label = y_test[0]

    predicted_label = predict_emotion(sample)

    print("\nTest sample inference")
    print("-" * 40)
    print(f"Actual emotion:    {actual_label}")
    print(f"Predicted emotion: {predicted_label}")
    print("\nInference successful.")
