from pathlib import Path

from src.features.extract_temporal_features import extract_temporal_features
from src.models.inference_day7 import predict_emotion


def predict_emotion_from_audio(audio_path):
    """
    Predict emotion directly from an audio file.

    Pipeline:
        audio file
        -> temporal feature extraction
        -> 2400-dimensional feature vector
        -> Day 7 emotion model
        -> emotion label
    """

    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    features = extract_temporal_features(audio_path)

    if features.shape[0] != 2400:
        raise ValueError(
            f"Expected 2400 features, got {features.shape[0]}"
        )

    emotion = predict_emotion(features)

    return emotion


if __name__ == "__main__":
    print("Auralis - Audio Emotion Inference")
    print("=" * 40)

    audio_path = input("Enter audio file path: ").strip()

    emotion = predict_emotion_from_audio(audio_path)

    print(f"Predicted emotion: {emotion}")
