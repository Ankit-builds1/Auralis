import librosa
import numpy as np
from pathlib import Path


DATASET_PATH = Path("data/raw/CREMA-D")


def extract_features(audio_path):
    audio, sample_rate = librosa.load(
        audio_path,
        sr=16000
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    mfcc_mean = np.mean(mfcc, axis=1)

    return mfcc_mean, sample_rate


if __name__ == "__main__":

    audio_files = list(DATASET_PATH.rglob("*.wav"))

    print("Auralis ML Audio Exploration")
    print("Total audio files:", len(audio_files))

    if not audio_files:
        print("No audio files found.")
    else:
        audio_path = audio_files[0]

        features, sample_rate = extract_features(audio_path)

        print("Audio file:", audio_path)
        print("Sample rate:", sample_rate)
        print("MFCC feature shape:", features.shape)
        print("Feature vector:", features)