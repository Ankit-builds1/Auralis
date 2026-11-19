import librosa
import numpy as np


def extract_features(audio_path):
    # Load audio
    audio, sample_rate = librosa.load(
        audio_path,
        sr=16000
    )

    # Extract MFCC features
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    # Average over time
    mfcc_mean = np.mean(mfcc, axis=1)

    return mfcc_mean


if __name__ == "__main__":
    print("Auralis ML audio feature extraction ready")