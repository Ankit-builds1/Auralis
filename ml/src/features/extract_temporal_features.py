from pathlib import Path

import librosa
import numpy as np
import pandas as pd


SAMPLE_RATE = 16000
N_MFCC = 40

FEATURE_DIR = Path("data/features/temporal")


def extract_temporal_features(audio_path):
    audio, sample_rate = librosa.load(
        audio_path,
        sr=SAMPLE_RATE
    )

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=N_MFCC
    )

    # Delta
    delta = librosa.feature.delta(mfcc)

    # Delta-delta
    delta_delta = librosa.feature.delta(
        mfcc,
        order=2
    )

    # Combine feature maps
    combined = np.vstack([
        mfcc,
        delta,
        delta_delta
    ])

    # Fixed-size temporal representation
    # Divide the recording into 10 time regions.
    segments = np.array_split(
        combined,
        10,
        axis=1
    )

    features = []

    for segment in segments:

        # Mean for each MFCC coefficient
        features.extend(
            np.mean(segment, axis=1)
        )

        # Standard deviation for each coefficient
        features.extend(
            np.std(segment, axis=1)
        )

    return np.asarray(features, dtype=np.float32)


def process_split(split_name):

    metadata_path = (
        Path("data/metadata/splits")
        / f"{split_name}.csv"
    )

    df = pd.read_csv(metadata_path)

    X = []
    y = []

    total = len(df)

    print("\nProcessing:", split_name)
    print("Total files:", total)

    for index, row in df.iterrows():

        audio_path = Path(row["audio_path"])

        features = extract_temporal_features(
            audio_path
        )

        X.append(features)
        y.append(row["emotion"])

        if (index + 1) % 500 == 0:
            print(
                f"Processed {index + 1}/{total}"
            )

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)

    FEATURE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        FEATURE_DIR
        / f"{split_name}_temporal.npz"
    )

    np.savez_compressed(
        output_path,
        X=X,
        y=y
    )

    print(
        "Saved:",
        output_path
    )

    print(
        "Feature shape:",
        X.shape
    )

    print(
        "Label shape:",
        y.shape
    )


def main():

    print("=" * 70)
    print("AURALIS TEMPORAL FEATURE EXTRACTION")
    print("=" * 70)

    for split in [
        "train",
        "validation",
        "test"
    ]:

        process_split(split)

    print("\n" + "=" * 70)
    print("TEMPORAL FEATURE EXTRACTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()