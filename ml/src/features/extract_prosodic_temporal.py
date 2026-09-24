from pathlib import Path

import librosa
import numpy as np
import pandas as pd


SAMPLE_RATE = 16000
N_REGIONS = 10


def extract_temporal_prosodic_features(audio_path):
    """
    Extract temporal acoustic/prosodic features.

    Features:
    - RMS energy
    - Zero-crossing rate
    - Spectral centroid
    - Spectral bandwidth
    - Spectral rolloff
    - Spectral contrast

    Each feature is summarized by mean and standard deviation
    across 10 temporal regions.
    """

    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

    if len(y) == 0:
        return np.zeros(240, dtype=np.float32)

    rms = librosa.feature.rms(y=y)[0]

    zcr = librosa.feature.zero_crossing_rate(y)[0]

    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )[0]

    bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr
    )[0]

    rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr
    )[0]

    contrast = librosa.feature.spectral_contrast(
        y=y,
        sr=sr
    )

    feature_sequences = [
        rms,
        zcr,
        centroid,
        bandwidth,
        rolloff
    ]

    all_features = []

    for region in range(N_REGIONS):

        region_features = []

        for sequence in feature_sequences:

            start = int(
                len(sequence) * region / N_REGIONS
            )

            end = int(
                len(sequence) * (region + 1) / N_REGIONS
            )

            segment = sequence[start:end]

            if len(segment) == 0:
                region_features.extend([0.0, 0.0])
            else:
                region_features.extend([
                    float(np.mean(segment)),
                    float(np.std(segment))
                ])

        for band in contrast:

            start = int(
                len(band) * region / N_REGIONS
            )

            end = int(
                len(band) * (region + 1) / N_REGIONS
            )

            segment = band[start:end]

            if len(segment) == 0:
                region_features.extend([0.0, 0.0])
            else:
                region_features.extend([
                    float(np.mean(segment)),
                    float(np.std(segment))
                ])

        all_features.extend(region_features)

    return np.asarray(
        all_features,
        dtype=np.float32
    )


def process_split(split_name):

    metadata_path = Path(
        f"data/metadata/splits/{split_name}.csv"
    )

    output_path = Path(
        "data/features/prosodic_temporal/"
        f"{split_name}_prosodic_temporal.npz"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.read_csv(metadata_path)

    print(f"\nProcessing {split_name}...")
    print(f"Samples: {len(df)}")

    features = []

    for index, row in df.iterrows():

        audio_path = Path(row["audio_path"])

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        feature_vector = extract_temporal_prosodic_features(
            audio_path
        )

        features.append(feature_vector)

        if (index + 1) % 500 == 0:
            print(
                f"Processed {index + 1}/{len(df)}"
            )

    features = np.asarray(
        features,
        dtype=np.float32
    )

    print(
        f"{split_name} feature shape: {features.shape}"
    )

    np.savez_compressed(
        output_path,
        X=features,
        y=df["emotion"].values
    )

    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":

    print("=" * 60)
    print("Temporal Prosodic Feature Extraction")
    print("=" * 60)

    for split in [
        "train",
        "validation",
        "test"
    ]:
        process_split(split)

    print("\nFeature extraction complete.")
