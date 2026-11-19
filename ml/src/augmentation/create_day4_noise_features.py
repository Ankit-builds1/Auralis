from pathlib import Path

import librosa
import numpy as np
import pandas as pd


SAMPLE_RATE = 16000
N_MFCC = 40
N_REGIONS = 10

TRAIN_CSV = Path(
    "data/metadata/splits/train.csv"
)

OUTPUT_FILE = Path(
    "data/features/day4_augmented/"
    "train_noisy_combined.npz"
)


def extract_temporal_features(audio):

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC
    )

    delta = librosa.feature.delta(mfcc)

    delta_delta = librosa.feature.delta(
        mfcc,
        order=2
    )

    combined = np.vstack([
        mfcc,
        delta,
        delta_delta
    ])

    segments = np.array_split(
        combined,
        10,
        axis=1
    )

    features = []

    for segment in segments:

        features.extend(
            np.mean(segment, axis=1)
        )

        features.extend(
            np.std(segment, axis=1)
        )

    return np.asarray(
        features,
        dtype=np.float32
    )


def extract_prosodic_features(audio):

    if len(audio) == 0:

        return np.zeros(
            240,
            dtype=np.float32
        )

    rms = librosa.feature.rms(
        y=audio
    )[0]

    zcr = librosa.feature.zero_crossing_rate(
        audio
    )[0]

    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=SAMPLE_RATE
    )[0]

    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=SAMPLE_RATE
    )[0]

    rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=SAMPLE_RATE
    )[0]

    contrast = librosa.feature.spectral_contrast(
        y=audio,
        sr=SAMPLE_RATE
    )

    sequences = [
        rms,
        zcr,
        centroid,
        bandwidth,
        rolloff
    ]

    features = []

    for region in range(N_REGIONS):

        for sequence in sequences:

            start = int(
                len(sequence)
                * region
                / N_REGIONS
            )

            end = int(
                len(sequence)
                * (region + 1)
                / N_REGIONS
            )

            segment = sequence[start:end]

            if len(segment) == 0:

                features.extend([
                    0.0,
                    0.0
                ])

            else:

                features.extend([
                    float(np.mean(segment)),
                    float(np.std(segment))
                ])

        for band in contrast:

            start = int(
                len(band)
                * region
                / N_REGIONS
            )

            end = int(
                len(band)
                * (region + 1)
                / N_REGIONS
            )

            segment = band[start:end]

            if len(segment) == 0:

                features.extend([
                    0.0,
                    0.0
                ])

            else:

                features.extend([
                    float(np.mean(segment)),
                    float(np.std(segment))
                ])

    return np.asarray(
        features,
        dtype=np.float32
    )


def extract_combined_features(audio):

    temporal = extract_temporal_features(
        audio
    )

    prosodic = extract_prosodic_features(
        audio
    )

    return np.concatenate([
        temporal,
        prosodic
    ]).astype(np.float32)


def add_noise(audio, noise_level=0.02):

    noise = np.random.normal(
        0,
        noise_level,
        size=audio.shape
    ).astype(np.float32)

    return audio + noise


def main():

    print("=" * 70)
    print("AURALIS DAY 4 NOISE AUGMENTATION")
    print("=" * 70)

    df = pd.read_csv(
        TRAIN_CSV
    )

    print(
        f"Training samples: {len(df)}"
    )

    np.random.seed(42)

    features = []
    labels = []

    for index, row in df.iterrows():

        audio_path = Path(
            row["audio_path"]
        )

        audio, _ = librosa.load(
            audio_path,
            sr=SAMPLE_RATE
        )

        noisy_audio = add_noise(
            audio,
            noise_level=0.02
        )

        feature_vector = (
            extract_combined_features(
                noisy_audio
            )
        )

        features.append(
            feature_vector
        )

        labels.append(
            row["emotion"]
        )

        if (index + 1) % 500 == 0:

            print(
                f"Processed "
                f"{index + 1}/{len(df)}"
            )

    X = np.asarray(
        features,
        dtype=np.float32
    )

    y = np.asarray(
        labels
    )

    print(
        "Feature shape:",
        X.shape
    )

    print(
        "Label shape:",
        y.shape
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        OUTPUT_FILE,
        X=X,
        y=y
    )

    print(
        f"Saved: {OUTPUT_FILE}"
    )

    print(
        "\nNoise augmentation complete."
    )


if __name__ == "__main__":
    main()
