from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score, f1_score


SAMPLE_RATE = 16000
N_MFCC = 40
N_REGIONS = 10

TEST_CSV = Path(
    "data/metadata/splits/test.csv"
)

DAY2_SCALER = Path(
    "models/day2_tuned_scaler.joblib"
)

DAY2_MODEL = Path(
    "models/day2_tuned_svm_rbf.joblib"
)

DAY4_SCALER = Path(
    "models/day4_noise_augmented_scaler.joblib"
)

DAY4_MODEL = Path(
    "models/day4_noise_augmented_svm_rbf.joblib"
)

RESULTS_FILE = Path(
    "results/day4/noise_robustness_comparison.csv"
)


def add_noise(audio, noise_level=0.02):
    noise = np.random.normal(
        0,
        noise_level,
        size=audio.shape
    ).astype(np.float32)

    return audio + noise


def extract_temporal_features(audio, sr):
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
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
        N_REGIONS,
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


def extract_prosodic_features(audio, sr):

    rms = librosa.feature.rms(
        y=audio
    )[0]

    zcr = librosa.feature.zero_crossing_rate(
        audio
    )[0]

    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sr
    )[0]

    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sr
    )[0]

    rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sr
    )[0]

    contrast = librosa.feature.spectral_contrast(
        y=audio,
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
                len(sequence) *
                (region + 1) /
                N_REGIONS
            )

            segment = sequence[start:end]

            if len(segment) == 0:
                region_features.extend([
                    0.0,
                    0.0
                ])
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
                len(band) *
                (region + 1) /
                N_REGIONS
            )

            segment = band[start:end]

            if len(segment) == 0:
                region_features.extend([
                    0.0,
                    0.0
                ])
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


def extract_combined_features(audio, sr):

    temporal = extract_temporal_features(
        audio,
        sr
    )

    prosodic = extract_prosodic_features(
        audio,
        sr
    )

    combined = np.concatenate([
        temporal,
        prosodic
    ])

    return combined.astype(
        np.float32
    )


def evaluate_model(
    name,
    model_path,
    scaler_path,
    X,
    y
):

    scaler = load(
        scaler_path
    )

    model = load(
        model_path
    )

    X_scaled = scaler.transform(X)

    predictions = model.predict(
        X_scaled
    )

    accuracy = accuracy_score(
        y,
        predictions
    )

    macro_f1 = f1_score(
        y,
        predictions,
        average="macro"
    )

    weighted_f1 = f1_score(
        y,
        predictions,
        average="weighted"
    )

    return {
        "model": name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    }


def main():

    print("=" * 70)
    print("AURALIS DAY 4 NOISE ROBUSTNESS EVALUATION")
    print("=" * 70)

    np.random.seed(42)

    df = pd.read_csv(
        TEST_CSV
    )

    print(
        f"Test samples: {len(df)}"
    )

    clean_features = []
    noisy_features = []
    labels = []

    for index, row in df.iterrows():

        audio_path = Path(
            row["audio_path"]
        )

        audio, sr = librosa.load(
            audio_path,
            sr=SAMPLE_RATE
        )

        clean = extract_combined_features(
            audio,
            sr
        )

        noisy_audio = add_noise(
            audio,
            noise_level=0.02
        )

        noisy = extract_combined_features(
            noisy_audio,
            sr
        )

        clean_features.append(clean)
        noisy_features.append(noisy)
        labels.append(row["emotion"])

        if (index + 1) % 200 == 0:
            print(
                f"Processed {index + 1}/{len(df)}"
            )

    X_clean = np.asarray(
        clean_features,
        dtype=np.float32
    )

    X_noisy = np.asarray(
        noisy_features,
        dtype=np.float32
    )

    y = np.asarray(
        labels
    )

    print(
        f"\nClean feature shape: {X_clean.shape}"
    )

    print(
        f"Noisy feature shape: {X_noisy.shape}"
    )

    print("\nEvaluating Day 2 baseline...")

    day2_clean = evaluate_model(
        "Day2_Baseline_Clean",
        DAY2_MODEL,
        DAY2_SCALER,
        X_clean,
        y
    )

    day2_noisy = evaluate_model(
        "Day2_Baseline_Noisy",
        DAY2_MODEL,
        DAY2_SCALER,
        X_noisy,
        y
    )

    print(
        f"Day 2 clean accuracy: "
        f"{day2_clean['accuracy']:.4f}"
    )

    print(
        f"Day 2 noisy accuracy: "
        f"{day2_noisy['accuracy']:.4f}"
    )

    print("\nEvaluating Day 4 augmented model...")

    day4_clean = evaluate_model(
        "Day4_Augmented_Clean",
        DAY4_MODEL,
        DAY4_SCALER,
        X_clean,
        y
    )

    day4_noisy = evaluate_model(
        "Day4_Augmented_Noisy",
        DAY4_MODEL,
        DAY4_SCALER,
        X_noisy,
        y
    )

    print(
        f"Day 4 clean accuracy: "
        f"{day4_clean['accuracy']:.4f}"
    )

    print(
        f"Day 4 noisy accuracy: "
        f"{day4_noisy['accuracy']:.4f}"
    )

    results = [
        day2_clean,
        day2_noisy,
        day4_clean,
        day4_noisy
    ]

    results_df = pd.DataFrame(
        results
    )

    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        RESULTS_FILE,
        index=False
    )

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        f"\nSaved: {RESULTS_FILE}"
    )

    print("\nDay 4 noise robustness evaluation complete.")


if __name__ == "__main__":
    main()
