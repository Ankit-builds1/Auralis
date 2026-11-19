from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score, f1_score


SAMPLE_RATE = 16000
N_MFCC = 40
N_REGIONS = 10

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
RESULT_DIR = Path("results/robustness")

TEST_CSV = DATA_DIR / "metadata/splits/test.csv"

SCALER_FILE = MODEL_DIR / "day2_tuned_scaler.joblib"
MODEL_FILE = MODEL_DIR / "day2_tuned_svm_rbf.joblib"


# =========================================================
# TEMPORAL FEATURES
# Exact same logic as extract_temporal_features.py
# =========================================================

def extract_temporal_features_from_audio(audio):

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

    # IMPORTANT:
    # Must match the original training extractor.
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


# =========================================================
# PROSODIC FEATURES
# Exact same logic as extract_prosodic_temporal.py
# =========================================================

def extract_prosodic_features_from_audio(audio):

    if len(audio) == 0:
        return np.zeros(
            240,
            dtype=np.float32
        )

    y = audio
    sr = SAMPLE_RATE

    rms = librosa.feature.rms(
        y=y
    )[0]

    zcr = librosa.feature.zero_crossing_rate(
        y
    )[0]

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

                region_features.extend([
                    0.0,
                    0.0
                ])

            else:

                region_features.extend([
                    float(np.mean(segment)),
                    float(np.std(segment))
                ])

        all_features.extend(
            region_features
        )

    return np.asarray(
        all_features,
        dtype=np.float32
    )


# =========================================================
# COMBINE FEATURES
# 2400 temporal + 240 prosodic = 2640
# =========================================================

def extract_combined_features(audio):

    temporal = extract_temporal_features_from_audio(
        audio
    )

    prosodic = extract_prosodic_features_from_audio(
        audio
    )

    combined = np.concatenate([
        temporal,
        prosodic
    ])

    return combined.astype(
        np.float32
    )


# =========================================================
# AUDIO TRANSFORMS
# =========================================================

def clean(audio):

    return audio


def volume_down(audio):

    return audio * 0.5


def volume_up(audio):

    return audio * 1.5


def add_silence(audio):

    silence = np.zeros(
        int(SAMPLE_RATE * 0.25),
        dtype=np.float32
    )

    return np.concatenate([
        silence,
        audio,
        silence
    ])


def shorten(audio):

    length = int(
        len(audio) * 0.75
    )

    start = (
        len(audio) - length
    ) // 2

    return audio[
        start:start + length
    ]


def extend(audio):

    target_length = int(
        len(audio) * 1.25
    )

    extra = target_length - len(audio)

    left = extra // 2
    right = extra - left

    return np.pad(
        audio,
        (left, right),
        mode="constant"
    )


def normalize(audio):

    peak = np.max(
        np.abs(audio)
    )

    if peak == 0:

        return audio

    return audio / peak


def add_noise(audio):

    noise = np.random.normal(
        0,
        0.02,
        size=audio.shape
    ).astype(
        np.float32
    )

    return audio + noise


# =========================================================
# EVALUATION
# =========================================================

def evaluate_condition(
    name,
    transform,
    audio_paths,
    labels,
    scaler,
    model
):

    print()
    print("=" * 60)
    print(f"CONDITION: {name}")
    print("=" * 60)

    features = []

    total = len(audio_paths)

    for index, path in enumerate(audio_paths):

        audio, _ = librosa.load(
            path,
            sr=SAMPLE_RATE
        )

        transformed_audio = transform(
            audio
        )

        feature_vector = extract_combined_features(
            transformed_audio
        )

        features.append(
            feature_vector
        )

        if (index + 1) % 100 == 0:

            print(
                f"Processed "
                f"{index + 1}/{total}"
            )

    X = np.asarray(
        features,
        dtype=np.float32
    )

    print(
        "Feature shape:",
        X.shape
    )

    X_scaled = scaler.transform(
        X
    )

    predictions = model.predict(
        X_scaled
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro"
    )

    weighted_f1 = f1_score(
        labels,
        predictions,
        average="weighted"
    )

    print(
        f"Accuracy:    {accuracy:.4f}"
    )

    print(
        f"Macro F1:    {macro_f1:.4f}"
    )

    print(
        f"Weighted F1: {weighted_f1:.4f}"
    )

    return {
        "condition": name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    }


# =========================================================
# MAIN
# =========================================================

def main():

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 60)
    print("AURALIS DAY 3 ROBUSTNESS TEST")
    print("=" * 60)

    print(
        "\nLoading test metadata..."
    )

    test_df = pd.read_csv(
        TEST_CSV
    )

    audio_paths = (
        test_df["audio_path"].tolist()
    )

    labels = (
        test_df["emotion"].to_numpy()
    )

    print(
        f"Test samples: {len(audio_paths)}"
    )

    print(
        "\nLoading scaler..."
    )

    scaler = load(
        SCALER_FILE
    )

    print(
        "Loading model..."
    )

    model = load(
        MODEL_FILE
    )

    np.random.seed(42)

    # -----------------------------------------------------
    # FIRST RUN ONLY CLEAN.
    #
    # After clean reaches approximately:
    # Accuracy  ~0.5035
    # Macro F1  ~0.4976
    #
    # uncomment the remaining conditions.
    # -----------------------------------------------------

    conditions = {

        "clean": clean,

        "volume_0.5": volume_down,
        "volume_1.5": volume_up,
        "silence_0.25s": add_silence,
        "short_75pct": shorten,
        "extended_125pct": extend,
        "normalized": normalize,
        "noise_0.02": add_noise,
    }

    results = []

    for name, transform in conditions.items():

        result = evaluate_condition(
            name=name,
            transform=transform,
            audio_paths=audio_paths,
            labels=labels,
            scaler=scaler,
            model=model
        )

        results.append(
            result
        )

    results_df = pd.DataFrame(
        results
    )

    output_file = (
        RESULT_DIR
        / "day3_robustness_results.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print()
    print("=" * 60)
    print("DAY 3 ROBUSTNESS TEST COMPLETE")
    print("=" * 60)

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        f"\nSaved results to: {output_file}"
    )


if __name__ == "__main__":
    main()
