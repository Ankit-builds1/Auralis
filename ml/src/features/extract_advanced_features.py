from pathlib import Path

import librosa
import numpy as np
import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

TRAIN_CSV = Path("data/metadata/splits/train.csv")
VALIDATION_CSV = Path("data/metadata/splits/validation.csv")
TEST_CSV = Path("data/metadata/splits/test.csv")

OUTPUT_DIR = Path("data/features/advanced")


# --------------------------------------------------
# Audio configuration
# --------------------------------------------------

SAMPLE_RATE = 16000
N_MFCC = 40


# --------------------------------------------------
# Feature extraction
# --------------------------------------------------

def extract_features(audio_path):
    """
    Extract a richer acoustic feature vector.

    Features:
        MFCC mean + std
        Delta MFCC mean + std
        Delta-delta MFCC mean + std
        Zero-crossing rate
        RMS energy
        Spectral centroid
        Spectral bandwidth
        Spectral rolloff
    """

    audio, sample_rate = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True,
    )

    # ----------------------------------------------
    # MFCC
    # ----------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=N_MFCC,
    )

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # ----------------------------------------------
    # Delta MFCC
    # ----------------------------------------------

    delta = librosa.feature.delta(mfcc)

    delta_mean = np.mean(delta, axis=1)
    delta_std = np.std(delta, axis=1)

    # ----------------------------------------------
    # Delta-delta MFCC
    # ----------------------------------------------

    delta_delta = librosa.feature.delta(
        mfcc,
        order=2,
    )

    delta_delta_mean = np.mean(
        delta_delta,
        axis=1,
    )

    delta_delta_std = np.std(
        delta_delta,
        axis=1,
    )

    # ----------------------------------------------
    # Zero crossing rate
    # ----------------------------------------------

    zcr = librosa.feature.zero_crossing_rate(audio)

    zcr_features = np.array(
        [
            np.mean(zcr),
            np.std(zcr),
        ]
    )

    # ----------------------------------------------
    # RMS energy
    # ----------------------------------------------

    rms = librosa.feature.rms(y=audio)

    rms_features = np.array(
        [
            np.mean(rms),
            np.std(rms),
        ]
    )

    # ----------------------------------------------
    # Spectral centroid
    # ----------------------------------------------

    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sample_rate,
    )

    centroid_features = np.array(
        [
            np.mean(centroid),
            np.std(centroid),
        ]
    )

    # ----------------------------------------------
    # Spectral bandwidth
    # ----------------------------------------------

    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sample_rate,
    )

    bandwidth_features = np.array(
        [
            np.mean(bandwidth),
            np.std(bandwidth),
        ]
    )

    # ----------------------------------------------
    # Spectral rolloff
    # ----------------------------------------------

    rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sample_rate,
    )

    rolloff_features = np.array(
        [
            np.mean(rolloff),
            np.std(rolloff),
        ]
    )

    # ----------------------------------------------
    # Combine everything
    # ----------------------------------------------

    features = np.concatenate(
        [
            mfcc_mean,
            mfcc_std,
            delta_mean,
            delta_std,
            delta_delta_mean,
            delta_delta_std,
            zcr_features,
            rms_features,
            centroid_features,
            bandwidth_features,
            rolloff_features,
        ]
    )

    return features


# --------------------------------------------------
# Process dataset split
# --------------------------------------------------

def process_split(csv_path, output_name):

    print("\nProcessing:", csv_path)

    df = pd.read_csv(csv_path)

    features = []
    labels = []
    actor_ids = []
    filenames = []

    failed_files = []

    for index, row in df.iterrows():

        audio_path = Path(row["audio_path"])

        try:

            feature_vector = extract_features(
                audio_path
            )

            features.append(feature_vector)
            labels.append(row["emotion"])
            actor_ids.append(row["actor_id"])
            filenames.append(row["filename"])

        except Exception as error:

            print(
                f"Failed: {audio_path} | {error}"
            )

            failed_files.append(
                {
                    "filename": row["filename"],
                    "error": str(error),
                }
            )

        if (index + 1) % 500 == 0:

            print(
                f"Processed "
                f"{index + 1}/{len(df)} files"
            )

    X = np.array(features)
    y = np.array(labels)

    output_path = (
        OUTPUT_DIR /
        f"{output_name}_advanced.npz"
    )

    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        actor_ids=np.array(actor_ids),
        filenames=np.array(filenames),
    )

    print("\nCompleted:", output_name)
    print("Feature shape:", X.shape)
    print("Label shape:", y.shape)
    print("Saved:", output_path)

    if failed_files:

        failed_path = (
            OUTPUT_DIR /
            f"{output_name}_failed.csv"
        )

        pd.DataFrame(
            failed_files
        ).to_csv(
            failed_path,
            index=False,
        )

        print(
            "Failed files:",
            len(failed_files),
        )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    process_split(
        TRAIN_CSV,
        "train",
    )

    process_split(
        VALIDATION_CSV,
        "validation",
    )

    process_split(
        TEST_CSV,
        "test",
    )

    print("\n")
    print("=" * 60)
    print("ADVANCED FEATURE EXTRACTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()