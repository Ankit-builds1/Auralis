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

OUTPUT_DIR = Path("data/features")


# --------------------------------------------------
# Audio / MFCC configuration
# --------------------------------------------------

SAMPLE_RATE = 16000
N_MFCC = 40


# --------------------------------------------------
# MFCC extraction
# --------------------------------------------------

def extract_mfcc(audio_path):
    """
    Load an audio file and extract mean MFCC features.

    Returns:
        numpy array with shape (40,)
    """

    audio, sample_rate = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True,
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=N_MFCC,
    )

    # Convert variable-length audio into
    # a fixed-size feature vector.
    mfcc_mean = np.mean(mfcc, axis=1)

    return mfcc_mean


# --------------------------------------------------
# Process a dataset split
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

            mfcc = extract_mfcc(audio_path)

            features.append(mfcc)
            labels.append(row["emotion"])
            actor_ids.append(row["actor_id"])
            filenames.append(row["filename"])

        except Exception as error:

            print(
                f"Failed: {audio_path} | Error: {error}"
            )

            failed_files.append(
                {
                    "filename": row["filename"],
                    "error": str(error),
                }
            )

        # Progress indicator
        if (index + 1) % 500 == 0:
            print(
                f"Processed {index + 1}/{len(df)} files"
            )

    # Convert to numpy arrays
    X = np.array(features)
    y = np.array(labels)

    # Save features
    output_path = OUTPUT_DIR / f"{output_name}_mfcc.npz"

    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        actor_ids=np.array(actor_ids),
        filenames=np.array(filenames),
    )

    print("\nCompleted:", output_name)
    print("Features shape:", X.shape)
    print("Labels shape:", y.shape)
    print("Saved:", output_path)

    if failed_files:
        print(
            f"WARNING: {len(failed_files)} files failed."
        )

        failed_path = (
            OUTPUT_DIR / f"{output_name}_failed.csv"
        )

        pd.DataFrame(failed_files).to_csv(
            failed_path,
            index=False,
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
    print("=" * 50)
    print("MFCC FEATURE EXTRACTION COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()