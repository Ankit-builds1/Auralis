from pathlib import Path
import pandas as pd


# CREMA-D dataset location
DATASET_PATH = Path("data/raw/CREMA-D")

# Output CSV location
OUTPUT_PATH = Path("data/metadata/labels.csv")


# CREMA-D emotion codes
EMOTION_MAP = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}


def create_metadata():
    audio_files = sorted(DATASET_PATH.rglob("*.wav"))

    print("Auralis ML Metadata Creation")
    print("Total WAV files found:", len(audio_files))

    rows = []
    skipped_files = []

    for audio_path in audio_files:
        filename = audio_path.name
        file_stem = audio_path.stem

        # Example filename:
        # 1001_DFA_ANG_XX.wav
        #
        # Parts:
        # 1001 = actor ID
        # DFA  = sentence code
        # ANG  = emotion code
        # XX   = intensity code

        parts = file_stem.split("_")

        if len(parts) < 3:
            skipped_files.append(filename)
            continue

        actor_id = parts[0]
        sentence_code = parts[1]
        emotion_code = parts[2].upper()

        if emotion_code not in EMOTION_MAP:
            skipped_files.append(filename)
            continue

        emotion = EMOTION_MAP[emotion_code]

        rows.append(
            {
                "filename": filename,
                "audio_path": audio_path.as_posix(),
                "actor_id": actor_id,
                "sentence_code": sentence_code,
                "emotion_code": emotion_code,
                "emotion": emotion,
            }
        )

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Create output directory if it does not exist
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save metadata CSV
    df.to_csv(OUTPUT_PATH, index=False)

    print("\nMetadata creation completed.")
    print("Valid audio files:", len(df))
    print("Skipped files:", len(skipped_files))
    print("CSV saved at:", OUTPUT_PATH)

    print("\nEmotion distribution:")
    print(df["emotion"].value_counts().sort_index())


if __name__ == "__main__":
    create_metadata()