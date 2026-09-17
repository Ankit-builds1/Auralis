from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


# Input metadata
METADATA_PATH = Path("data/metadata/labels.csv")

# Output directory
OUTPUT_DIR = Path("data/metadata/splits")


def split_dataset():
    print("Auralis ML Dataset Splitting")
    print("-" * 40)

    # Load metadata
    df = pd.read_csv(METADATA_PATH)

    print("Total audio files:", len(df))
    print("Total unique actors:", df["actor_id"].nunique())

    # First split:
    # 70% training, 30% temporary data
    first_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=42,
    )

    train_indices, temporary_indices = next(
        first_split.split(
            df,
            groups=df["actor_id"],
        )
    )

    train_df = df.iloc[train_indices].copy()
    temporary_df = df.iloc[temporary_indices].copy()

    # Second split:
    # Temporary data becomes:
    # 15% validation, 15% test
    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=42,
    )

    validation_indices, test_indices = next(
        second_split.split(
            temporary_df,
            groups=temporary_df["actor_id"],
        )
    )

    validation_df = temporary_df.iloc[validation_indices].copy()
    test_df = temporary_df.iloc[test_indices].copy()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save CSV files
    train_path = OUTPUT_DIR / "train.csv"
    validation_path = OUTPUT_DIR / "validation.csv"
    test_path = OUTPUT_DIR / "test.csv"

    train_df.to_csv(train_path, index=False)
    validation_df.to_csv(validation_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Display summary
    print("\nSplit completed successfully.")
    print("-" * 40)

    print("Training files:", len(train_df))
    print("Validation files:", len(validation_df))
    print("Test files:", len(test_df))

    print("\nTraining actors:", train_df["actor_id"].nunique())
    print("Validation actors:", validation_df["actor_id"].nunique())
    print("Test actors:", test_df["actor_id"].nunique())

    # Verify actor separation
    train_actors = set(train_df["actor_id"])
    validation_actors = set(validation_df["actor_id"])
    test_actors = set(test_df["actor_id"])

    assert train_actors.isdisjoint(validation_actors)
    assert train_actors.isdisjoint(test_actors)
    assert validation_actors.isdisjoint(test_actors)

    print("\nActor separation check: PASSED")

    print("\nFiles saved:")
    print(train_path)
    print(validation_path)
    print(test_path)


if __name__ == "__main__":
    split_dataset()