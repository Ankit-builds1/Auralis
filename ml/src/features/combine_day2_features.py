from pathlib import Path

import numpy as np


TEMPORAL_DIR = Path("data/features/temporal")
PROSODIC_DIR = Path("data/features/prosodic_temporal")
OUTPUT_DIR = Path("data/features/day2_combined")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def combine_features(split_name):

    temporal_path = (
        TEMPORAL_DIR /
        f"{split_name}_temporal.npz"
    )

    prosodic_path = (
        PROSODIC_DIR /
        f"{split_name}_prosodic_temporal.npz"
    )

    output_path = (
        OUTPUT_DIR /
        f"{split_name}_combined.npz"
    )

    print(f"\nProcessing {split_name}...")

    temporal_data = np.load(temporal_path)
    prosodic_data = np.load(prosodic_path)

    X_temporal = temporal_data["X"]
    X_prosodic = prosodic_data["X"]

    y_temporal = temporal_data["y"]

    print(f"Temporal features: {X_temporal.shape}")
    print(f"Prosodic features: {X_prosodic.shape}")

    if X_temporal.shape[0] != X_prosodic.shape[0]:
        raise ValueError(
            f"Samples counts do not match for {split_name}"
        )

    X_combined = np.concatenate(
        [X_temporal, X_prosodic],
        axis=1
    )

    print(
        f"Combined features: {X_combined.shape}"
    )

    np.savez_compressed(
        output_path,
        X=X_combined,
        y=y_temporal
    )

    print(f"Saved: {output_path}")


if __name__ == "__main__":

    print("=" * 60)
    print("Day 2 Feature Combination")
    print("=" * 60)

    for split in [
        "train",
        "validation",
        "test"
    ]:
        combine_features(split)

    print("\nFeature combination complete.")
