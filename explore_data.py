"""
Data exploration script.

Run this to understand the dataset structure before diving into modeling.
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path to import utils
sys.path.insert(0, "src")
from src.utils import extract_date_from_recording_id


def explore_dataset(data_path="data/dataset.parquet"):
    """Explore the dataset and print useful statistics."""

    print("=" * 70)
    print("DATASET EXPLORATION")
    print("=" * 70)

    # Load data
    df = pd.read_parquet(data_path)
    print(f"\nDataset shape: {df.shape}")
    print(f"Total recordings: {len(df):,}")

    # Patient information
    print(f"\n{'=' * 70}")
    print("PATIENT INFORMATION")
    print("=" * 70)
    unique_patients = df["patient_short_id"].unique()
    print(f"Number of patients: {len(unique_patients)}")

    # Recordings per patient
    recordings_per_patient = df.groupby("patient_short_id").size()
    print("\nRecordings per patient:")
    print(f"  Mean: {recordings_per_patient.mean():.1f}")
    print(f"  Median: {recordings_per_patient.median():.1f}")
    print(f"  Min: {recordings_per_patient.min()}")
    print(f"  Max: {recordings_per_patient.max()}")

    # Label distribution
    print(f"\n{'=' * 70}")
    print("LABEL DISTRIBUTION")
    print("=" * 70)
    label_counts = df["label"].value_counts()
    print("Overall:")
    print(f"  Stable (0): {label_counts[0]:,} ({label_counts[0] / len(df) * 100:.1f}%)")
    print(
        f"  Pre-hospitalization (1): {label_counts[1]:,} ({label_counts[1] / len(df) * 100:.1f}%)"
    )

    print("\nPer patient:")
    for patient in sorted(unique_patients):
        patient_df = df[df["patient_short_id"] == patient]
        patient_labels = patient_df["label"].value_counts()
        n_stable = patient_labels.get(0, 0)
        n_prehosp = patient_labels.get(1, 0)
        print(
            f"  {patient}: {n_stable} stable, {n_prehosp} pre-hosp "
            f"({n_prehosp / (n_stable + n_prehosp) * 100:.1f}% positive)"
        )

    # Temporal information
    print(f"\n{'=' * 70}")
    print("TEMPORAL INFORMATION")
    print("=" * 70)

    # Extract dates from recording_id
    df["recording_date"] = extract_date_from_recording_id(df["recording_id"])

    print("\nRecording date range:")
    print(f"  First recording: {df['recording_date'].min().date()}")
    print(f"  Last recording: {df['recording_date'].max().date()}")
    print(
        f"  Total span: {(df['recording_date'].max() - df['recording_date'].min()).days} days"
    )

    print("\nFollow-up period per patient:")
    for patient in sorted(unique_patients):
        patient_df = df[df["patient_short_id"] == patient]
        first_date = patient_df["recording_date"].min()
        last_date = patient_df["recording_date"].max()
        follow_up_days = (last_date - first_date).days
        n_recordings = len(patient_df)
        print(
            f"  {patient}: {follow_up_days} days ({first_date.date()} to {last_date.date()}), {n_recordings} recordings"
        )

    # Feature information
    print(f"\n{'=' * 70}")
    print("FEATURE INFORMATION")
    print("=" * 70)

    metadata_cols = [
        "recording_id",
        "patient_short_id",
        "label",
        "recording_date",  # Added by our extraction
    ]
    feature_cols = [col for col in df.columns if col not in metadata_cols]

    print(f"Number of acoustic features: {len(feature_cols)}")
    print("\nFeature categories:")

    # Count feature types
    feature_types = {}
    for col in feature_cols:
        feature_type = col.split("_")[0]
        feature_types[feature_type] = feature_types.get(feature_type, 0) + 1

    for ftype, count in sorted(feature_types.items()):
        print(f"  {ftype}: {count} features")

    # Check for missing values
    print(f"\n{'=' * 70}")
    print("DATA QUALITY")
    print("=" * 70)

    missing_counts = df[feature_cols].isnull().sum()
    if missing_counts.sum() > 0:
        print(f"Features with missing values: {(missing_counts > 0).sum()}")
        print(f"Total missing values: {missing_counts.sum()}")
    else:
        print("No missing values in features!")

    # Create visualizations
    print(f"\n{'=' * 70}")
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    fig, axes = plt.subplots(2, 1, figsize=(15, 12))

    # 1. Label distribution per patient
    ax = axes[0]
    label_dist = df.groupby(["patient_short_id", "label"]).size().unstack(fill_value=0)
    label_dist.plot(kind="bar", stacked=True, ax=ax, color=["steelblue", "red"])
    ax.set_title("Label Distribution per Patient", fontsize=14, fontweight="bold")
    ax.set_xlabel("Patient ID")
    ax.set_ylabel("Number of Recordings")
    ax.legend(["Stable", "Pre-hospitalization"])
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # 2. Temporal distribution per patient timeline
    ax = axes[1]
    for i, patient in enumerate(sorted(unique_patients)):
        patient_df = df[df["patient_short_id"] == patient].sort_values("recording_date")
        # Normalize dates to days since first recording for this patient
        first_date = patient_df["recording_date"].min()
        patient_df["days_since_start"] = (
            patient_df["recording_date"] - first_date
        ).dt.days

        colors = ["steelblue" if label == 0 else "red" for label in patient_df["label"]]
        ax.scatter(
            patient_df["days_since_start"],
            [i] * len(patient_df),
            c=colors,
            alpha=0.6,
            s=30,
            label=patient if i < 5 else None,  # Only label first 5 for legend
        )

    ax.set_title(
        "Patient Timelines (Days Since First Recording)", fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Days Since First Recording")
    ax.set_ylabel("Patient (ordered)")
    ax.set_yticks(range(len(unique_patients)))
    ax.set_yticklabels([f"P{i}" for i in range(len(unique_patients))], fontsize=8)
    if len(unique_patients) <= 5:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    # Add custom legend for colors
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="steelblue", alpha=0.6, label="Stable"),
        Patch(facecolor="red", alpha=0.6, label="Pre-hospitalization"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("data_exploration.png", dpi=300, bbox_inches="tight")
    print("\nVisualization saved to: data_exploration.png")
    plt.show()

    print(f"\n{'=' * 70}")
    print("EXPLORATION COMPLETE")
    print("=" * 70)
    print("\nKey takeaways:")
    print("1. Dataset is imbalanced (more stable than pre-hospitalization recordings)")
    print("2. Patients have varying numbers of recordings and follow-up periods")
    print("3. Temporal information is available - use it for modeling!")
    print("4. Recording dates extracted from recording_id show clear temporal patterns")
    print("\nNext steps:")
    print("- Run: python src/train.py (to train baseline model)")
    print("- Modify: src/models.py (to improve the embedding model)")
    print("- Experiment with temporal modeling and patient-specific features!")


if __name__ == "__main__":
    explore_dataset()
