"""
Utility functions for evaluation and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Union
from datetime import datetime


def extract_date_from_recording_id(
    recording_id: Union[str, pd.Series],
) -> Union[datetime, pd.Series]:
    """
    Extract date from recording_id.

    The recording_id format is "patient_XXXX/YYYY-MM-DD" where YYYY-MM-DD is the date.

    Parameters
    ----------
    recording_id : str or pd.Series
        Recording ID(s) in format "patient_XXXX/YYYY-MM-DD".

    Returns
    -------
    datetime or pd.Series
        Extracted date(s) as datetime object(s).

    Examples
    --------
    >>> extract_date_from_recording_id("patient_0000/2021-09-17")
    datetime.datetime(2021, 9, 17, 0, 0)

    >>> df['date'] = extract_date_from_recording_id(df['recording_id'])
    """
    if isinstance(recording_id, pd.Series):
        return (
            recording_id.str.split("/")
            .str[1]
            .apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
        )
    else:
        date_str = recording_id.split("/")[1]
        return datetime.strptime(date_str, "%Y-%m-%d")


def aggregate_predictions_per_recording(
    recording_ids: np.ndarray, y_pred: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Average prediction values per recording_id.
    
    Parameters
    ----------
    recording_ids : np.ndarray
        Array of recording identifiers.
    y_pred : np.ndarray
        Predicted probabilities or scores.
        
    Returns
    -------
    unique_recording_ids : np.ndarray
        Unique recording IDs.
    averaged_predictions : np.ndarray
        Averaged predictions per recording.
    """
    df = pd.DataFrame({'recording_id': recording_ids, 'prediction': y_pred})
    aggregated = df.groupby('recording_id')['prediction'].mean()
    return aggregated.index.values, aggregated.values


def compute_per_patient_auc(
    patient_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    recording_ids: np.ndarray = None,
) -> Dict[str, float]:
    """
    Compute ROC AUC score for each patient individually.
    
    If recording_ids are provided, predictions are first averaged per recording
    before computing AUC at the recording level.

    Parameters
    ----------
    patient_ids : np.ndarray
        Array of patient identifiers for each sample.
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_pred : np.ndarray
        Predicted probabilities or scores.
    recording_ids : np.ndarray, optional
        Array of recording identifiers. If provided, predictions are averaged
        per recording before computing AUC.

    Returns
    -------
    dict
        Dictionary mapping patient_id (str) to their ROC AUC score (float).
        Patients with only one class in y_true will have AUC = None.

    Notes
    -----
    ROC AUC requires at least one sample of each class. If a patient has only
    positive or only negative samples, their AUC cannot be computed and will
    be set to None.
    """
    per_patient_auc = {}

    unique_patients = np.unique(patient_ids)
    for patient in unique_patients:
        # Get indices for this patient
        patient_mask = patient_ids == patient
        patient_y_true = y_true[patient_mask]
        patient_y_pred = y_pred[patient_mask]
        patient_recording_ids = recording_ids[patient_mask] if recording_ids is not None else None

        # If recording_ids provided, average predictions per recording
        if patient_recording_ids is not None:
            # Create a DataFrame for aggregation
            df = pd.DataFrame({
                'recording_id': patient_recording_ids,
                'label': patient_y_true,
                'prediction': patient_y_pred
            })
            # Average predictions per recording, take first label (should be same for all)
            agg_df = df.groupby('recording_id').agg({
                'label': 'first',
                'prediction': 'mean'
            }).reset_index()
            
            patient_y_true = agg_df['label'].values
            patient_y_pred = agg_df['prediction'].values

        # Check if we have both classes (needed for AUC)
        if len(np.unique(patient_y_true)) < 2:
            per_patient_auc[patient] = None
        else:
            try:
                auc = roc_auc_score(patient_y_true, patient_y_pred)
                per_patient_auc[patient] = auc
            except ValueError:
                per_patient_auc[patient] = None

    return per_patient_auc


def aggregate_patient_aucs(
    per_patient_auc: Dict[str, float],
) -> Tuple[float, float, List[float]]:
    """
    Aggregate per-patient AUC scores by computing mean and standard deviation.

    Parameters
    ----------
    per_patient_auc : dict
        Dictionary mapping patient_id to their ROC AUC score.

    Returns
    -------
    mean_auc : float
        Mean ROC AUC across all patients (excluding None values).
    std_auc : float
        Standard deviation of ROC AUC across all patients.
    valid_aucs : list
        List of valid AUC scores (non-None values).

    Notes
    -----
    Patients with None AUC values (due to having only one class) are excluded
    from the aggregation.
    """
    # Filter out None values
    valid_aucs = [auc for auc in per_patient_auc.values() if auc is not None]

    if len(valid_aucs) == 0:
        return 0.0, 0.0, []

    mean_auc = np.mean(valid_aucs)
    std_auc = np.std(valid_aucs)

    return mean_auc, std_auc, valid_aucs


def plot_embeddings_2d(
    embeddings: np.ndarray,
    labels: np.ndarray,
    patient_ids: np.ndarray = None,
    title: str = "Embedding Space (PCA)",
    save_path: str = None,
) -> plt.Figure:
    """
    Visualize embeddings in 2D using PCA.

    Parameters
    ----------
    embeddings : np.ndarray
        Embedding vectors of shape (n_samples, embedding_dim).
    labels : np.ndarray
        Binary labels (0 or 1) of shape (n_samples,).
    patient_ids : np.ndarray, optional
        Patient identifiers for coloring by patient. If None, only labels are used.
    title : str, default="Embedding Space (PCA)"
        Title for the plot.
    save_path : str, optional
        If provided, save the figure to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object.

    Notes
    -----
    This function reduces the embedding dimension to 2D using PCA for visualization.
    Points are colored by their label (stable vs. pre-hospitalization).
    """
    # Apply PCA to reduce to 2D
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each class with different colors
    colors = {0: "blue", 1: "red"}
    labels_map = {0: "Stable", 1: "Pre-hospitalization"}

    for label in [0, 1]:
        mask = labels == label
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=colors[label],
            label=labels_map[label],
            alpha=0.6,
            s=20,
        )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def compute_random_baseline(
    patient_ids: np.ndarray,
    y_true: np.ndarray,
    recording_ids: np.ndarray = None,
    n_iterations: int = 100,
    random_seed: int = 42,
) -> Tuple[float, float, List[float]]:
    """
    Compute random baseline performance using Monte Carlo sampling.
    
    Parameters
    ----------
    patient_ids : np.ndarray
        Array of patient identifiers for each sample.
    y_true : np.ndarray
        True binary labels (0 or 1).
    recording_ids : np.ndarray, optional
        Array of recording identifiers. If provided, random predictions are
        generated at the recording level.
    n_iterations : int, default=100
        Number of Monte Carlo iterations.
    random_seed : int, default=42
        Random seed for reproducibility.
        
    Returns
    -------
    mean_auc : float
        Mean ROC AUC of random baseline across iterations.
    std_auc : float
        Standard deviation of ROC AUC across iterations.
    all_mean_aucs : list
        List of mean AUC values from all iterations.
    """
    np.random.seed(random_seed)
    all_mean_aucs = []
    
    for i in range(n_iterations):
        # Generate random predictions
        random_predictions = np.random.rand(len(y_true))
        
        # Compute per-patient AUC
        per_patient_auc = compute_per_patient_auc(
            patient_ids, y_true, random_predictions, recording_ids
        )
        
        # Aggregate
        _, _, valid_aucs = aggregate_patient_aucs(per_patient_auc)
        if len(valid_aucs) > 0:
            all_mean_aucs.append(np.mean(valid_aucs))
    
    return np.mean(all_mean_aucs), np.std(all_mean_aucs), all_mean_aucs


def print_evaluation_results(
    per_patient_auc: Dict[str, float],
    mean_auc: float,
    std_auc: float,
    fold_name: str = "Test",
    random_baseline: Tuple[float, float] = None,
) -> None:
    """
    Pretty print evaluation results.

    Parameters
    ----------
    per_patient_auc : dict
        Dictionary mapping patient_id to their ROC AUC score.
    mean_auc : float
        Mean ROC AUC across all patients.
    std_auc : float
        Standard deviation of ROC AUC.
    fold_name : str, default="Test"
        Name of the evaluation fold (for display purposes).
    random_baseline : tuple of (float, float), optional
        Random baseline (mean, std) for comparison.
    """
    print(f"\n{'=' * 60}")
    print(f"{fold_name} Set - Per-Patient ROC AUC Scores")
    print(f"{'=' * 60}")

    for patient_id, auc in sorted(per_patient_auc.items()):
        if auc is not None:
            print(f"  {patient_id}: {auc:.4f}")
        else:
            print(f"  {patient_id}: N/A (only one class present)")

    print(f"\n{'-' * 60}")
    print(f"Model Performance:")
    print(f"  Mean ROC AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    
    if random_baseline is not None:
        random_mean, random_std = random_baseline
        print(f"\nRandom Baseline (Monte Carlo, N=100):")
        print(f"  Mean ROC AUC: {random_mean:.4f} ± {random_std:.4f}")
        print(f"\nImprovement over random: {mean_auc - random_mean:.4f}")
    
    print(f"{'=' * 60}\n")
