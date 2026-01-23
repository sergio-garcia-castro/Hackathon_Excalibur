"""
Advanced dataset exploration for LOPO voice HF prediction.

Focus:
- leakage risk (patient identity / age / sex)
- temporal structure and label episodes
- feature pathologies (constant, outliers, redundancy)
- within-patient univariate signal stable across patients

Outputs:
- figures/*.png
- eda_summary.csv
- top_features_by_within_patient_auc.csv
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Add src to path to import utils
import sys

sys.path.insert(0, "src")
from utils import extract_date_from_recording_id


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def get_numeric_X(df: pd.DataFrame, cols: list, dtype=np.float64):
    """
    Always returns a *true numeric* ndarray (never dtype=object).
    Any non-numeric values are coerced to NaN first.
    """
    X_df = df[cols].apply(pd.to_numeric, errors="coerce")
    X_df = X_df.replace([np.inf, -np.inf], np.nan)

    nan_frac = X_df.isna().mean()
    bad_cols = nan_frac[nan_frac > 0].sort_values(ascending=False)

    # IMPORTANT: force a numeric ndarray (prevents object-dtype ufunc crashes)
    X = X_df.to_numpy()
    X = np.asarray(X, dtype=dtype)  # hard-cast

    return X, bad_cols


def basic_integrity(df: pd.DataFrame, feature_cols: list) -> dict:
    out = {}

    out["n_rows"] = len(df)
    out["n_patients"] = df["patient_short_id"].nunique()
    out["n_recording_ids_unique"] = df["recording_id"].nunique()
    out["recording_id_duplicates"] = len(df) - df["recording_id"].nunique()
    out["duplicate_full_rows"] = int(df.duplicated().sum())

    # --- NEW: force numeric, and detect non-numeric columns ---
    X_df = df[feature_cols].copy()
    non_numeric_cols = [
        c for c in feature_cols if not pd.api.types.is_numeric_dtype(X_df[c])
    ]
    out["non_numeric_feature_cols"] = len(non_numeric_cols)

    # Convert everything to numeric; non-parsable becomes NaN
    X_num = X_df.apply(pd.to_numeric, errors="coerce")
    out["values_coerced_to_nan"] = int(X_num.isna().sum().sum())

    X = X_num.to_numpy(dtype=np.float32)  # now numeric
    out["non_finite_values"] = int(np.sum(~np.isfinite(X)))

    # Constant / near-constant
    stds = np.nanstd(X, axis=0)
    out["constant_features_std0"] = int(np.sum(stds == 0))
    out["near_constant_std_lt_1e-6"] = int(np.sum(stds < 1e-6))

    # Zero fraction per feature
    zero_frac = np.mean(X == 0.0, axis=0)
    out["features_zero_frac_gt_0p95"] = int(np.sum(zero_frac > 0.95))

    # Save a quick report of offenders (optional)
    if non_numeric_cols:
        out["non_numeric_examples"] = ", ".join(non_numeric_cols[:10])
    else:
        out["non_numeric_examples"] = ""

    return out


def patient_metadata_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Checks whether age/sex are constant per patient.
    If they are, they can act like patient identifiers.
    """
    cols = [c for c in ["age", "sex"] if c in df.columns]
    if not cols:
        return pd.DataFrame()

    rows = []
    for pid, g in df.groupby("patient_short_id"):
        row = {"patient_short_id": pid}
        for c in cols:
            row[f"{c}_n_unique"] = g[c].nunique(dropna=False)
            # show the first few unique values (debug)
            vals = g[c].dropna().unique()
            row[f"{c}_example_vals"] = ", ".join(map(str, vals[:5]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("patient_short_id")


def recordings_per_day(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["patient_short_id", "recording_date"]).size().reset_index(name="n")
    return g


def label_episodes(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each patient: sort by date, then find runs of consecutive labels.
    Since we only have dates (not times), an "episode" is a run in sorted order.
    """
    rows = []
    for pid, g in df.sort_values("recording_date").groupby("patient_short_id"):
        y = g["label"].to_numpy()
        if len(y) == 0:
            continue

        # run-length encoding
        starts = [0]
        for i in range(1, len(y)):
            if y[i] != y[i - 1]:
                starts.append(i)
        starts.append(len(y))

        for j in range(len(starts) - 1):
            a, b = starts[j], starts[j + 1]
            lab = int(y[a])
            seg = g.iloc[a:b]
            rows.append(
                {
                    "patient_short_id": pid,
                    "label": lab,
                    "n_recordings": len(seg),
                    "start_date": seg["recording_date"].iloc[0].date(),
                    "end_date": seg["recording_date"].iloc[-1].date(),
                    "span_days": int(
                        (
                            seg["recording_date"].iloc[-1]
                            - seg["recording_date"].iloc[0]
                        ).days
                    ),
                }
            )
    return pd.DataFrame(rows)


def rolling_positive_rate_plot(df: pd.DataFrame, outdir: str, window_days: int = 14):
    """
    Rolling positive rate per patient over time.
    We aggregate per day, then compute rolling mean of daily positive fraction.
    """
    ensure_dir(outdir)
    for pid, g in df.groupby("patient_short_id"):
        daily = g.groupby("recording_date")["label"].agg(["mean", "count"]).sort_index()
        # window in "days": since index is daily-ish, use rolling with a window size in rows
        # approximate: treat each unique date as a step
        w = min(window_days, len(daily))
        daily["roll_pos_rate"] = (
            daily["mean"].rolling(window=w, min_periods=max(2, w // 4)).mean()
        )

        plt.figure(figsize=(10, 4))
        plt.plot(daily.index, daily["roll_pos_rate"])
        plt.title(f"Rolling positive rate (approx {w}-day window) - {pid}")
        plt.xlabel("Date")
        plt.ylabel("Rolling fraction label=1")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"rolling_pos_rate_{pid}.png"), dpi=200)
        plt.close()


def feature_scale_report(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    X, bad_cols = get_numeric_X(df, feature_cols, dtype=np.float64)

    if not bad_cols.empty:
        print("\n[WARN] Some columns produced NaNs after coercion (top 20):")
        print(bad_cols.head(20).to_string())

    if X.dtype == object:
        raise TypeError("X is still dtype=object. Numeric coercion failed.")

    q = np.nanpercentile(X, [0, 1, 5, 50, 95, 99, 100], axis=0)
    std = np.nanstd(X, axis=0)
    mean = np.nanmean(X, axis=0)

    out = pd.DataFrame(
        {
            "feature": feature_cols,
            "mean": mean,
            "std": std,
            "p0": q[0],
            "p1": q[1],
            "p5": q[2],
            "p50": q[3],
            "p95": q[4],
            "p99": q[5],
            "p100": q[6],
        }
    )
    out["p99_abs"] = np.abs(out["p99"])
    return out.sort_values(["std", "p99_abs"], ascending=False)


def correlation_redundancy(
    df: pd.DataFrame, feature_cols: list, outdir: str, sample_n: int = 20000
):
    """
    Saves a histogram of absolute correlations to diagnose redundancy.
    Computing full correlation on 73k x 786 can be heavy; sample rows.
    """
    ensure_dir(outdir)
    n = min(sample_n, len(df))

    samp = df.sample(n=n, random_state=0)
    X, _ = get_numeric_X(samp, feature_cols, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = StandardScaler().fit_transform(X)

    C = np.corrcoef(X, rowvar=False)
    # take upper triangle without diagonal
    iu = np.triu_indices(C.shape[0], k=1)
    vals = np.abs(C[iu])

    plt.figure(figsize=(8, 4))
    plt.hist(vals, bins=80)
    plt.title("Histogram of |feature-feature correlation| (sampled rows)")
    plt.xlabel("|corr|")
    plt.ylabel("count")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "corr_abs_hist.png"), dpi=200)
    plt.close()

    # Count highly correlated pairs
    for thr in [0.90, 0.95, 0.98, 0.99]:
        n_pairs = int(np.sum(vals > thr))
        print(f"Highly correlated pairs |corr| > {thr}: {n_pairs:,}")


def pca_plots(df: pd.DataFrame, feature_cols: list, outdir: str, sample_n: int = 30000):
    """
    PCA scatter: color by patient, then color by label.
    Only a sample to keep it readable.
    """
    ensure_dir(outdir)
    n = min(sample_n, len(df))

    samp = df.sample(n=n, random_state=0).copy()
    X, _ = get_numeric_X(samp, feature_cols, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = StandardScaler().fit_transform(X)

    Z = PCA(n_components=2, random_state=0).fit_transform(X)

    # Color by patient (categorical)
    plt.figure(figsize=(7, 6))
    for pid in sorted(samp["patient_short_id"].unique()):
        m = samp["patient_short_id"] == pid
        plt.scatter(Z[m, 0], Z[m, 1], s=6, alpha=0.35, label=pid)
    plt.title("PCA(2D) of features - colored by patient (sampled)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(alpha=0.2)
    plt.legend(fontsize=6, ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pca_by_patient.png"), dpi=200)
    plt.close()

    # Color by label (binary)
    plt.figure(figsize=(7, 6))
    for lab, name in [(0, "stable"), (1, "pre-hosp")]:
        m = samp["label"].to_numpy() == lab
        plt.scatter(Z[m, 0], Z[m, 1], s=6, alpha=0.35, label=name)
    plt.title("PCA(2D) of features - colored by label (sampled)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(alpha=0.2)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pca_by_label.png"), dpi=200)
    plt.close()


def within_patient_univariate_auc(
    df: pd.DataFrame, feature_cols: list, min_pos: int = 30, min_neg: int = 30
):
    """
    For each feature f:
      For each patient p with enough positives and negatives:
        compute AUC of f vs label within that patient
      aggregate across patients (median AUC, n_patients_used)

    This targets features whose directionality is consistent across people.
    """
    rows = []
    for f in feature_cols:
        aucs = []
        for pid, g in df.groupby("patient_short_id"):
            y = g["label"].to_numpy()
            n_pos = int(np.sum(y == 1))
            n_neg = int(np.sum(y == 0))
            if n_pos < min_pos or n_neg < min_neg:
                continue
            x = g[f].to_numpy()
            # if feature constant within patient, AUC undefined
            if np.nanstd(x) == 0:
                continue
            try:
                a = roc_auc_score(y, x)
                aucs.append(a)
            except Exception:
                pass
        if len(aucs) == 0:
            continue
        rows.append(
            {
                "feature": f,
                "median_auc_across_patients": float(np.median(aucs)),
                "mean_auc_across_patients": float(np.mean(aucs)),
                "n_patients_used": int(len(aucs)),
            }
        )
    out = pd.DataFrame(rows).sort_values(
        ["median_auc_across_patients", "n_patients_used"], ascending=False
    )
    return out


def patient_id_probe(df: pd.DataFrame, feature_cols: list, sample_n: int = 40000):
    """
    Quick-and-dirty check: can we predict patient ID from features?
    If yes (high balanced accuracy), patient identity is strongly encoded.
    """
    n = min(sample_n, len(df))

    samp = df.sample(n=n, random_state=0)
    X, _ = get_numeric_X(samp, feature_cols, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    y = samp["patient_short_id"].astype("category").cat.codes.to_numpy()

    X = StandardScaler().fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )

    clf = LogisticRegression(
        max_iter=200,
        n_jobs=-1,
        multi_class="multinomial",
        solver="saga",
    )
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    bal_acc = balanced_accuracy_score(yte, pred)
    return bal_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", type=str, default="data/dataset.parquet")
    ap.add_argument("--outdir", type=str, default="figures")
    ap.add_argument(
        "--run-patient-probe",
        action="store_true",
        help="runs patient-ID predictability probe",
    )
    ap.add_argument("--sample-n", type=int, default=30000)
    args = ap.parse_args()

    ensure_dir(args.outdir)

    df = pd.read_parquet(args.data_path)
    df["recording_date"] = extract_date_from_recording_id(df["recording_id"])

    metadata_cols = ["recording_id", "patient_short_id", "label", "recording_date"]
    candidate_cols = [c for c in df.columns if c not in metadata_cols]

    # Coerce everything once, keep columns that are "mostly numeric"
    X_df = (
        df[candidate_cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    nan_frac = X_df.isna().mean()

    feature_cols = nan_frac[
        nan_frac <= 0.0
    ].index.tolist()  # strict: keep only perfectly numeric

    dropped = nan_frac[nan_frac > 0].sort_values(ascending=False)
    print(f"Numeric features kept: {len(feature_cols)} / {len(candidate_cols)}")
    if len(dropped) > 0:
        print("\nDropped (non-numeric or mixed) columns (top 20):")
        print(dropped.head(20).to_string())

    # Overwrite df numeric block so later functions are safe
    df[feature_cols] = X_df[feature_cols].astype(np.float32)

    print(f"Using {len(feature_cols)} numeric feature columns.")
    if not dropped.empty:
        print(f"Dropped non-numeric columns: {dropped[:20]}")

    print("\n=== BASIC INTEGRITY ===")
    integrity = basic_integrity(df, feature_cols)
    for k, v in integrity.items():
        print(f"{k}: {v}")

    print("\n=== PATIENT METADATA CONSISTENCY (age/sex) ===")
    meta_cons = patient_metadata_consistency(df)
    if len(meta_cons) == 0:
        print("No age/sex columns found.")
    else:
        print(meta_cons.to_string(index=False))
        meta_cons.to_csv("eda_age_sex_per_patient.csv", index=False)

    print("\n=== TEMPORAL DENSITY ===")
    rpd = recordings_per_day(df)
    print(rpd.groupby("patient_short_id")["n"].describe().to_string())

    plt.figure(figsize=(10, 4))
    plt.hist(rpd["n"], bins=80)
    plt.title("Recordings per patient-day (all patients pooled)")
    plt.xlabel("n recordings per day")
    plt.ylabel("count of patient-days")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "recordings_per_day_hist.png"), dpi=200)
    plt.close()

    print("\n=== LABEL EPISODES ===")
    episodes = label_episodes(df)
    print(episodes.head(10).to_string(index=False))
    episodes.to_csv("eda_label_episodes.csv", index=False)

    # Plot rolling positive rates per patient
    rolling_positive_rate_plot(df, outdir=args.outdir, window_days=14)

    print("\n=== FEATURE SCALE / OUTLIERS REPORT ===")
    scale = feature_scale_report(df, feature_cols)
    scale.head(50).to_csv("eda_top50_by_scale.csv", index=False)
    print("Saved: eda_top50_by_scale.csv (features with largest std / extreme p99)")

    print("\n=== CORRELATION / REDUNDANCY (SAMPLED) ===")
    correlation_redundancy(df, feature_cols, outdir=args.outdir, sample_n=args.sample_n)

    print("\n=== PCA PLOTS (SAMPLED) ===")
    pca_plots(df, feature_cols, outdir=args.outdir, sample_n=args.sample_n)
    print(f"Saved PCA plots to {args.outdir}/")

    print("\n=== WITHIN-PATIENT UNIVARIATE FEATURE AUC ===")
    univ = within_patient_univariate_auc(df, feature_cols, min_pos=30, min_neg=30)
    univ.head(50).to_csv("top_features_by_within_patient_auc.csv", index=False)
    print("Saved: top_features_by_within_patient_auc.csv (top 50 shown)")
    print(univ.head(10).to_string(index=False))

    if args.run_patient_probe:
        print("\n=== PATIENT ID PREDICTABILITY PROBE ===")
        bal_acc = patient_id_probe(df, feature_cols, sample_n=min(40000, len(df)))
        print(f"Balanced accuracy predicting patient ID from features: {bal_acc:.3f}")
        print("If this is high, patient identity is strongly encoded in features.")

    # Save a compact summary CSV
    summary = pd.DataFrame([integrity])
    summary.to_csv("eda_summary.csv", index=False)
    print("\nSaved: eda_summary.csv")
    print("\nDONE.")


if __name__ == "__main__":
    main()

# RUN IT WITH:
# python src/explore_advanced.py --data-path data/dataset.parquet --outdir figures
# # optionally:
# python src/explore_advanced.py --run-patient-probe
