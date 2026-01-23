"""
Export feature names to a .txt file.

- Loads the dataset parquet
- Excludes metadata columns
- (Optionally) keeps only numeric features
- Writes all feature names to a text file (one per line)
"""

import os
import argparse
import pandas as pd


DEFAULT_METADATA_COLS = [
    "recording_id",
    "patient_short_id",
    "label",
]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def get_feature_cols(df: pd.DataFrame, metadata_cols: list, numeric_only: bool) -> list:
    candidate_cols = [c for c in df.columns if c not in metadata_cols]

    if not numeric_only:
        return candidate_cols

    # Keep only numeric columns
    feature_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c])]
    return feature_cols


def prefix_breakdown(cols: list) -> list[tuple[str, int]]:
    """
    Split feature names at the first '_' and count prefixes.
    Example: 'mfcc_1' -> prefix 'mfcc'
    """
    counts = {}
    for c in cols:
        prefix = c.split("_")[0] if "_" in c else c
        counts[prefix] = counts.get(prefix, 0) + 1
    return sorted(counts.items(), key=lambda x: (-x[1], x[0]))


def export_feature_names(
    data_path: str,
    out_path: str,
    metadata_cols: list,
    numeric_only: bool,
    add_header: bool,
):
    df = pd.read_parquet(data_path)

    feature_cols = get_feature_cols(
        df, metadata_cols=metadata_cols, numeric_only=numeric_only
    )

    ensure_dir(os.path.dirname(out_path) or ".")

    lines = []
    if add_header:
        lines.append(f"# data_path: {data_path}")
        lines.append(f"# total_columns: {df.shape[1]}")
        lines.append(f"# metadata_cols_excluded: {metadata_cols}")
        lines.append(f"# numeric_only: {numeric_only}")
        lines.append(f"# n_features: {len(feature_cols)}")
        lines.append("# feature_prefix_breakdown:")
        for prefix, n in prefix_breakdown(feature_cols):
            lines.append(f"#   {prefix}: {n}")
        lines.append("# ---- feature names ----")

    lines.extend(feature_cols)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Saved {len(feature_cols)} feature names to: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", type=str, default="data/dataset.parquet")
    ap.add_argument("--out-path", type=str, default="feature_names.txt")
    ap.add_argument(
        "--numeric-only",
        action="store_true",
        help="If set, keep only numeric feature columns.",
    )
    ap.add_argument(
        "--add-header",
        action="store_true",
        help="If set, prepend a small summary header to the file.",
    )
    args = ap.parse_args()

    export_feature_names(
        data_path=args.data_path,
        out_path=args.out_path,
        metadata_cols=DEFAULT_METADATA_COLS,
        numeric_only=args.numeric_only,
        add_header=args.add_header,
    )


if __name__ == "__main__":
    main()
