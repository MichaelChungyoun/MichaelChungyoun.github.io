#!/usr/bin/env python3
"""
Combine per-model scored CSVs for each dataset into a single wide-format CSV.

Usage (run from anywhere):
    python score/combine_csv.py

Output: one <dataset>_combined.csv per dataset in flab-website/score/.

Strategy:
  - For each dataset, start from the original data CSV (ground-truth labels).
  - Detect the unique key columns (string/object dtype: sequences, IDs).
  - Left-join each model's score column onto the original data via those keys.
  - This avoids float-precision issues that arise from merging on label columns.
"""

import os
import glob
import pandas as pd

SCORE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCORE_DIR, "../data")


def get_datasets():
    """Return {dataset_stem: full_path} for all CSVs in the data directory."""
    return {
        os.path.splitext(os.path.basename(p))[0]: p
        for p in glob.glob(os.path.join(DATA_DIR, "*.csv"))
    }


def classify_file(filename, dataset_names):
    """
    Return the dataset name a scored file belongs to, or None.
    Matches by longest prefix to handle names that share a prefix.
    """
    base = os.path.splitext(filename)[0]
    for name in sorted(dataset_names, key=len, reverse=True):
        if base.startswith(name + "_"):
            return name
    return None


def string_key_cols(df):
    """Return column names whose dtype is object (strings — sequences, IDs)."""
    return [c for c in df.columns if df[c].dtype == object]


def main():
    datasets = get_datasets()
    if not datasets:
        print(f"No datasets found in {DATA_DIR}")
        return

    # Find all scored CSVs one level down: score/<model>/<file>.csv
    csv_files = glob.glob(os.path.join(SCORE_DIR, "*", "*.csv"))

    # Group scored files by dataset
    groups = {}
    for path in csv_files:
        dataset = classify_file(os.path.basename(path), datasets)
        if dataset:
            groups.setdefault(dataset, []).append(path)

    if not groups:
        print("No scored files found. Run the model score scripts first.")
        return

    for dataset, scored_files in sorted(groups.items()):
        print(f"\nProcessing '{dataset}' ({len(scored_files)} model file(s))")

        # Load original data as the base
        base_df = pd.read_csv(datasets[dataset], low_memory=False)
        key_cols = string_key_cols(base_df)
        print(f"  Key columns: {key_cols}")

        combined = base_df.copy()

        for path in sorted(scored_files):
            print(f"  Adding {os.path.basename(path)}")
            scored = pd.read_csv(path, low_memory=False)

            # Score columns = columns not in the original data
            score_cols = [c for c in scored.columns if c not in base_df.columns]
            if not score_cols:
                print(f"    WARNING: no new score columns found, skipping.")
                continue

            if len(scored) != len(base_df):
                print(f"    WARNING: row count mismatch ({len(scored)} vs {len(base_df)}), skipping.")
                continue

            # Join by position to avoid cartesian products from duplicate sequences
            for col in score_cols:
                combined[col] = scored[col].values

        output_path = os.path.join(SCORE_DIR, f"{dataset}_combined.csv")
        combined.to_csv(output_path, index=False)
        print(f"  Saved → {output_path}  ({len(combined)} rows × {len(combined.columns)} cols)")


if __name__ == "__main__":
    main()
