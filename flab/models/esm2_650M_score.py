import pandas as pd
import os
import argparse
import numpy as np

from esm2_650M_model import esm2_650M_score

MODEL_NAME = "esm2_650M"


def safe_score(seq):
    if pd.isna(seq) or str(seq).strip() == "":
        return np.nan
    try:
        return esm2_650M_score(seq)
    except Exception:
        return np.nan


def score_dataframe(df, heavy_col, light_col):
    def score_row(row):
        heavy_score = safe_score(row.get(heavy_col, np.nan))
        light_score = safe_score(row.get(light_col, np.nan))
        return pd.Series({f"{MODEL_NAME}_unpaired": np.nanmean([heavy_score, light_score])})

    scores = df.apply(score_row, axis=1)
    return pd.concat([df, scores], axis=1)


def process_file(file_path, heavy_col="heavy", light_col="light"):
    df = pd.read_csv(file_path)
    df = score_dataframe(df, heavy_col, light_col)

    output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"../score/{MODEL_NAME}/")
    os.makedirs(output_folder, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(output_folder, f"{base_name}_{MODEL_NAME}.csv")

    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score antibody sequences with ESM2 650M pseudo-perplexity.")
    parser.add_argument("file_path", type=str, help="Path to input CSV file.")
    parser.add_argument("--heavy-col", type=str, default="heavy", help="Column name for heavy chain sequence (default: heavy).")
    parser.add_argument("--light-col", type=str, default="light", help="Column name for light chain sequence (default: light).")
    args = parser.parse_args()

    process_file(args.file_path, args.heavy_col, args.light_col)
