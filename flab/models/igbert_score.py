import pandas as pd
import os
import argparse
import numpy as np

from igbert_model import igbert_paired_score

MODEL_NAME = "igbert"


def safe_score(heavy_seq, light_seq=None):
    if pd.isna(heavy_seq) or str(heavy_seq).strip() == "":
        return np.nan
    if light_seq is not None and (pd.isna(light_seq) or str(light_seq).strip() == ""):
        light_seq = None
    try:
        return igbert_paired_score(str(heavy_seq).strip(), light_seq)
    except Exception:
        return np.nan


def score_dataframe(df, heavy_col, light_col):
    def score_row(row):
        heavy_seq = row.get(heavy_col, np.nan)
        light_seq = row.get(light_col, np.nan)

        # Normalize empty/NaN light chain to None
        if pd.isna(light_seq) or str(light_seq).strip() == "":
            light_seq = None

        # Paired heavy + light
        if light_seq is not None:
            paired_score = safe_score(heavy_seq, light_seq)
        # Heavy-only (nanobody / VHH)
        else:
            paired_score = safe_score(heavy_seq, None)

        return pd.Series({f"{MODEL_NAME}_paired": paired_score})

    scores = df.apply(score_row, axis=1)
    return pd.concat([df, scores], axis=1)


def process_file(file_path, heavy_col="heavy", light_col="light", output_folder=None):
    if output_folder is None:
        output_folder = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"../score/{MODEL_NAME}/",
        )

    df = pd.read_csv(file_path)
    df = score_dataframe(df, heavy_col, light_col)

    os.makedirs(output_folder, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(output_folder, f"{base_name}_{MODEL_NAME}.csv")

    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score antibody sequences with IgBERT paired pseudo-perplexity.")
    parser.add_argument("file_path", type=str, help="Path to input CSV file.")
    parser.add_argument("--heavy-col", type=str, default="heavy", help="Column name for heavy chain sequence (default: heavy).")
    parser.add_argument("--light-col", type=str, default="light", help="Column name for light chain sequence (default: light).")
    parser.add_argument("--output-folder", type=str, default=None,
                        help="Output directory (default: ../score/igbert/ relative to this script).")
    args = parser.parse_args()

    process_file(args.file_path, args.heavy_col, args.light_col, args.output_folder)
