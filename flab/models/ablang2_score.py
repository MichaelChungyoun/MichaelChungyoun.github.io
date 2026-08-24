import numpy as np
import pandas as pd
import os
import argparse

import ablang2

MODEL_NAME = "ablang2_paired"
BATCH_SIZE = 10

# Load model once at module level
_ablang = None


def get_model(device="cpu"):
    global _ablang
    if _ablang is None:
        _ablang = ablang2.pretrained(model_to_use="ablang2-paired", random_init=False, ncpu=1, device=device)
    return _ablang


def compute_perplexity(seq_input: list, device: str = "cpu") -> float:
    """
    Compute pseudo-perplexity for a sequence input using AbLang2.

    Args:
        seq_input: list of [heavy, light] or [heavy, ""] for heavy-only
        device: torch device string

    Returns:
        Pseudo-perplexity (scalar float). Lower = more confident/natural.
    """
    ablang = get_model(device=device)
    pll = ablang(seq_input, mode="pseudo_log_likelihood")
    return float(np.exp(-pll[0]))


def score_row(row, heavy_col, light_col, device="cpu"):
    heavy_seq = row.get(heavy_col, "")
    light_seq = row.get(light_col, "")

    heavy = "" if pd.isna(heavy_seq) else str(heavy_seq).strip()
    light = "" if pd.isna(light_seq) else str(light_seq).strip()

    if not heavy and not light:
        return float("nan")

    try:
        # VHH: heavy only, no light chain
        if heavy and not light:
            return compute_perplexity([[heavy, ""]], device=device)

        # Paired: heavy + light
        return compute_perplexity([[heavy, light]], device=device)

    except Exception as e:
        print(f"Scoring error: {e}", flush=True)
        return float("nan")


def process_file(file_path: str, output_folder: str, heavy_col: str = "heavy", light_col: str = "light", device: str = "cpu"):
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"{base_name}_{MODEL_NAME}.csv")

    # Load / resume
    if os.path.exists(output_path):
        print(f"Resuming from existing output: {output_path}", flush=True)
        df = pd.read_csv(output_path)
    else:
        df = pd.read_csv(file_path)
        df[MODEL_NAME] = np.nan
        df["_attempted"] = False

    if "_attempted" not in df.columns:
        df["_attempted"] = False

    n_done = int(df["_attempted"].sum())
    total = len(df)
    print(f"Loaded {total} rows — {n_done} already scored, {total - n_done} remaining.", flush=True)

    # Score pending rows with checkpointing
    rows_done = 0
    for i in range(total):
        if df.at[i, "_attempted"]:
            continue

        print(f"[{i + 1}/{total}] scoring row {i}...", flush=True)
        df.at[i, MODEL_NAME] = score_row(df.iloc[i], heavy_col, light_col, device=device)
        df.at[i, "_attempted"] = True
        rows_done += 1

        if rows_done % BATCH_SIZE == 0:
            df.to_csv(output_path, index=False)
            print(f"  checkpoint saved ({rows_done} new rows done) → {output_path}", flush=True)

    # Final save
    df = df.drop(columns=["_attempted"])
    df.to_csv(output_path, index=False)
    print(f"\nDone. Output: {output_path}", flush=True)
    print(f"New column: {MODEL_NAME}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score antibody sequences with AbLang2 pseudo-perplexity.")
    parser.add_argument("file_path", type=str, help="Path to input CSV file.")
    parser.add_argument("--output-folder", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), f"../score/{MODEL_NAME}/"),
                        help="Folder to write scored CSV.")
    parser.add_argument("--heavy-col", type=str, default="heavy", help="Column name for heavy chain sequence (default: heavy).")
    parser.add_argument("--light-col", type=str, default="light", help="Column name for light chain sequence (default: light).")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device, e.g. 'cpu' or 'cuda'.")
    args = parser.parse_args()

    process_file(args.file_path, args.output_folder, args.heavy_col, args.light_col, device=args.device)
