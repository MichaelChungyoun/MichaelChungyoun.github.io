#!/bin/bash
# Run all model scoring for flab-website datasets.
# Usage: bash run_scoring.sh
# Run from anywhere — paths are relative to this script's location.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
M="$SCRIPT_DIR/models"
D="$SCRIPT_DIR/data"

STD_DATASETS=(
    "$D/garbinski2023_tm1.csv"
    "$D/garbinski2023_kd.csv"
    "$D/jain2017biophysical_HEK.csv"
    "$D/jain2017biophysical_Tm.csv"
    "$D/marks2021humanization_immunogenicity.csv"
)
GDPA1="$D/GDPa1_v1.3_averaged.csv"
GDPA1_COLS="--heavy-col vh_protein_sequence --light-col vl_protein_sequence"

score_all() {
    local env=$1
    local script=$2
    for f in "${STD_DATASETS[@]}"; do
        echo "  $(basename $f)..."
        conda run -n "$env" python "$M/$script" "$f"
    done
    echo "  GDPa1_combined.csv..."
    conda run -n "$env" python "$M/$script" "$GDPA1" $GDPA1_COLS
}

# ESM2 models (skip 8M and 35M — already done)
echo "=== ESM2 150M ==="
conda run -n esm2 python "$M/esm2_150M_score.py" "$D/jain2017biophysical_Tm.csv"
conda run -n esm2 python "$M/esm2_150M_score.py" "$D/marks2021humanization_immunogenicity.csv"
conda run -n esm2 python "$M/esm2_150M_score.py" "$GDPA1" $GDPA1_COLS

echo "=== ESM2 650M ==="
score_all esm2 esm2_650M_score.py

echo "=== ESM2 3B ==="
score_all esm2 esm2_3B_score.py

# ESMC models
echo "=== ESMC 300M ==="
score_all esmc esmc_300M_score.py

echo "=== ESMC 600M ==="
score_all esmc esmc_600M_score.py

# LD model
echo "=== LD ==="
score_all antiberty ld_score.py

# Combine all results
echo "=== Combining results ==="
conda run -n esm2 python "$SCRIPT_DIR/score/combine_csv.py"

echo "=== All done ==="
