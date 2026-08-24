#!/bin/bash
# Score the GDPa1 dataset with all 11 models.
# Usage: bash run_gdpa1_scoring.sh
# Run from anywhere — paths are relative to this script's location.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
M="$SCRIPT_DIR/models"
GDPA1="$SCRIPT_DIR/data/GDPa1_v1.3_averaged.csv"
GDPA1_COLS="--heavy-col vh_protein_sequence --light-col vl_protein_sequence"

echo "=== Scoring GDPa1 with all models ==="
echo "Dataset: $GDPA1"
echo ""

# ESM2 models
echo "=== ESM2 8M ==="
conda run -n esm2 python "$M/esm2_8M_score.py" "$GDPA1" $GDPA1_COLS

echo "=== ESM2 35M ==="
conda run -n esm2 python "$M/esm2_35M_score.py" "$GDPA1" $GDPA1_COLS

echo "=== ESM2 150M ==="
conda run -n esm2 python "$M/esm2_150M_score.py" "$GDPA1" $GDPA1_COLS

echo "=== ESM2 650M ==="
conda run -n esm2 python "$M/esm2_650M_score.py" "$GDPA1" $GDPA1_COLS

echo "=== ESM2 3B ==="
conda run -n esm2 python "$M/esm2_3B_score.py" "$GDPA1" $GDPA1_COLS

# ESMC models
echo "=== ESMC 300M ==="
conda run -n esmc python "$M/esmc_300M_score.py" "$GDPA1" $GDPA1_COLS

echo "=== ESMC 600M ==="
conda run -n esmc python "$M/esmc_600M_score.py" "$GDPA1" $GDPA1_COLS

# AbLang2
echo "=== AbLang2 paired ==="
conda run -n ablang2_env python "$M/ablang2_score.py" "$GDPA1" $GDPA1_COLS

# AntiBERTy
echo "=== AntiBERTy ==="
conda run -n antiberty python "$M/antiberty_score.py" "$GDPA1" $GDPA1_COLS

# IgBERT
echo "=== IgBERT ==="
conda run -n igbert python "$M/igbert_score.py" "$GDPA1" $GDPA1_COLS

# IgLM
echo "=== IgLM ==="
conda run -n iglm python "$M/iglm_score.py" "$GDPA1" $GDPA1_COLS

# LD model
echo "=== LD ==="
conda run -n antiberty python "$M/ld_score.py" "$GDPA1" $GDPA1_COLS

echo ""
echo "=== All GDPa1 scoring done ==="
