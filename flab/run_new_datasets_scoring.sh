#!/bin/bash
# Score the 4 new datasets with all 11 models.
# Usage: bash run_new_datasets_scoring.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
M="$SCRIPT_DIR/models"
D="$SCRIPT_DIR/data"

NEW_DATASETS=(
    "$D/adams2017measuring_4420-fluorescein_exp_er.csv"
    "$D/koenig2017mutational_er_g6.csv"
    "$D/peterson2024integrated_ab_H1HA_kd.csv"
    "$D/shanehsazzadeh2023unlocking_zerokd_trastuzumab.csv"
)

score_all() {
    local env=$1
    local script=$2
    for f in "${NEW_DATASETS[@]}"; do
        echo "  $(basename $f)..."
        conda run -n "$env" python "$M/$script" "$f"
    done
}

echo "=== ESM2 8M ==="
score_all esm2 esm2_8M_score.py

echo "=== ESM2 35M ==="
score_all esm2 esm2_35M_score.py

echo "=== ESM2 150M ==="
score_all esm2 esm2_150M_score.py

echo "=== ESM2 650M ==="
score_all esm2 esm2_650M_score.py

echo "=== ESM2 3B ==="
score_all esm2 esm2_3B_score.py

echo "=== ESMC 300M ==="
score_all esmc esmc_300M_score.py

echo "=== ESMC 600M ==="
score_all esmc esmc_600M_score.py

echo "=== AbLang2 paired ==="
score_all ablang2_env ablang2_score.py

echo "=== AntiBERTy ==="
score_all antiberty antiberty_score.py

echo "=== IgBERT ==="
score_all igbert igbert_score.py

echo "=== IgLM ==="
score_all iglm iglm_score.py

echo "=== LD ==="
score_all antiberty ld_score.py

echo "=== Combining results ==="
conda run -n esm2 python "$SCRIPT_DIR/score/combine_csv.py"

echo "=== All done ==="
