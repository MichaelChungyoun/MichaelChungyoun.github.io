# FLAb Benchmark

Scoring pipeline and results for the FLAb (Fitness Landscape for Antibodies) benchmark datasets. Evaluates protein language models on their ability to predict antibody developability properties.

## Datasets

Six published antibody datasets in `data/`:

| File | Assay | Sequences |
|------|-------|-----------|
| `garbinski2023_tm1.csv` | Thermal stability (Tm1, nanoDSF) | 85 |
| `garbinski2023_kd.csv` | Binding affinity (KD) | 80 |
| `jain2017biophysical_Tm.csv` | Fab thermal stability (DSF) | 136 |
| `jain2017biophysical_HEK.csv` | HEK293 expression titer (mg/L) | 136 |
| `marks2021humanization_immunogenicity.csv` | Humanization immunogenicity (%ADA) | 216 |
| `GDPa1_combined.csv` | 50 clinical antibodies — titer, purity, Tm, SEC, HIC, polyreactivity, KD, and more | 1673 |

Standard datasets use `heavy`/`light` sequence columns. `GDPa1_combined` uses `vh_protein_sequence`/`vl_protein_sequence`.

## Models

Scoring scripts live in `models/`. Each model has a `*_model.py` (core scoring logic) and a `*_score.py` (CSV processor).

| Model | Score Column | Method |
|-------|-------------|--------|
| ESM2 8M | `esm2_8M_unpaired` | Pseudo-perplexity (masked LM) |
| ESM2 35M | `esm2_35M_unpaired` | Pseudo-perplexity (masked LM) |
| ESM2 150M | `esm2_150M_unpaired` | Pseudo-perplexity (masked LM) |
| ESM2 650M | `esm2_650M_unpaired` | Pseudo-perplexity (masked LM) |
| ESM2 3B | `esm2_3B_unpaired` | Pseudo-perplexity (masked LM) |
| ESM-C 300M | `esmc_300M_unpaired` | Pseudo-perplexity (masked LM) |
| ESM-C 600M | `esmc_600M_unpaired` | Pseudo-perplexity (masked LM) |
| AntiBERTy | `antiberty` | Pseudo-log-likelihood |
| IgLM | `iglm` | Perplexity with chain tokens |
| AbLang2 | `ablang2_paired` | Paired pseudo-perplexity |
| IgBERT | `igbert_paired` | Paired pseudo-perplexity |

Heavy and light chains are scored separately and averaged (`_unpaired`). Models that natively accept paired input (AbLang2, IgBERT) take the full heavy+light pair directly.

## Scored Outputs

`score/` contains:
- Per-model subdirectories (`score/esm2_8M/`, etc.) with one scored CSV per dataset
- Combined wide-format CSVs (`score/<dataset>_combined.csv`) with all model scores merged

## Running the Scoring Pipeline

**Prerequisites:** `esm2` conda env for ESM2 models, `esmc` conda env for ESM-C models.

Score a single dataset with one model:
```bash
cd models/
conda run -n esm2 python esm2_650M_score.py ../data/garbinski2023_tm1.csv
# For GDPa1 (different column names):
conda run -n esm2 python esm2_650M_score.py ../data/GDPa1_combined.csv \
    --heavy-col vh_protein_sequence --light-col vl_protein_sequence
```

Re-run all models across all datasets:
```bash
bash run_scoring.sh
```

Combine per-model outputs into per-dataset wide CSVs:
```bash
cd score/
conda run -n esm2 python combine_csv.py
```
