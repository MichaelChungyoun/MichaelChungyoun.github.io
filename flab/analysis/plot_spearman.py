"""
Plot Spearman correlations between model scores and developability labels.

One subplot per dataset/property. Bars show each model's Spearman r
against the ground-truth label.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

SCORE_DIR = os.path.join(os.path.dirname(__file__), "..", "score")

# (display label, column name in combined CSVs)
MODEL_INFO = [
    ("ESM2\n8M",      "esm2_8M_unpaired"),
    ("ESM2\n35M",     "esm2_35M_unpaired"),
    ("ESM2\n150M",    "esm2_150M_unpaired"),
    ("ESM2\n650M",    "esm2_650M_unpaired"),
    ("ESM2\n3B",      "esm2_3B_unpaired"),
    ("ESMc\n300M",    "esmc_300M_unpaired"),
    ("ESMc\n600M",    "esmc_600M_unpaired"),
    ("AbLang2\npaired", "ablang2_paired"),
    ("AntiBERTy",     "antiberty"),
    ("IgBERT\npaired", "igbert_paired"),
    ("IgLM",          "iglm"),
]
MODELS     = [label for label, _ in MODEL_INFO]
MODEL_COLS = [col   for _, col   in MODEL_INFO]

# ── Non-GDPa1 datasets ──────────────────────────────────────────────────────
SIMPLE_DATASETS = [
    {
        "file": "garbinski2023_kd_combined.csv",
        "label_col": "-log (KD (M) )",
        "title": "Garbinski 2023\nKD (-log)",
    },
    {
        "file": "garbinski2023_tm1_combined.csv",
        "label_col": "Tm1 (nanoDSF)",
        "title": "Garbinski 2023\nTm1",
    },
    {
        "file": "jain2017biophysical_HEK_combined.csv",
        "label_col": "HEK Titer (mg/L)",
        "title": "Jain 2017\nHEK Titer",
    },
    {
        "file": "jain2017biophysical_Tm_combined.csv",
        "label_col": "Fab Tm by DSF (°C)",
        "title": "Jain 2017\nTm",
    },
    {
        "file": "marks2021humanization_immunogenicity_combined.csv",
        "label_col": "%ADA response",
        "title": "Marks 2021\n%ADA Response",
    },
]

# ── GDPa1 developability properties ─────────────────────────────────────────
GDPA1_PROPS = [
    ("normalized_titer_productionbatch1", "Norm. Titer\n(Batch 1)"),
    ("normalized_titer_productionbatch2", "Norm. Titer\n(Batch 2)"),
    ("titer_productionbatch1",            "Titer\n(Batch 1)"),
    ("titer_productionbatch2",            "Titer\n(Batch 2)"),
    ("purity_%lc+hc",                    "Purity\n(%LC+HC)"),
    ("tonset_nanodsf",                   "Tonset\n(nanoDSF)"),
    ("tm1_nanodsf",                      "Tm1\n(nanoDSF)"),
    ("tm2_nanodsf",                      "Tm2\n(nanoDSF)"),
    ("tm3_nanodsf",                      "Tm3\n(nanoDSF)"),
    ("sec_%monomer",                     "SEC\n(%Monomer)"),
    ("smac_rt",                          "SMAC RT"),
    ("hic_rt",                           "HIC RT"),
    ("hac_rt",                           "HAC RT"),
    ("acsins_dLmax_ph7.4",               "AC-SINS dLmax\n(pH 7.4)"),
    ("acsins_dLmax_ph6.0",               "AC-SINS dLmax\n(pH 6.0)"),
    ("polyreactivity_prscore_cho",       "Polyreactivity\nPRScore CHO"),
    ("polyreactivity_prscore_ova",       "Polyreactivity\nPRScore OVA"),
    ("dlskd_pbsph7.4",                   "DLS Kd\n(PBS pH 7.4)"),
    ("dlskd_hisargph6.0",                "DLS Kd\n(HisArg pH 6.0)"),
]


def spearman_corr(df, label_col, model_col):
    """Return Spearman r, dropping rows with NaN in either column."""
    sub = df[[label_col, model_col]].dropna()
    if len(sub) < 5:
        return np.nan
    r, _ = spearmanr(sub[label_col], sub[model_col])
    return r


def make_bar_axes(ax, corrs, title, model_labels):
    """Draw a grouped bar chart of Spearman correlations on ax."""
    x = np.arange(len(MODELS))
    colors = plt.cm.tab20(np.linspace(0, 0.95, len(MODELS)))

    bars = ax.bar(x, corrs, color=colors, edgecolor="white", linewidth=0.5, width=0.65)

    # Colour bars by sign
    for bar, r in zip(bars, corrs):
        if np.isnan(r):
            bar.set_alpha(0.2)
        elif r < 0:
            bar.set_color("#d62728")
            bar.set_alpha(0.75)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_ylim(-1, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=45, ha="right", fontsize=5.5)
    ax.set_title(title, fontsize=7, pad=3)
    ax.set_ylabel("Spearman r", fontsize=5.5)
    ax.tick_params(axis="y", labelsize=5.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate values
    for bar, r in zip(bars, corrs):
        if not np.isnan(r):
            yoff = 0.03 if r >= 0 else -0.05
            va = "bottom" if r >= 0 else "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                r + yoff,
                f"{r:.2f}",
                ha="center",
                va=va,
                fontsize=4.5,
                rotation=0,
            )


def main():
    # ── Load data ─────────────────────────────────────────────────────────
    gdpa1_path = os.path.join(SCORE_DIR, "GDPa1_v1.3_averaged_combined.csv")
    df_gdpa1 = pd.read_csv(gdpa1_path, low_memory=False)

    # ── Build subplot grid ─────────────────────────────────────────────────
    n_simple = len(SIMPLE_DATASETS)
    n_gdpa1 = len(GDPA1_PROPS)
    n_total = n_simple + n_gdpa1  # 5 + 19 = 24

    ncols = 5
    nrows = int(np.ceil(n_total / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 3.5, nrows * 3.2),
        constrained_layout=True,
    )
    axes_flat = axes.flatten()

    model_labels = MODELS  # already formatted in MODEL_INFO

    ax_idx = 0

    # ── Simple datasets ────────────────────────────────────────────────────
    for ds in SIMPLE_DATASETS:
        path = os.path.join(SCORE_DIR, ds["file"])
        df = pd.read_csv(path, low_memory=False)
        corrs = [spearman_corr(df, ds["label_col"], col) for col in MODEL_COLS]
        make_bar_axes(axes_flat[ax_idx], corrs, ds["title"], model_labels)
        ax_idx += 1

    # ── GDPa1 properties ──────────────────────────────────────────────────
    for col, title in GDPA1_PROPS:
        corrs = [spearman_corr(df_gdpa1, col, mc) for mc in MODEL_COLS]
        make_bar_axes(axes_flat[ax_idx], corrs, f"GDPa1\n{title}", model_labels)
        ax_idx += 1

    # Hide unused axes
    for i in range(ax_idx, len(axes_flat)):
        axes_flat[i].set_visible(False)

    # ── Global title & legend ──────────────────────────────────────────────
    fig.suptitle(
        "Spearman Correlation: Model Scores vs Developability Labels",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    # Shared legend
    import matplotlib.patches as mpatches
    colors = plt.cm.tab20(np.linspace(0, 0.95, len(MODELS)))
    handles = [mpatches.Patch(color=c, label=m.replace("\n", " ")) for c, m in zip(colors, MODELS)]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(MODELS),
        fontsize=7,
        bbox_to_anchor=(0.5, -0.03),
        frameon=False,
    )

    out_path = os.path.join(os.path.dirname(__file__), "spearman_correlations.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
