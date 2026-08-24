import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

df = pd.read_csv("flab/performance/GDPa1_v1.3_averaged_combined.csv")

assays = {
    "Purity":         "purity_%lc+hc",
    "PR CHO":         "polyreactivity_prscore_cho",
    "PR Ova":         "polyreactivity_prscore_ova",
    "AC-SINS pH 7.4": "acsins_dLmax_ph7.4",
    "AC-SINS pH 6.0": "acsins_dLmax_ph6.0",
    "HAC":            "hac_rt",
    "SMAC":           "smac_rt",
    "HIC":            "hic_rt",
    "Titer":          "normalized_titer_productionbatch1",
    "SEC % Monomer":  "sec_%monomer",
    "Tm1":            "tm1_nanodsf",
    "Tm2":            "tm2_nanodsf",
}

models = {
    "ESM2\n8M":    "esm2_8M_unpaired",
    "ESM2\n35M":   "esm2_35M_unpaired",
    "ESM2\n150M":  "esm2_150M_unpaired",
    "ESM2\n650M":  "esm2_650M_unpaired",
    "ESM2\n3B":    "esm2_3B_unpaired",
    "ESMc\n300M":  "esmc_300M_unpaired",
    "ESMc\n600M":  "esmc_600M_unpaired",
    "IgBERT":      "igbert_paired",
    "IgLM":        "iglm",
}

# Color groups
colors = (
    ["#4C72B0"] * 5   # ESM2 shades (same family)
    + ["#DD8452"] * 2  # ESMc
    + ["#55A868"]      # IgBERT
    + ["#C44E52"]      # IgLM
)

ncols = 4
nrows = 3
fig, axes = plt.subplots(nrows, ncols, figsize=(14, 9), sharey=False)
axes = axes.flatten()

model_labels = list(models.keys())
model_cols   = list(models.values())
x = np.arange(len(model_labels))
bar_width = 0.65

for i, (assay_label, assay_col) in enumerate(assays.items()):
    ax = axes[i]
    valid = df[[assay_col] + model_cols].dropna()
    corrs = []
    for col in model_cols:
        r, _ = stats.spearmanr(valid[assay_col], valid[col])
        corrs.append(r)

    bars = ax.bar(x, corrs, width=bar_width, color=colors, edgecolor="white", linewidth=0.4)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title(assay_label, fontsize=9, fontweight="bold", pad=4)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=6.5)
    ax.set_ylim(-1, 1)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylabel("Spearman ρ", fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Hide unused subplot (12 assays, 12 subplots — none to hide)

legend_patches = [
    mpatches.Patch(color="#4C72B0", label="ESM2"),
    mpatches.Patch(color="#DD8452", label="ESMc"),
    mpatches.Patch(color="#55A868", label="IgBERT"),
    mpatches.Patch(color="#C44E52", label="IgLM"),
]
fig.legend(handles=legend_patches, loc="lower right", fontsize=8, frameon=False,
           bbox_to_anchor=(0.98, 0.01), ncol=4)

fig.suptitle("GDPa1 — Spearman Correlation: Model Score vs. Developability Assay",
             fontsize=11, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("flab/performance/GDPa1_spearman.png", dpi=150, bbox_inches="tight")
print("Saved flab/performance/GDPa1_spearman.png")
