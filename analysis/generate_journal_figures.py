# ============================================================
# generate_journal_figures.py
# Real-data figure generation from trained model outputs
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, mannwhitneyu
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA

# ------------------------------------------------------------
# PATH CONFIGURATION
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PHASE6_DIR = PROJECT_ROOT / "results" / "phase6_normality"
ABSTAIN_DIR = PROJECT_ROOT / "results" / "phase6D_abstention"
PHASE4_DIR = PROJECT_ROOT / "results" / "phase4_channel_datasets"

OUTPUT_DIR = PROJECT_ROOT / "analysis" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHANNELS = ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"]

print(f"Figures will be saved to: {OUTPUT_DIR}\n")

# ============================================================
# FIGURE 1 — TEMPORAL PANELS (REAL SUBJECT)
# ============================================================

def plot_temporal_subject(channel="Fp2-F4", subject_id=None):

    df = pd.read_csv(PHASE4_DIR / f"{channel}.csv")

    # Must already have epoch_uncertainty saved during Phase6C
    if "epoch_uncertainty" not in df.columns:
        raise ValueError("epoch_uncertainty missing. Re-run Phase6C.")

    if subject_id is None:
        subject_id = df["subject"].iloc[0]

    g = df[df["subject"] == subject_id].reset_index(drop=True)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(g["epoch_uncertainty"])
    axes[0].set_title("Epoch Uncertainty")

    axes[1].plot(g["cap_present"])
    axes[1].set_title("CAP Presence")

    axes[2].plot(g["global_energy"])
    axes[2].set_title("Global Energy")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig1_temporal_real.png", dpi=200)
    plt.close()

    print("✓ Fig1 generated")


# ============================================================
# FIGURE 2 — CHANNEL × CAP SENSITIVITY
# ============================================================

def plot_channel_cap_heatmap():

    matrix = []

    for ch in CHANNELS:
        df = pd.read_csv(PHASE6_DIR / f"cap_tail_uncertainty_{ch}.csv")

        # sensitivity defined as mean cap_ratio
        sens = df.groupby("label")["cap_ratio"].mean().values
        matrix.append(sens)

    matrix = np.array(matrix)

    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, aspect="auto")
    plt.colorbar(label="Mean CAP Ratio")
    plt.xticks([0,1], ["Control","Patient"])
    plt.yticks(range(len(CHANNELS)), CHANNELS)
    plt.title("Channel × CAP Sensitivity")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig2_channel_cap_real.png", dpi=200)
    plt.close()

    print("✓ Fig2 generated")


# ============================================================
# FIGURE 3 — ATTRIBUTION–ERROR CORRELATION
# ============================================================

def plot_error_attribution(channel="Fp2-F4"):

    df = pd.read_csv(PHASE4_DIR / f"{channel}.csv")

    if "epoch_uncertainty" not in df.columns:
        raise ValueError("Missing uncertainty column.")

    if "global_energy" not in df.columns:
        raise ValueError("Need attribution proxy.")

    x = df["epoch_uncertainty"]
    y = df["global_energy"]

    r, p = pearsonr(x, y)

    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, alpha=0.5)
    plt.title(f"Error–Energy Correlation\nr={r:.2f}, p={p:.2e}")
    plt.xlabel("Uncertainty")
    plt.ylabel("Energy")
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "fig3_error_attribution_real.png", dpi=200)
    plt.close()

    print("✓ Fig3 generated")


# ============================================================
# FIGURE 4 — τ DISTRIBUTION (CAP PRESENT VS ABSENT)
# ============================================================

def plot_tau_distribution(channel="Fp2-F4"):

    df = pd.read_csv(PHASE4_DIR / f"{channel}.csv")

    if "epoch_uncertainty" not in df.columns:
        raise ValueError("Need uncertainty column.")

    tau = df["epoch_uncertainty"]

    tau_cap = tau[df["cap_present"] == 1]
    tau_non = tau[df["cap_present"] == 0]

    stat, p = mannwhitneyu(tau_non, tau_cap)

    plt.figure(figsize=(6, 5))
    plt.boxplot([tau_non, tau_cap], 
    tick_labels=["No CAP","CAP"])
    plt.title(f"τ Distribution\np={p:.3e}")
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "fig4_tau_distribution_real.png", dpi=200)
    plt.close()

    print("✓ Fig4 generated")


# ============================================================
# FIGURE 5 — ROC CURVE (REAL)
# ============================================================

def plot_real_roc():

    df = pd.read_csv(ABSTAIN_DIR / "phase6D_subject_abstention.csv")

    y_true = df["label"]
    y_score = df["abstain"]

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0,1],[0,1],'--')
    plt.legend()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "fig5_roc_real.png", dpi=200)
    plt.close()

    print("✓ Fig5 generated")


# ============================================================
# FIGURE 6 — PCA LATENT SPACE
# ============================================================

def plot_pca_latent(channel="Fp2-F4"):

    df = pd.read_csv(PHASE4_DIR / f"{channel}.csv")

    feature_cols = [c for c in df.columns if channel in c]

    X = df[feature_cols].values
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)

    plt.figure(figsize=(6,5))
    plt.scatter(X2[:,0], X2[:,1], c=df["label"], alpha=0.4)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Latent PCA Projection")
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "fig6_pca_real.png", dpi=200)
    plt.close()

    print("✓ Fig6 generated")


# ============================================================
# RUN ALL
# ============================================================

if __name__ == "__main__":

    plot_temporal_subject()
    plot_channel_cap_heatmap()
    plot_error_attribution()
    plot_tau_distribution()
    plot_real_roc()
    plot_pca_latent()

    print("\nAll real-data figures generated successfully.")