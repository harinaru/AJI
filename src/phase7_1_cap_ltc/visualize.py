"""
Phase 7.x — Multi-Overlap Visualization (FINAL)
-----------------------------------------------
✔ Loss curves
✔ Confusion matrices
✔ ROC curves
✔ Accuracy / Precision / Recall / F1 comparison
✔ Channel-wise heatmaps
✔ Normal vs Abnormal error distributions
✔ CAP-modulated error analysis
✔ τ dynamics visualization
✔ Supports overlap = {0, 50, 75}
✔ Publication-ready
"""

import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from . import config

# =============================================================================
# GLOBAL STYLE
# =============================================================================
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 120,
})

OVERLAPS = [0, 50, 75]


# =============================================================================
# UTILITIES
# =============================================================================
def safe_read(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        logging.warning(f"[Viz] Missing file → {path}")
        return None
    return pd.read_csv(path)


def load_scores_labels(overlap: int):
    path = config.RESULTS_DIR / "inference" / f"overlap_{overlap}" / "scores_labels.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=True)


# =============================================================================
# LOSS CURVES
# =============================================================================
def plot_loss_curves():
    plt.figure(figsize=(6, 4))

    for overlap in OVERLAPS:
        loss_path = config.LOG_DIR / f"loss_overlap_{overlap}.csv"
        if not loss_path.exists():
            continue

        df = pd.read_csv(loss_path)
        plt.plot(df["epoch"], df["loss"], label=f"{overlap}% overlap")

    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss")
    plt.title("Training Loss vs Epoch")
    plt.legend()

    out = config.FIGURE_DIR / "fig_loss_curves.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()


# =============================================================================
# CONFUSION MATRIX & ROC
# =============================================================================
def plot_confusion_and_roc(overlap: int):
    scores, labels = load_scores_labels(overlap)
    preds = (scores > scores.mean()).astype(int)

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Abnormal"],
                yticklabels=["Normal", "Abnormal"])
    plt.title(f"Confusion Matrix ({overlap}%)")
    plt.tight_layout()
    plt.savefig(config.FIGURE_DIR / f"cm_{overlap}.png", dpi=300)
    plt.close()

    # ROC
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC Curve ({overlap}%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.FIGURE_DIR / f"roc_{overlap}.png", dpi=300)
    plt.close()


# =============================================================================
# METRIC COMPARISON
# =============================================================================
def plot_metric_comparison():
    rows = []

    for overlap in OVERLAPS:
        scores, labels = load_scores_labels(overlap)
        preds = (scores > scores.mean()).astype(int)

        rows.append({
            "Overlap": overlap,
            "Accuracy": accuracy_score(labels, preds),
            "Precision": precision_score(labels, preds, zero_division="warn"),
            "Recall": recall_score(labels, preds, zero_division="warn"),
            "F1": f1_score(labels, preds, zero_division="warn"),
        })

    df = pd.DataFrame(rows).melt(id_vars="Overlap", var_name="Metric", value_name="Score")

    plt.figure(figsize=(7, 4))
    sns.barplot(data=df, x="Metric", y="Score", hue="Overlap")
    plt.ylim(0, 1)
    plt.title("Classification Performance vs Overlap")
    plt.tight_layout()
    plt.savefig(config.FIGURE_DIR / "metrics_vs_overlap.png", dpi=300)
    plt.close()


# =============================================================================
# NORMAL vs ABNORMAL ERROR DISTRIBUTION
# =============================================================================
def plot_normal_vs_abnormal(overlap: int):
    df = safe_read(config.RESULTS_DIR / "inference" / f"overlap_{overlap}" / "subject_level_metrics.csv")
    if df is None:
        return

    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x="label_name", y="p90_error",
                palette=["tab:green", "tab:red"])
    plt.title(f"Normal vs Abnormal Error ({overlap}%)")
    plt.tight_layout()
    plt.savefig(config.FIGURE_DIR / f"normal_vs_abnormal_{overlap}.png", dpi=300)
    plt.close()


# =============================================================================
# CHANNEL-WISE HEATMAP
# =============================================================================
def plot_channel_heatmap(overlap: int):
    df = safe_read(config.RESULTS_DIR / "inference" / f"overlap_{overlap}" / "channel_level_metrics.csv")
    if df is None:
        return

    pivot = df.pivot_table(index="channel", columns="label_name", values="p90_error", aggfunc="mean")

    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis")
    plt.title(f"Channel Vulnerability Heatmap ({overlap}%)")
    plt.tight_layout()
    plt.savefig(config.FIGURE_DIR / f"channel_heatmap_{overlap}.png", dpi=300)
    plt.close()


# =============================================================================
# τ CAP MODULATION
# =============================================================================
def plot_tau_modulation():
    tau_path = config.TAU_DIR / "learned_tau.json"
    if not tau_path.exists():
        return

    tau = pd.read_json(tau_path)
    plt.figure(figsize=(6, 4))
    plt.bar(["Non-CAP", "A1", "A2", "A3", "A?"], tau["tau_cap"])
    plt.ylabel("Learned τ")
    plt.title("CAP-Modulated Time Constants")
    plt.tight_layout()
    plt.savefig(config.FIGURE_DIR / "tau_cap_modulation.png", dpi=300)
    plt.close()


# =============================================================================
# MASTER RUNNER
# =============================================================================
def run_all_visualizations():
    logging.basicConfig(level=logging.INFO)

    plot_loss_curves()
    plot_metric_comparison()
    plot_tau_modulation()

    for overlap in OVERLAPS:
        plot_confusion_and_roc(overlap)
        plot_normal_vs_abnormal(overlap)
        plot_channel_heatmap(overlap)

    logging.info("[Visualization] All figures generated successfully")


if __name__ == "__main__":
    run_all_visualizations()
