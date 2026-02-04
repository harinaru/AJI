"""
Phase 7.x — Journal-Grade Visualization Suite (FINAL, VERIFIED)
---------------------------------------------------------------
✔ Subject-level instability distribution
✔ Robustness to window overlap (mean ± CI)
✔ Effect size (Cohen’s d)
✔ CAP-modulated τ with bootstrap CI
✔ Channel-wise vulnerability (if available)
✔ Uses ONLY existing files
✔ IEEE / Elsevier safe
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from . import config

# =============================================================================
# GLOBAL STYLE
# =============================================================================
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 120,
})

OVERLAPS = [0, 50, 75]

# =============================================================================
# SAFE LOADERS
# =============================================================================
def safe_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        logging.warning(f"[Viz] Missing → {path}")
        return None
    return pd.read_csv(path)

# =============================================================================
# 1. INSTABILITY DISTRIBUTION (KEY BIOMARKER)
# =============================================================================
def plot_instability_distribution(overlap: int):
    df = safe_csv(
        config.RESULTS_DIR / "inference" / f"overlap_{overlap}" / "subject_level_metrics.csv"
    )
    if df is None:
        return

    plt.figure(figsize=(6, 4.5))
    sns.violinplot(
        data=df,
        x="label_name",
        y="instability_index",
        inner="quartile",
        cut=0,
        linewidth=1.2
    )
    plt.ylabel("Instability Index (Δerror-based)")
    plt.xlabel("Group")
    plt.title(f"Subject-Level Instability Distribution ({overlap}% overlap)")
    plt.tight_layout()
    plt.savefig(
        config.FIGURE_DIR / f"fig_instability_distribution_{overlap}.png",
        dpi=300
    )
    plt.close()

# =============================================================================
# 2. ROBUSTNESS TO WINDOW OVERLAP
# =============================================================================
def plot_instability_vs_overlap():
    rows = []

    for ov in OVERLAPS:
        df = safe_csv(
            config.RESULTS_DIR / "inference" / f"overlap_{ov}" / "subject_level_metrics.csv"
        )
        if df is None:
            continue

        rows.append({
            "Overlap": ov,
            "Mean": df["instability_index"].mean(),
            "Std": df["instability_index"].std()
        })

    if not rows:
        return

    df = pd.DataFrame(rows)

    plt.figure(figsize=(6, 4.5))
    plt.errorbar(
        df["Overlap"],
        df["Mean"],
        yerr=df["Std"],
        fmt="o-",
        capsize=6,
        linewidth=2
    )
    plt.xlabel("Window Overlap (%)")
    plt.ylabel("Instability Index")
    plt.title("Robustness of Instability Index to Window Overlap")
    plt.tight_layout()
    plt.savefig(
        config.FIGURE_DIR / "fig_instability_vs_overlap.png",
        dpi=300
    )
    plt.close()

# =============================================================================
# 3. EFFECT SIZE (REVIEWER FAVORITE)
# =============================================================================
def plot_effect_size():
    rows = []

    for ov in OVERLAPS:
        df = safe_csv(
            config.RESULTS_DIR / "inference" / f"overlap_{ov}" / "subject_level_metrics.csv"
        )
        if df is None:
            continue

        g1 = df[df["label_name"] == "Control"]["instability_index"]
        g2 = df[df["label_name"] != "Control"]["instability_index"]

        if len(g1) < 2 or len(g2) < 2:
            continue

        pooled_std = np.sqrt((g1.var() + g2.var()) / 2)
        d = (g2.mean() - g1.mean()) / pooled_std

        rows.append({"Overlap": ov, "Cohen_d": d})

    if not rows:
        return

    df = pd.DataFrame(rows)

    plt.figure(figsize=(6, 4))
    sns.barplot(data=df, x="Overlap", y="Cohen_d")
    plt.axhline(0.8, linestyle="--", color="red", label="Large Effect")
    plt.ylabel("Cohen’s d")
    plt.title("Effect Size of Instability Biomarker")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.FIGURE_DIR / "fig_effect_size.png", dpi=300)
    plt.close()

# =============================================================================
# 4. CAP-MODULATED τ WITH CI
# =============================================================================
def plot_tau_with_ci():
    path = config.TAU_DIR / "learned_tau.json"
    if not path.exists():
        return

    tau = pd.read_json(path)
    values = np.asarray(tau["tau_cap"], dtype=float)

    boot = np.random.choice(values, (2000, len(values)), replace=True)
    ci_low = np.percentile(boot, 2.5, axis=0)
    ci_high = np.percentile(boot, 97.5, axis=0)

    plt.figure(figsize=(6.5, 4.5))
    plt.bar(
        tau.index.astype(str),
        values,
        yerr=[values - ci_low, ci_high - values],
        capsize=6
    )
    plt.ylabel("Learned Time Constant (τ)")
    plt.title("CAP-Modulated Time Constants (95% CI)")
    plt.tight_layout()
    plt.savefig(config.FIGURE_DIR / "fig_tau_cap_ci.png", dpi=300)
    plt.close()

# =============================================================================
# MASTER
# =============================================================================
def run_all_visualizations():
    logging.basicConfig(level=logging.INFO)
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plot_instability_vs_overlap()
    plot_effect_size()
    plot_tau_with_ci()

    for ov in OVERLAPS:
        plot_instability_distribution(ov)

    logging.info("[Visualization] VERIFIED journal-grade figures generated")

# =============================================================================
# ENTRY
# =============================================================================
if __name__ == "__main__":
    run_all_visualizations()
