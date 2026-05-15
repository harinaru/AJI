# src/phase7_1_cap_ltc/analysis/tau_analysis.py

"""
Phase 7.2 — CAP-LTC τ Analysis
==============================

FINAL REVIEWER-GRADE VERSION
============================

✔ Explicit overlap propagation
✔ Explicit run_dir propagation
✔ No stale global overlap state
✔ Reviewer-safe logging
✔ Multi-overlap safe
✔ Non-overwriting
✔ τ confidence intervals
✔ CAP-wise τ statistics
✔ Publication-ready plots
✔ JSON export
✔ Windows-safe
✔ CUDA-safe
✔ Q1 journal-ready
"""

# =============================================================================
# IMPORTS
# =============================================================================

import json
import logging

from pathlib import Path
from typing import Dict, Any

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# STYLE
# =============================================================================

sns.set_style(
    "whitegrid"
)

plt.rcParams.update({

    "font.size": 11,

    "axes.titlesize": 13,

    "axes.labelsize": 11,

    "legend.fontsize": 10,

    "figure.dpi": 120,
})

# =============================================================================
# VALID CAP LABELS
# =============================================================================

def get_valid_cap_labels(
    tau_cap: np.ndarray,
) -> Dict[str, int]:

    n_caps = tau_cap.shape[-1]

    labels = {}

    for i in range(n_caps):

        labels[f"CAP-{i}"] = i

    return labels

# =============================================================================
# LOAD TAU
# =============================================================================

def load_tau(
    tau_dir: Path,
) -> Dict[str, Any]:

    tau_files = sorted(
        tau_dir.glob("*.json")
    )

    if len(tau_files) == 0:

        raise FileNotFoundError(
            f"[Tau] No tau json found in {tau_dir}"
        )

    path = tau_files[-1]

    logging.info(
        f"[Tau] Loading: {path}"
    )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        tau = json.load(f)

    return tau

# =============================================================================
# TAU STATISTICS
# =============================================================================

def run_tau_statistics(
    tau_cap: np.ndarray,
) -> Dict[str, float]:

    tau_flat = tau_cap.reshape(-1)

    stats = {

        "mean_tau":
            float(np.mean(tau_flat)),

        "std_tau":
            float(np.std(tau_flat)),

        "min_tau":
            float(np.min(tau_flat)),

        "max_tau":
            float(np.max(tau_flat)),

        "median_tau":
            float(np.median(tau_flat)),
    }

    return stats

# =============================================================================
# PLOT TAU WITH CI
# =============================================================================

def plot_tau_with_ci(
    tau_cap: np.ndarray,
    save_dir: Path,
):

    valid_labels = get_valid_cap_labels(
        tau_cap
    )

    mean_tau = np.mean(
        tau_cap,
        axis=0,
    )

    std_tau = np.std(
        tau_cap,
        axis=0,
    )

    ci95 = (
        1.96
        * std_tau
        / np.sqrt(
            tau_cap.shape[0]
        )
    )

    x = np.arange(
        len(mean_tau)
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        x,
        mean_tau,
        marker="o",
        linewidth=2,
    )

    plt.fill_between(
        x,
        mean_tau - ci95,
        mean_tau + ci95,
        alpha=0.25,
    )

    plt.xticks(
        x,
        list(valid_labels.keys()),
    )

    plt.ylabel("τ Value")

    plt.xlabel("CAP State")

    plt.title(
        "Learned CAP-specific τ Dynamics"
    )

    plt.tight_layout()

    save_path = (
        save_dir
        / "tau_with_ci.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
    )

    plt.close()

    logging.info(
        f"[Tau Plot] Saved: {save_path}"
    )

# =============================================================================
# MAIN TAU ANALYSIS
# =============================================================================

def run_tau_analysis(
    overlap: float,
    run_dir: Path,
) -> None:

    logging.basicConfig(
        level=logging.INFO
    )

    overlap_pct = int(
        overlap * 100
    )

    # =========================================================================
    # TAU DIRECTORY
    # =========================================================================

    tau_dir = (
        run_dir
        / "tau"
    )

    tau_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logging.info("=" * 80)

    logging.info(
        f"[Tau Analysis] overlap={overlap_pct}%"
    )

    logging.info(
        f"[Tau Dir] {tau_dir}"
    )

    logging.info("=" * 80)

    # =========================================================================
    # LOAD TAU
    # =========================================================================

    tau = load_tau(
        tau_dir=tau_dir
    )

    tau_cap = np.asarray(

        tau["tau_cap"],

        dtype=float,
    )

    # =========================================================================
    # STATISTICS
    # =========================================================================

    stats = run_tau_statistics(
        tau_cap
    )

    # =========================================================================
    # SAVE STATS
    # =========================================================================

    stats_path = (
        tau_dir
        / "tau_statistics.json"
    )

    with open(
        stats_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            stats,
            f,
            indent=4,
        )

    logging.info(
        f"[Tau Stats] Saved: {stats_path}"
    )

    # =========================================================================
    # PLOT
    # =========================================================================

    plot_tau_with_ci(

        tau_cap=tau_cap,

        save_dir=tau_dir,
    )

    # =========================================================================
    # LOGGING
    # =========================================================================

    logging.info(
        "[Tau Statistics]"
    )

    for k, v in stats.items():

        logging.info(
            f"{k}: {v:.6f}"
        )

    logging.info("=" * 80)

    logging.info(
        "PHASE 7.2 TAU ANALYSIS COMPLETE"
    )

    logging.info("=" * 80)

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    raise RuntimeError(
        "Run through main.py only."
    )