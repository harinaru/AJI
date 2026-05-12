# src/phase7_1_cap_ltc/analysis/multi_seed.py

"""
Phase 7.x — Multi-Seed Evaluation
=================================

Provides:
✔ Multi-seed reproducibility analysis
✔ Stability benchmarking
✔ Mean ± std metric aggregation
✔ Reviewer-grade reproducibility evidence
✔ Cross-seed robustness validation
✔ Statistical reporting support

Outputs
-------
results/
└── multi_seed/
    ├── seed_42_metrics.json
    ├── seed_49_metrics.json
    ├── seed_56_metrics.json
    ├── aggregated_metrics.json
    ├── multi_seed_summary.csv
    └── reproducibility_report.txt
"""

import os
import json
import random
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import torch

from . import config
from .train import run_training
from .infer import run_inference
from .evaluate import run_evaluation


# Prefer multi-seed list when available; fall back to single seed.
SEEDS = getattr(config, "SEEDS", [config.SEED])


# =============================================================================
# OUTPUT DIR
# =============================================================================
MULTI_SEED_DIR = (
    config.RESULTS_DIR
    / "multi_seed"
)

MULTI_SEED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# GLOBAL SEED SETTER
# =============================================================================
def set_seed(
    seed: int,
):
    """
    Fully deterministic setup.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    os.environ[
        "PYTHONHASHSEED"
    ] = str(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False

    logging.info(
        f"[MultiSeed] Seed set → {seed}"
    )


# =============================================================================
# SAFE JSON SAVE
# =============================================================================
def save_json(
    data: Dict,
    path: Path,
):

    with open(path, "w") as f:

        json.dump(
            data,
            f,
            indent=4,
        )


# =============================================================================
# LOAD METRICS
# =============================================================================
def load_eval_metrics() -> Dict:
    """
    Loads evaluation metrics generated
    by evaluate.py
    """

    metrics_path = (
        config.RESULTS_DIR
        / "evaluation"
        / f"overlap_{int(config.OVERLAP_PERCENT * 100)}"
        / "evaluation_metrics.json"
    )

    if not metrics_path.exists():

        raise FileNotFoundError(
            f"Missing metrics: {metrics_path}"
        )

    with open(metrics_path, "r") as f:

        metrics = json.load(f)

    return metrics


# =============================================================================
# SINGLE-SEED RUN
# =============================================================================
def run_single_seed(
    seed: int,
) -> Dict:

    logging.info(
        "=" * 70
    )

    logging.info(
        f"[MultiSeed] Running seed={seed}"
    )

    logging.info(
        "=" * 70
    )

    # -------------------------------------------------------------------------
    # Set seed
    # -------------------------------------------------------------------------
    set_seed(seed)

    # -------------------------------------------------------------------------
    # Update config
    # -------------------------------------------------------------------------
    config.SEED = seed

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------
    logging.info(
        "[MultiSeed] Training..."
    )

    run_training()

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------
    logging.info(
        "[MultiSeed] Inference..."
    )

    run_inference()

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------
    logging.info(
        "[MultiSeed] Evaluation..."
    )

    run_evaluation()

    # -------------------------------------------------------------------------
    # Load metrics
    # -------------------------------------------------------------------------
    metrics = load_eval_metrics()

    metrics["seed"] = seed

    # -------------------------------------------------------------------------
    # Save per-seed metrics
    # -------------------------------------------------------------------------
    save_json(
        metrics,
        MULTI_SEED_DIR
        / f"seed_{seed}_metrics.json",
    )

    logging.info(
        f"[MultiSeed] Seed {seed} complete"
    )

    return metrics


# =============================================================================
# AGGREGATE METRICS
# =============================================================================
def aggregate_metrics(
    results: List[Dict],
) -> Dict:

    metric_names = []

    # -------------------------------------------------------------------------
    # Extract numeric metrics
    # -------------------------------------------------------------------------
    for key, value in results[0].items():

        if isinstance(value, (int, float)):

            metric_names.append(key)

    aggregated = {}

    # -------------------------------------------------------------------------
    # Mean ± std
    # -------------------------------------------------------------------------
    for metric in metric_names:

        vals = []

        for r in results:

            val = r.get(metric)

            if isinstance(val, (int, float)):

                vals.append(float(val))

        if len(vals) == 0:
            continue

        aggregated[metric] = {
            "mean":
                float(np.mean(vals)),

            "std":
                float(np.std(vals)),

            "min":
                float(np.min(vals)),

            "max":
                float(np.max(vals)),
        }

    return aggregated


# =============================================================================
# GENERATE SUMMARY CSV
# =============================================================================
def save_summary_csv(
    results: List[Dict],
):

    rows = []

    for r in results:

        row = {}

        for k, v in r.items():

            if isinstance(v, (int, float)):

                row[k] = v

        rows.append(row)

    df = pd.DataFrame(rows)

    df.to_csv(
        MULTI_SEED_DIR
        / "multi_seed_summary.csv",
        index=False,
    )

    logging.info(
        "[MultiSeed] Summary CSV saved"
    )


# =============================================================================
# TEXT REPORT
# =============================================================================
def generate_report(
    aggregated: Dict,
):

    report_path = (
        MULTI_SEED_DIR
        / "reproducibility_report.txt"
    )

    with open(report_path, "w") as f:

        f.write(
            "MULTI-SEED REPRODUCIBILITY REPORT\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
            "Seeds evaluated:\n"
        )

        for s in SEEDS:

            f.write(f"  - {s}\n")

        f.write("\n")

        for metric, stats in aggregated.items():

            mean = stats["mean"]
            std = stats["std"]

            f.write(
                f"{metric}: "
                f"{mean:.4f} ± {std:.4f}\n"
            )

        f.write("\n")

        f.write(
            "Interpretation:\n"
        )

        f.write(
            "Low standard deviation indicates "
            "stable reproducibility across "
            "random initialization seeds.\n"
        )

    logging.info(
        "[MultiSeed] Report generated"
    )


# =============================================================================
# MAIN MULTI-SEED PIPELINE
# =============================================================================
def run_multi_seed_evaluation():

    logging.basicConfig(
        level=logging.INFO
    )

    logging.info(
        "=" * 80
    )

    logging.info(
        "MULTI-SEED REPRODUCIBILITY ANALYSIS"
    )

    logging.info(
        "=" * 80
    )

    all_results = []

    # -------------------------------------------------------------------------
    # Run all seeds
    # -------------------------------------------------------------------------
    for seed in SEEDS:

        result = run_single_seed(
            seed
        )

        all_results.append(result)

    # -------------------------------------------------------------------------
    # Aggregate
    # -------------------------------------------------------------------------
    aggregated = aggregate_metrics(
        all_results
    )

    # -------------------------------------------------------------------------
    # Save aggregated JSON
    # -------------------------------------------------------------------------
    save_json(
        aggregated,
        MULTI_SEED_DIR
        / "aggregated_metrics.json",
    )

    # -------------------------------------------------------------------------
    # Save CSV
    # -------------------------------------------------------------------------
    save_summary_csv(
        all_results
    )

    # -------------------------------------------------------------------------
    # Text report
    # -------------------------------------------------------------------------
    generate_report(
        aggregated
    )

    logging.info(
        "✅ MULTI-SEED ANALYSIS COMPLETE"
    )


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":

    run_multi_seed_evaluation()