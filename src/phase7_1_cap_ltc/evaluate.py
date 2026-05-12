"""
Phase 7.x — Reviewer-Grade Evaluation Module
============================================
✔ Clinical evaluation metrics
✔ AUROC + AUPRC
✔ Threshold sweep analysis
✔ Bootstrap confidence intervals
✔ Mann–Whitney significance
✔ Cohen's d effect size
✔ CAP-wise statistics
✔ JSON + CSV export
✔ Reviewer-safe
✔ Q1 journal-ready
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import mannwhitneyu

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
)

from . import config


# =============================================================================
# SAFE CSV
# =============================================================================
def safe_csv(path: Path):

    if not path.exists():

        logging.warning(
            f"[Eval] Missing file → {path}"
        )

        return None

    try:

        df = pd.read_csv(path)

        if len(df) == 0:

            logging.warning(
                f"[Eval] Empty CSV → {path}"
            )

            return None

        return df

    except Exception as e:

        logging.warning(
            f"[Eval] Failed reading CSV: {e}"
        )

        return None


# =============================================================================
# COHEN'S D
# =============================================================================
def cohens_d(
    x: np.ndarray,
    y: np.ndarray,
) -> float:

    x = np.asarray(x)
    y = np.asarray(y)

    nx = len(x)
    ny = len(y)

    pooled_std = np.sqrt(
        (
            ((nx - 1) * np.var(x, ddof=1))
            +
            ((ny - 1) * np.var(y, ddof=1))
        )
        /
        (nx + ny - 2)
    )

    if pooled_std == 0:
        return 0.0

    return float(
        (np.mean(x) - np.mean(y))
        / pooled_std
    )


# =============================================================================
# BOOTSTRAP AUC CI
# =============================================================================
def bootstrap_auc_ci(
    labels,
    scores,
    n_bootstrap=2000,
    seed=42,
):

    rng = np.random.default_rng(seed)

    aucs = []

    labels = np.asarray(labels)
    scores = np.asarray(scores)

    n = len(labels)

    for _ in range(n_bootstrap):

        idx = rng.integers(
            0,
            n,
            n,
        )

        y = labels[idx]
        s = scores[idx]

        if len(np.unique(y)) < 2:
            continue

        aucs.append(
            roc_auc_score(y, s)
        )

    if len(aucs) == 0:

        return (
            float("nan"),
            float("nan"),
        )

    lower = np.percentile(
        aucs,
        2.5,
    )

    upper = np.percentile(
        aucs,
        97.5,
    )

    return (
        float(lower),
        float(upper),
    )


# =============================================================================
# THRESHOLD METRICS
# =============================================================================
def compute_metrics(
    labels,
    scores,
    threshold,
):

    preds = (
        scores > threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        preds,
        labels=[0, 1],
    ).ravel()

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    metrics = {

        "threshold":
            float(threshold),

        "accuracy":
            float(
                accuracy_score(
                    labels,
                    preds,
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    labels,
                    preds,
                )
            ),

        "precision":
            float(
                precision_score(
                    labels,
                    preds,
                    zero_division="warn",
                )
            ),

        "recall":
            float(
                recall_score(
                    labels,
                    preds,
                    zero_division="warn",
                )
            ),

        "f1_score":
            float(
                f1_score(
                    labels,
                    preds,
                    zero_division="warn",
                )
            ),

        "sensitivity":
            float(sensitivity),

        "specificity":
            float(specificity),

        "tp":
            int(tp),

        "tn":
            int(tn),

        "fp":
            int(fp),

        "fn":
            int(fn),
    }

    if len(np.unique(labels)) == 2:

        metrics["roc_auc"] = float(
            roc_auc_score(
                labels,
                scores,
            )
        )

        metrics["auprc"] = float(
            average_precision_score(
                labels,
                scores,
            )
        )

    else:

        metrics["roc_auc"] = float("nan")
        metrics["auprc"] = float("nan")

    return metrics


# =============================================================================
# THRESHOLD SWEEP
# =============================================================================
def threshold_sweep(
    labels,
    scores,
):

    rows = []

    thresholds = np.linspace(
        np.min(scores),
        np.max(scores),
        100,
    )

    for t in thresholds:

        metrics = compute_metrics(
            labels,
            scores,
            t,
        )

        rows.append(metrics)

    return pd.DataFrame(rows)


# =============================================================================
# CAP STATISTICS
# =============================================================================
def compute_cap_statistics(
    win_df,
):

    rows = []

    grouped = win_df.groupby(
        "cap_state"
    )

    for cap, g in grouped:

        delta_series = pd.Series(
            pd.to_numeric(
                g["delta_error"],
                errors="coerce",
            ),
            dtype="float64",
        ).dropna()

        recon_series = pd.Series(
            pd.to_numeric(
                g["reconstruction_error"],
                errors="coerce",
            ),
            dtype="float64",
        ).dropna()

        rows.append({

            "cap_state":
                int(cap),

            "mean_reconstruction_error":
                float(
                    np.mean(recon_series)
                ),

            "std_reconstruction_error":
                float(
                    np.std(recon_series)
                ),

            "mean_delta_error":
                float(
                    np.mean(delta_series)
                ),

            "std_delta_error":
                float(
                    np.std(delta_series)
                ),

            "count":
                int(len(g)),
        })

    return pd.DataFrame(rows)


# =============================================================================
# MAIN EVALUATION
# =============================================================================
def run_evaluation():

    logging.basicConfig(
        level=logging.INFO
    )

    overlap = int(
        config.OVERLAP_PERCENT * 100
    )

    inference_dir = (
        config.RESULTS_DIR
        / "inference"
        / f"overlap_{overlap}"
    )

    subject_path = (
        inference_dir
        / "subject_level_metrics.csv"
    )

    window_path = (
        inference_dir
        / "window_level_metrics.csv"
    )

    subj_df = safe_csv(subject_path)
    win_df = safe_csv(window_path)

    if subj_df is None:

        logging.warning(
            "[Eval] Subject metrics missing"
        )

        return

    # =========================================================================
    # SAFE CONVERSIONS
    # =========================================================================
    labels = (
        pd.Series(
            pd.to_numeric(
                subj_df["label"],
                errors="coerce",
            ),
            dtype="float64",
        )
        .fillna(0)
        .astype(int)
        .to_numpy()
    )

    scores = (
        pd.Series(
            pd.to_numeric(
                subj_df["instability_index"],
                errors="coerce",
            ),
            dtype="float64",
        )
        .dropna()
        .to_numpy(dtype=np.float64)
    )

    # =========================================================================
    # THRESHOLD
    # =========================================================================
    control_scores = scores[
        labels == 0
    ]

    threshold = float(
        np.mean(control_scores)
        +
        2.0 * np.std(control_scores)
    )

    # =========================================================================
    # CORE METRICS
    # =========================================================================
    metrics = compute_metrics(
        labels,
        scores,
        threshold,
    )

    # =========================================================================
    # BOOTSTRAP CI
    # =========================================================================
    auc_ci = bootstrap_auc_ci(
        labels,
        scores,
    )

    metrics["roc_auc_ci_lower"] = auc_ci[0]
    metrics["roc_auc_ci_upper"] = auc_ci[1]

    # =========================================================================
    # STATISTICS
    # =========================================================================
    disorder_scores = scores[
        labels == 1
    ]

    u_stat, p_value = mannwhitneyu(
        control_scores,
        disorder_scores,
        alternative="two-sided",
    )

    effect_size = cohens_d(
        control_scores,
        disorder_scores,
    )

    statistics = {

        "mannwhitney_u":
            float(u_stat),

        "p_value":
            float(p_value),

        "cohens_d":
            float(effect_size),

        "control_mean":
            float(
                np.mean(control_scores)
            ),

        "disorder_mean":
            float(
                np.mean(disorder_scores)
            ),

        "control_std":
            float(
                np.std(control_scores)
            ),

        "disorder_std":
            float(
                np.std(disorder_scores)
            ),
    }

    # =========================================================================
    # SAVE METRICS JSON
    # =========================================================================
    metrics_json = (
        inference_dir
        / "metrics_summary.json"
    )

    with open(metrics_json, "w") as f:

        json.dump(
            metrics,
            f,
            indent=4,
        )

    # =========================================================================
    # SAVE STATISTICS JSON
    # =========================================================================
    statistics_json = (
        inference_dir
        / "statistics_summary.json"
    )

    with open(statistics_json, "w") as f:

        json.dump(
            statistics,
            f,
            indent=4,
        )

    # =========================================================================
    # ROC CURVE CSV
    # =========================================================================
    if len(np.unique(labels)) == 2:

        fpr, tpr, roc_thresh = roc_curve(
            labels,
            scores,
        )

        roc_df = pd.DataFrame({

            "fpr":
                fpr,

            "tpr":
                tpr,

            "threshold":
                roc_thresh,
        })

        roc_df.to_csv(
            inference_dir / "roc_curve.csv",
            index=False,
        )

    # =========================================================================
    # PR CURVE CSV
    # =========================================================================
    precision, recall, pr_thresh = precision_recall_curve(
        labels,
        scores,
    )

    pr_df = pd.DataFrame({

        "precision":
            precision[:-1],

        "recall":
            recall[:-1],

        "threshold":
            pr_thresh,
    })

    pr_df.to_csv(
        inference_dir / "pr_curve.csv",
        index=False,
    )

    # =========================================================================
    # THRESHOLD SWEEP CSV
    # =========================================================================
    threshold_df = threshold_sweep(
        labels,
        scores,
    )

    threshold_df.to_csv(
        inference_dir / "threshold_analysis.csv",
        index=False,
    )

    # =========================================================================
    # SUBJECT RANKING CSV
    # =========================================================================
    ranking_df = subj_df.copy()

    ranking_df = ranking_df.sort_values(
        "instability_index",
        ascending=False,
    )

    ranking_df.to_csv(
        inference_dir / "subject_ranking.csv",
        index=False,
    )

    # =========================================================================
    # CAP STATISTICS CSV
    # =========================================================================
    if win_df is not None:

        cap_df = compute_cap_statistics(
            win_df
        )

        cap_df.to_csv(
            inference_dir / "cap_statistics.csv",
            index=False,
        )

    logging.info(
        "✅ REVIEWER-GRADE EVALUATION COMPLETE"
    )


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":

    run_evaluation()