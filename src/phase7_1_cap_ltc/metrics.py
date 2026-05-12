"""
Phase 7.x — FINAL Q1 JOURNAL-GRADE METRICS MODULE
=================================================
✔ Reconstruction error
✔ Temporal instability (Δerror)
✔ Subject-level instability biomarkers
✔ CAP-conditioned analysis
✔ Robust percentile thresholding
✔ Clinical metrics (Sensitivity/Specificity)
✔ AUROC + AUPRC
✔ Bootstrap confidence intervals
✔ Statistical significance testing
✔ Effect size computation
✔ Calibration support
✔ Reviewer-safe implementation
✔ Numerical stability safeguards
✔ Edge-case safe
✔ Pyright-safe
✔ NaN-safe
✔ MAD robust statistics
✔ Bootstrap AUPRC CI
✔ Calibration-safe
✔ Small-sample safe
"""

from typing import Dict, Tuple, TypeVar

import numpy as np
from numpy.typing import NDArray

import torch

from scipy.stats import (
    mannwhitneyu,
    kruskal,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    balanced_accuracy_score,
)

# =============================================================================
# CONSTANTS
# =============================================================================
EPS = 1e-8


# =============================================================================
# SAFE QUANTILE
# =============================================================================
def safe_quantile(
    x: torch.Tensor,
    q: float,
) -> torch.Tensor:
    """
    Safe quantile computation.
    """

    if x.numel() == 0:

        return torch.tensor(
            0.0,
            device=x.device,
        )

    return torch.quantile(x, q)


# =============================================================================
# MEDIAN ABSOLUTE DEVIATION
# =============================================================================
def median_absolute_deviation(
    x: torch.Tensor,
) -> float:
    """
    Robust dispersion estimator.

    MAD = median(|x - median(x)|)
    """

    if x.numel() == 0:
        return 0.0

    median = torch.median(x)

    mad = torch.median(
        torch.abs(x - median)
    )

    return float(mad.item())


# =============================================================================
# SAFE NUMPY CLEAN
# =============================================================================
_DType = TypeVar("_DType", bound=np.generic)


def clean_numpy(
    x: NDArray[_DType],
) -> NDArray[_DType]:
    """
    Remove NaN and Inf.
    """

    x = np.asarray(
        x,
    )

    return x[
        np.isfinite(x)
    ]


# =============================================================================
# BASIC RECONSTRUCTION ERROR
# =============================================================================
def reconstruction_error(
    recon: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Reconstruction error per timestep.

    Parameters
    ----------
    recon : [B,T,C,F]
    target : [B,T,C,F]
    mask : [B,T]

    Returns
    -------
    error_ts : [B,T]
    """

    err = (recon - target) ** 2

    err = err.mean(dim=(2, 3))

    err = err * mask.float()

    if reduction == "mean":

        return (
            err.sum()
            /
            mask.sum().clamp(min=1)
        )

    return err


# =============================================================================
# TEMPORAL INSTABILITY (Δerror)
# =============================================================================
def delta_error(
    error_ts: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Temporal instability:

        Δe(t) = |e(t) - e(t-1)|

    Returns
    -------
    delta_ts : [B,T-1]
    """

    delta = torch.abs(
        error_ts[:, 1:]
        -
        error_ts[:, :-1]
    )

    delta_mask = (
        mask[:, 1:].bool()
        &
        mask[:, :-1].bool()
    )

    return delta * delta_mask.float()


# =============================================================================
# SUBJECT-LEVEL INSTABILITY INDEX
# =============================================================================
def instability_index(
    error_ts: torch.Tensor,
    delta_ts: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, float]:
    """
    Subject-level scalar biomarkers.
    """

    valid_error = error_ts[
        mask.bool()
    ]

    valid_delta = delta_ts[
        mask[:, 1:].bool()
    ]

    # -------------------------------------------------------------------------
    # Safety guards
    # -------------------------------------------------------------------------
    if valid_error.numel() == 0:

        valid_error = torch.tensor(
            [0.0],
            device=error_ts.device,
        )

    if valid_delta.numel() == 0:

        valid_delta = torch.tensor(
            [0.0],
            device=delta_ts.device,
        )

    return {

        # ---------------------------------------------------------------------
        # Reconstruction biomarkers
        # ---------------------------------------------------------------------
        "mean_error":
            float(
                valid_error.mean().item()
            ),

        "median_error":
            float(
                torch.median(
                    valid_error
                ).item()
            ),

        "std_error":
            float(
                valid_error.std(
                    unbiased=False
                ).item()
            ),

        "mad_error":
            median_absolute_deviation(
                valid_error
            ),

        "p90_error":
            float(
                safe_quantile(
                    valid_error,
                    0.90,
                ).item()
            ),

        "p95_error":
            float(
                safe_quantile(
                    valid_error,
                    0.95,
                ).item()
            ),

        "max_error":
            float(
                valid_error.max().item()
            ),

        # ---------------------------------------------------------------------
        # Instability biomarkers
        # ---------------------------------------------------------------------
        "mean_delta":
            float(
                valid_delta.mean().item()
            ),

        "median_delta":
            float(
                torch.median(
                    valid_delta
                ).item()
            ),

        "std_delta":
            float(
                valid_delta.std(
                    unbiased=False
                ).item()
            ),

        "mad_delta":
            median_absolute_deviation(
                valid_delta
            ),

        "p90_delta":
            float(
                safe_quantile(
                    valid_delta,
                    0.90,
                ).item()
            ),

        "p95_delta":
            float(
                safe_quantile(
                    valid_delta,
                    0.95,
                ).item()
            ),

        "instability_index":
            float(
                valid_delta.std(
                    unbiased=False
                ).item()
            ),

        # ---------------------------------------------------------------------
        # Combined anomaly score
        # ---------------------------------------------------------------------
        "combined_score":
            float(
                (
                    valid_error.mean()
                    +
                    valid_delta.mean()
                ).item()
            ),
    }


# =============================================================================
# CAP-ALIGNED ERROR
# =============================================================================
def cap_aligned_error(
    error_ts: torch.Tensor,
    cap: torch.Tensor,
    cap_type: int,
) -> torch.Tensor:
    """
    Reconstruction error during specific CAP subtype.
    """

    return error_ts[
        cap == cap_type
    ]


# =============================================================================
# ROBUST THRESHOLD ESTIMATION
# =============================================================================
def estimate_threshold(
    normal_scores: NDArray[np.float64],
    method: str = "percentile",
    percentile: float = 95,
    k: float = 2.5,
) -> float:
    """
    Robust threshold estimation.
    """

    normal_scores = clean_numpy(
        normal_scores
    )

    if len(normal_scores) == 0:
        return float("nan")

    if method == "percentile":

        return float(
            np.percentile(
                normal_scores,
                percentile,
            )
        )

    elif method == "gaussian":

        mu = normal_scores.mean()

        sigma = normal_scores.std()

        return float(
            mu + k * sigma
        )

    raise ValueError(
        f"Unknown threshold method: {method}"
    )


# =============================================================================
# CLASSIFICATION METRICS
# =============================================================================
def classification_metrics(
    scores: NDArray[np.float64],
    labels: NDArray[np.int_],
    threshold: float,
) -> Dict[str, float]:
    """
    Clinical anomaly detection metrics.
    """

    scores = np.asarray(
        scores,
        dtype=float,
    )

    labels = np.asarray(
        labels,
        dtype=int,
    )

    # -------------------------------------------------------------------------
    # Remove NaN / Inf
    # -------------------------------------------------------------------------
    valid_mask = (
        np.isfinite(scores)
        &
        np.isfinite(labels)
    )

    scores = scores[
        valid_mask
    ]

    labels = labels[
        valid_mask
    ]

    # -------------------------------------------------------------------------
    # Safety
    # -------------------------------------------------------------------------
    if len(scores) == 0:

        return {
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "f1_score": float("nan"),
            "roc_auc": float("nan"),
            "auprc": float("nan"),
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
        }

    preds = (
        scores > threshold
    ).astype(int)

    # -------------------------------------------------------------------------
    # Confusion matrix
    # -------------------------------------------------------------------------
    cm = confusion_matrix(
        labels,
        preds,
        labels=[0, 1],
    )

    if cm.shape != (2, 2):

        tn, fp, fn, tp = 0, 0, 0, 0

    else:

        tn, fp, fn, tp = cm.ravel()

    # -------------------------------------------------------------------------
    # Clinical metrics
    # -------------------------------------------------------------------------
    specificity = (
        tn
        /
        (tn + fp + EPS)
    )

    sensitivity = (
        tp
        /
        (tp + fn + EPS)
    )

    metrics = {

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

        "sensitivity":
            float(sensitivity),

        "specificity":
            float(specificity),

        "f1_score":
            float(
                f1_score(
                    labels,
                    preds,
                    zero_division="warn",
                )
            ),

        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }

    # -------------------------------------------------------------------------
    # AUROC + AUPRC
    # -------------------------------------------------------------------------
    if len(np.unique(labels)) == 2:

        try:

            metrics["roc_auc"] = float(
                roc_auc_score(
                    labels,
                    scores,
                )
            )

        except Exception:

            metrics["roc_auc"] = float("nan")

        try:

            metrics["auprc"] = float(
                average_precision_score(
                    labels,
                    scores,
                )
            )

        except Exception:

            metrics["auprc"] = float("nan")

    else:

        metrics["roc_auc"] = float("nan")

        metrics["auprc"] = float("nan")

    return metrics


# =============================================================================
# BOOTSTRAP AUROC CI
# =============================================================================
def bootstrap_auc_ci(
    labels: NDArray[np.int_],
    scores: NDArray[np.float64],
    n_bootstrap: int = 2000,
    alpha: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Bootstrap confidence interval for AUROC.
    """

    labels = clean_numpy(np.asarray(labels, dtype=np.int_))
    scores = clean_numpy(scores)

    if len(labels) == 0:

        return (
            float("nan"),
            float("nan"),
        )

    rng = np.random.default_rng(seed)

    aucs = []

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

        try:

            aucs.append(
                roc_auc_score(y, s)
            )

        except Exception:
            continue

    if len(aucs) == 0:

        return (
            float("nan"),
            float("nan"),
        )

    lower = np.percentile(
        aucs,
        (1 - alpha) / 2 * 100,
    )

    upper = np.percentile(
        aucs,
        (1 + alpha) / 2 * 100,
    )

    return (
        float(lower),
        float(upper),
    )


# =============================================================================
# BOOTSTRAP AUPRC CI
# =============================================================================
def bootstrap_auprc_ci(
    labels: NDArray[np.int_],
    scores: NDArray[np.float64],
    n_bootstrap: int = 2000,
    alpha: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Bootstrap confidence interval for AUPRC.
    """

    labels = clean_numpy(labels)
    scores = clean_numpy(scores)

    if len(labels) == 0:

        return (
            float("nan"),
            float("nan"),
        )

    rng = np.random.default_rng(seed)

    values = []

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

        try:

            values.append(
                average_precision_score(
                    y,
                    s,
                )
            )

        except Exception:
            continue

    if len(values) == 0:

        return (
            float("nan"),
            float("nan"),
        )

    lower = np.percentile(
        values,
        (1 - alpha) / 2 * 100,
    )

    upper = np.percentile(
        values,
        (1 + alpha) / 2 * 100,
    )

    return (
        float(lower),
        float(upper),
    )


# =============================================================================
# MANN-WHITNEY SIGNIFICANCE
# =============================================================================
def mann_whitney_significance(
    control_scores: NDArray[np.float64],
    disorder_scores: NDArray[np.float64],
) -> Dict[str, float]:
    """
    Non-parametric significance testing.
    """

    control_scores = clean_numpy(
        control_scores
    )

    disorder_scores = clean_numpy(
        disorder_scores
    )

    if (
        len(control_scores) == 0
        or
        len(disorder_scores) == 0
    ):

        return {
            "u_statistic": float("nan"),
            "p_value": float("nan"),
        }

    stat, p = mannwhitneyu(
        control_scores,
        disorder_scores,
        alternative="two-sided",
    )

    return {
        "u_statistic": float(stat),
        "p_value": float(p),
    }


# =============================================================================
# KRUSKAL-WALLIS
# =============================================================================
def kruskal_cap_test(
    *groups,
) -> Dict[str, float]:
    """
    CAP-wise significance analysis.
    """

    valid_groups = [
        clean_numpy(
            np.asarray(g)
        )
        for g in groups
        if len(g) > 0
    ]

    if len(valid_groups) < 2:

        return {
            "h_statistic": float("nan"),
            "p_value": float("nan"),
        }

    stat, p = kruskal(
        *valid_groups
    )

    return {
        "h_statistic": float(stat),
        "p_value": float(p),
    }


# =============================================================================
# EFFECT SIZE (COHEN'S d)
# =============================================================================
def cohens_d(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> float:
    """
    Cohen's d effect size.
    """

    x = clean_numpy(x)

    y = clean_numpy(y)

    if len(x) < 2 or len(y) < 2:
        return 0.0

    nx = len(x)

    ny = len(y)

    pooled_std = np.sqrt(
        (
            ((nx - 1) * np.var(x, ddof=1))
            +
            ((ny - 1) * np.var(y, ddof=1))
        )
        /
        (nx + ny - 2 + EPS)
    )

    if pooled_std < EPS:
        return 0.0

    return float(
        (
            np.mean(x)
            -
            np.mean(y)
        )
        /
        pooled_std
    )


# =============================================================================
# BOOTSTRAP EFFECT SIZE CI
# =============================================================================
def bootstrap_effect_size_ci(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    n_bootstrap: int = 2000,
    alpha: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Bootstrap CI for Cohen's d.
    """

    x = clean_numpy(x)

    y = clean_numpy(y)

    if len(x) == 0 or len(y) == 0:

        return (
            float("nan"),
            float("nan"),
        )

    rng = np.random.default_rng(seed)

    ds = []

    for _ in range(n_bootstrap):

        xb = rng.choice(
            x,
            size=len(x),
            replace=True,
        )

        yb = rng.choice(
            y,
            size=len(y),
            replace=True,
        )

        ds.append(
            cohens_d(xb, yb)
        )

    lower = np.percentile(
        ds,
        (1 - alpha) / 2 * 100,
    )

    upper = np.percentile(
        ds,
        (1 + alpha) / 2 * 100,
    )

    return (
        float(lower),
        float(upper),
    )


# =============================================================================
# EXPECTED CALIBRATION ERROR
# =============================================================================
def expected_calibration_error(
    probs: NDArray[np.float64],
    labels: NDArray[np.int_],
    n_bins: int = 10,
) -> float:
    """
    Expected calibration error (ECE).
    """

    probs = clean_numpy(probs)

    labels = clean_numpy(labels)

    if len(probs) == 0:
        return float("nan")

    probs = np.clip(
        probs,
        0.0,
        1.0,
    )

    bins = np.linspace(
        0,
        1,
        n_bins + 1,
    )

    ece = 0.0

    for i in range(n_bins):

        mask = (
            (probs >= bins[i])
            &
            (probs < bins[i + 1])
        )

        if mask.sum() == 0:
            continue

        acc = labels[mask].mean()

        conf = probs[mask].mean()

        ece += (
            np.abs(acc - conf)
            *
            mask.mean()
        )

    return float(ece)


# =============================================================================
# SAFE NUMPY EXPORT
# =============================================================================
def to_numpy_safe(
    x: torch.Tensor,
) -> NDArray[np.float64]:
    """
    Safe tensor -> numpy conversion.
    """

    return (
        x.detach()
        .cpu()
        .numpy()
    )