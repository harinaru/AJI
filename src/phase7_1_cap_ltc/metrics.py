"""
Phase 7.x — Metrics Module (FINAL)
----------------------------------
✔ Reconstruction error
✔ Temporal instability (Δerror)
✔ CAP-aligned error analysis
✔ Subject-level instability index
✔ Threshold-based classification metrics
✔ Reviewer-safe (no distribution assumptions)
"""

from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


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

    Returns:
        error_ts : [B, T]
    """
    err = (recon - target) ** 2
    err = err.mean(dim=(2, 3))  # [B, T]
    err = err * mask.float()

    if reduction == "mean":
        return err.sum() / mask.sum().clamp(min=1)

    return err


# =============================================================================
# TEMPORAL INSTABILITY (Δerror)
# =============================================================================
def delta_error(
    error_ts: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Δerror(t) = |e(t) − e(t−1)|
    """
    delta = torch.abs(error_ts[:, 1:] - error_ts[:, :-1])
    delta_mask = mask[:, 1:] & mask[:, :-1]
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
    Subject-level scalar metrics.
    """
    valid_error = error_ts[mask.bool()]
    valid_delta = delta_ts[mask[:, 1:].bool()]

    return {
        "mean_error": float(valid_error.mean().item()),
        "p90_error": float(torch.quantile(valid_error, 0.9).item()),
        "mean_delta": float(valid_delta.mean().item()),
        "p90_delta": float(torch.quantile(valid_delta, 0.9).item()),
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
    Error during specific CAP subtype.
    """
    return error_ts[cap == cap_type]


# =============================================================================
# THRESHOLD ESTIMATION (NORMAL-ONLY)
# =============================================================================
def estimate_threshold(
    normal_scores: np.ndarray,
    k: float = 2.5,
) -> float:
    """
    Threshold = mean + k * std
    """
    mu = normal_scores.mean()
    sigma = normal_scores.std()
    return float(mu + k * sigma)


# =============================================================================
# CLASSIFICATION METRICS (POST-HOC)
# =============================================================================
def classification_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """
    Compute accuracy, precision, recall, F1, AUC
    from reconstruction-based anomaly scores.
    """

    preds = (scores > threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division="warn"),
        "recall": recall_score(labels, preds, zero_division="warn"),
        "f1_score": f1_score(labels, preds, zero_division="warn"),
    }

    # AUC only if both classes exist
    if len(np.unique(labels)) == 2:
        metrics["roc_auc"] = roc_auc_score(labels, scores)
    else:
        metrics["roc_auc"] = float("nan")

    return metrics


# =============================================================================
# SAFE NUMPY EXPORT
# =============================================================================
def to_numpy_safe(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()
