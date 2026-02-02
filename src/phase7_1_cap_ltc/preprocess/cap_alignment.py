"""
Phase 7.1 — CAP Alignment Module
--------------------------------
✔ Aligns epoch-level CAP labels to temporal windows
✔ Handles partial CAP windows robustly
✔ Reviewer-safe strategies (no hard assumptions)
✔ Supports multiple labeling policies
"""

from typing import Literal
import numpy as np
import logging


# =============================================================================
# CAP ALIGNMENT STRATEGIES
# =============================================================================
CAPStrategy = Literal[
    "majority",     # majority CAP subtype inside window
    "any",          # CAP present if any epoch has CAP
    "center",       # CAP at center epoch
    "ratio_weighted" # weighted by CAP duration
]


# =============================================================================
# CORE FUNCTION
# =============================================================================
def align_cap_to_windows(
    cap_epochs: np.ndarray,
    window_size: int,
    overlap: float,
    strategy: CAPStrategy = "majority",
) -> np.ndarray:
    """
    Align epoch-level CAP labels to windows.

    Args:
        cap_epochs : (T,) array of CAP subtype labels
                     0 = Non-CAP, 1 = A1, 2 = A2, 3 = A3
        window_size: number of epochs per window
        overlap    : overlap fraction (0.0 / 0.5 / 0.75)
        strategy   : CAP alignment strategy

    Returns:
        cap_windows : (N,) window-level CAP labels
    """

    assert cap_epochs.ndim == 1, "cap_epochs must be 1D"
    assert 0 <= overlap < 1, "Overlap must be in [0, 1)"

    step = max(int(window_size * (1 - overlap)), 1)
    T = len(cap_epochs)

    cap_windows = []

    for start in range(0, T - window_size + 1, step):
        end = start + window_size
        window = cap_epochs[start:end]

        if strategy == "majority":
            # Majority non-zero CAP subtype
            non_zero = window[window > 0]
            label = int(np.bincount(non_zero).argmax()) if len(non_zero) > 0 else 0

        elif strategy == "any":
            # Any CAP present → mark as CAP (use strongest subtype)
            label = int(window.max())

        elif strategy == "center":
            # CAP at center epoch
            label = int(window[len(window) // 2])

        elif strategy == "ratio_weighted":
            # Weighted by duration inside window
            counts = np.bincount(window, minlength=4)
            if counts[1:].sum() == 0:
                label = 0
            else:
                label = int(np.argmax(counts[1:]) + 1)

        else:
            raise ValueError(f"Unknown CAP strategy: {strategy}")

        cap_windows.append(label)

    cap_windows = np.asarray(cap_windows, dtype=np.int64)

    logging.info(
        f"[CAP Alignment] Strategy='{strategy}', "
        f"Windows={len(cap_windows)}, "
        f"CAP%={np.mean(cap_windows > 0) * 100:.2f}"
    )

    return cap_windows


# =============================================================================
# OPTIONAL DIAGNOSTICS
# =============================================================================
def cap_alignment_summary(cap_windows: np.ndarray) -> None:
    """
    Print distribution of CAP labels after alignment.
    """
    unique, counts = np.unique(cap_windows, return_counts=True)
    dist = dict(zip(unique, counts))

    print("\nCAP WINDOW DISTRIBUTION")
    print("-" * 40)
    for k in sorted(dist):
        name = "Non-CAP" if k == 0 else f"A{k}"
        print(f"{name:8s}: {dist[k]}")
    print("-" * 40)
