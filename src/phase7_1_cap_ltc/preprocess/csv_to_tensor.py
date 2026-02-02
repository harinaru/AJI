"""
Phase 7.0 — CSV → Tensor Utilities
---------------------------------
✔ Shared helpers for CSV feature parsing
✔ Handles variable channels per subject
✔ Zero-pads missing channels
✔ Used by prepare_from_phase2.py
"""

from typing import List
import numpy as np
import pandas as pd


# =============================================================================
# CHANNEL INFERENCE
# =============================================================================
def infer_channels_from_columns(columns: List[str]) -> List[int]:
    """
    Extract channel indices from column names like:
    ch1_mean, ch2_std, ...
    """
    channels = set()
    for c in columns:
        if c.startswith("ch") and "_" in c:
            try:
                ch = int(c.split("_")[0][2:])
                channels.add(ch)
            except ValueError:
                continue
    return sorted(channels)


# =============================================================================
# CSV → (T, C, F)
# =============================================================================
def csv_to_tensor(
    df: pd.DataFrame,
    channel_ids: List[int],
    feature_dim: int,
) -> np.ndarray:
    """
    Convert flat CSV → tensor (T, C, F)

    Missing channels are zero-padded.
    """
    T = len(df)
    C = len(channel_ids)

    features = np.zeros((T, C, feature_dim), dtype=np.float32)

    for ci, ch in enumerate(channel_ids):
        cols = [c for c in df.columns if c.startswith(f"ch{ch}_")]
        if not cols:
            continue

        vals = df[cols].to_numpy(dtype=np.float32)

        # Trim / pad to feature_dim
        if vals.shape[1] > feature_dim:
            vals = vals[:, :feature_dim]
        elif vals.shape[1] < feature_dim:
            pad = feature_dim - vals.shape[1]
            vals = np.pad(vals, ((0, 0), (0, pad)))

        features[:, ci, :] = vals

    return features


# =============================================================================
# NORMALIZATION
# =============================================================================
def normalize_features(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Channel-wise z-score normalization.
    """
    mean = x.mean(axis=(0, 2), keepdims=True)
    std = x.std(axis=(0, 2), keepdims=True)
    return (x - mean) / (std + eps)
