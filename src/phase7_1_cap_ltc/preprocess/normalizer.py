# src/phase7_1_cap_ltc/preprocess/normalizer.py

"""
Normalization utilities for Phase 7
-----------------------------------
Channel-wise z-score normalization.
"""

import numpy as np


def normalize_features(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Normalize features per channel across time and feature dims.
    x: (T, C, F)
    """
    mean = x.mean(axis=(0, 2), keepdims=True)
    std = x.std(axis=(0, 2), keepdims=True)
    return (x - mean) / (std + eps)
