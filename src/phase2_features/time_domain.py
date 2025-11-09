# ===============================================================
# src/phase2_features/time_domain.py
# v4.3.3_stable — Research-Grade Time-Domain Feature Extractor
# ===============================================================
"""
EEG Time-Domain Feature Extractor (30+ features)
Compatible with Phase 2 Feature Manager (v4.3.3)
"""

import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks
import warnings
from typing import Dict, Any

# ---------------------------------------------------------------
# Optional Entropy Library (safe import)
# ---------------------------------------------------------------
try:
    import antropy as ant
    HAS_ANTROPY = True
except ImportError:
    ant = None
    HAS_ANTROPY = False
    warnings.warn("⚠️ AntroPy not installed — entropy features set to 0.0.")


# ---------------------------------------------------------------
# Hjorth Parameters
# ---------------------------------------------------------------
def hjorth_params(x: np.ndarray):
    """Compute Hjorth Activity, Mobility, Complexity safely."""
    if len(x) < 3 or np.allclose(x, x[0]):
        return 0.0, 0.0, 0.0
    dx = np.diff(x)
    ddx = np.diff(dx)
    var_x = np.var(x)
    var_dx = np.var(dx)
    var_ddx = np.var(ddx)
    if var_x == 0 or var_dx == 0:
        return var_x, 0.0, 0.0
    activity = var_x
    mobility = np.sqrt(var_dx / (var_x + 1e-12))
    complexity = np.sqrt(var_ddx / (var_dx + 1e-12)) / (mobility + 1e-12)
    return float(activity), float(mobility), float(complexity)


# ---------------------------------------------------------------
# Core Per-Channel Feature Extractor
# ---------------------------------------------------------------
def extract_time_features(x: np.ndarray, sfreq: float, prefix: str = "") -> Dict[str, Any]:
    """Compute 30+ time-domain features for one EEG channel."""
    feats = {}
    prefix = prefix.strip("_")
    x = np.asarray(x, dtype=np.float64)
    if x.size < 2:
        return {f"{prefix}_empty": 0.0}

    # --- Statistical ---
    feats[f"{prefix}_mean"] = float(np.mean(x))
    feats[f"{prefix}_std"] = float(np.std(x))
    feats[f"{prefix}_var"] = float(np.var(x))
    feats[f"{prefix}_min"] = float(np.min(x))
    feats[f"{prefix}_max"] = float(np.max(x))
    feats[f"{prefix}_iqr"] = float(np.percentile(x, 75) - np.percentile(x, 25))
    feats[f"{prefix}_mad"] = float(np.mean(np.abs(x - np.mean(x))))
    feats[f"{prefix}_mean_abs_val"] = float(np.mean(np.abs(x)))

    try:
        feats[f"{prefix}_skew"] = float(skew(x, bias=False))
        feats[f"{prefix}_kurt"] = float(kurtosis(x, bias=False))
    except Exception:
        feats[f"{prefix}_skew"] = 0.0
        feats[f"{prefix}_kurt"] = 0.0

    # --- Energy / Morphology ---
    feats[f"{prefix}_peak_to_peak"] = float(np.ptp(x))
    feats[f"{prefix}_energy"] = float(np.sum(x ** 2))
    feats[f"{prefix}_rms"] = float(np.sqrt(np.mean(x ** 2)))
    feats[f"{prefix}_line_length"] = float(np.sum(np.abs(np.diff(x))))
    feats[f"{prefix}_zero_cross"] = int(np.sum(x[:-1] * x[1:] < 0))

    # --- Hjorth ---
    a, m, c = hjorth_params(x)
    feats[f"{prefix}_hjorth_activity"] = a
    feats[f"{prefix}_hjorth_mobility"] = m
    feats[f"{prefix}_hjorth_complexity"] = c

    # --- Peaks / Troughs ---
    try:
        peaks, _ = find_peaks(x)
        troughs, _ = find_peaks(-x)
        feats[f"{prefix}_num_peaks"] = int(len(peaks))
        feats[f"{prefix}_num_troughs"] = int(len(troughs))
    except Exception:
        feats[f"{prefix}_num_peaks"] = 0
        feats[f"{prefix}_num_troughs"] = 0

    feats[f"{prefix}_curve_length"] = float(np.sum(np.abs(np.diff(x))))

    # --- Dynamic / Willison ---
    dx = np.diff(x)
    feats[f"{prefix}_slope_sign_changes"] = int(np.sum(dx[:-1] * dx[1:] < 0))
    feats[f"{prefix}_willison_amplitude"] = int(np.sum(np.abs(np.diff(x)) > 0.05))

    # --- Entropy & Complexity ---
    feats[f"{prefix}_approx_entropy"] = 0.0
    feats[f"{prefix}_sample_entropy"] = 0.0
    feats[f"{prefix}_perm_entropy"] = 0.0
    if HAS_ANTROPY and ant is not None and not np.allclose(np.std(x), 0):
        try:
            feats[f"{prefix}_approx_entropy"] = float(ant.app_entropy(x))
            feats[f"{prefix}_sample_entropy"] = float(ant.sample_entropy(x))
            feats[f"{prefix}_perm_entropy"] = float(ant.perm_entropy(x, normalize=True))
        except Exception:
            pass

    # --- Fractal Dimension ---
    try:
        diff = np.diff(x)
        N_delta = np.sum(diff[1:] * diff[:-1] < 0)
        n = len(x)
        feats[f"{prefix}_fractal_dimension"] = float(
            np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * (N_delta + 1e-12))))
        )
    except Exception:
        feats[f"{prefix}_fractal_dimension"] = 0.0

    # --- Hurst Exponent ---
    try:
        N = len(x)
        Y = np.cumsum(x - np.mean(x))
        R = np.max(Y) - np.min(Y)
        S = np.std(x)
        feats[f"{prefix}_hurst_exp"] = float(np.log(R / (S + 1e-12)) / np.log(N + 1e-12))
    except Exception:
        feats[f"{prefix}_hurst_exp"] = 0.0

    return feats


# ---------------------------------------------------------------
# Channel-wise Aggregation
# ---------------------------------------------------------------
def extract_all_channels(epoch: np.ndarray, sfreq: float, ch_names=None) -> Dict[str, Any]:
    """Extract time features for all channels in one epoch."""
    feats = {}
    if epoch.ndim == 1:
        epoch = epoch[np.newaxis, :]

    n_channels = epoch.shape[0]
    if not ch_names or len(ch_names) != n_channels:
        ch_names = [f"ch{i+1}" for i in range(n_channels)]

    for idx, ch in enumerate(ch_names):
        feats.update(extract_time_features(epoch[idx], sfreq, prefix=ch))

    feats["global_mean"] = float(np.mean(epoch))
    feats["global_std"] = float(np.std(epoch))
    return feats


# ---------------------------------------------------------------
# Epoch-level Wrapper (expected by feature_manager)
# ---------------------------------------------------------------
def extract_time_features_with_metadata(epoch, sfreq, ch_names, epoch_idx, start_s, end_s, overlap_s):
    feats = extract_all_channels(epoch, sfreq, ch_names)
    feats.update({
        "epoch_idx": int(epoch_idx),
        "epoch_start_s": float(start_s),
        "epoch_end_s": float(end_s),
        "epoch_overlap_s": float(overlap_s),
        "epoch_duration_s": float(end_s - start_s),
        "has_overlap": bool(overlap_s > 0)
    })
    return feats
