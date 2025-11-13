# ==============================================================
# Frequency Domain Feature Extraction (Enhanced with Ratios + Logging)
# ==============================================================
import numpy as np
from scipy.signal import welch
from scipy.stats import entropy
from scipy.integrate import trapezoid
from typing import Dict, Union
from loguru import logger  # ✅ integrated detailed logging

# --------------------------------------------------------------
# EEG Frequency Bands (Hz)
# --------------------------------------------------------------
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "sigma": (12.0, 15.0),
    "beta": (13.0, 30.0),
}


# --------------------------------------------------------------
# Bandpower Calculation
# --------------------------------------------------------------
def bandpower_from_psd(freqs: np.ndarray, psd: np.ndarray, fmin: float, fmax: float) -> float:
    """Integrate PSD between fmin and fmax using trapezoidal rule."""
    if freqs is None or psd is None or len(freqs) == 0:
        return 0.0
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return 0.0
    try:
        power = float(trapezoid(psd[mask], freqs[mask]))
        return power
    except Exception as e:
        logger.warning(f"bandpower_from_psd failed in range {fmin}-{fmax} Hz: {e}")
        return 0.0


# --------------------------------------------------------------
# Spectral Entropy
# --------------------------------------------------------------
def spectral_entropy(psd: np.ndarray) -> float:
    """Spectral entropy (Shannon) from normalized PSD."""
    if psd is None or np.sum(psd) <= 0:
        return 0.0
    psd_norm = psd / (np.sum(psd) + 1e-12)
    return float(entropy(psd_norm))


# --------------------------------------------------------------
# Spectral Edge Frequency
# --------------------------------------------------------------
def spectral_edge(freqs: np.ndarray, psd: np.ndarray, pct: float = 0.95) -> float:
    """
    Return spectral edge frequency that contains `pct` fraction of total power.
    Example: pct=0.95 → frequency below which 95% of total PSD power lies.
    """
    if freqs is None or psd is None or len(freqs) == 0:
        return 0.0
    cumsum = np.cumsum(psd)
    total = cumsum[-1]
    if total <= 0:
        return 0.0
    thresh = pct * total
    idx = np.searchsorted(cumsum, thresh)
    idx = min(max(idx, 0), len(freqs) - 1)
    return float(freqs[idx])


# --------------------------------------------------------------
# Frequency-Domain Feature Extraction
# --------------------------------------------------------------
def extract_freq_features(epoch: np.ndarray, sfreq: float) -> Dict[str, Union[float, str]]:
    """
    Compute frequency-domain features for a single epoch.

    Includes:
      ✅ Band powers (delta, theta, alpha, sigma, beta)
      ✅ Relative powers
      ✅ Spectral entropy
      ✅ Peak frequency
      ✅ Spectral edge (95%)
      ✅ NEW: Spectral ratios (Theta/Alpha, Alpha/Beta, Delta/Theta, (Delta+Theta)/Alpha, Sigma/Beta)
    """
    feats: Dict[str, Union[float, str]] = {}

    try:
        if epoch is None:
            return {}

        # ------------------------------------------------------
        # PSD Computation
        # ------------------------------------------------------
        if getattr(epoch, "ndim", 1) == 2:
            psds = []
            freqs = None
            for ch in range(epoch.shape[0]):
                f, p = welch(epoch[ch], fs=sfreq, nperseg=int(sfreq * 2))
                psds.append(p)
                freqs = f
            psd_mean = np.mean(psds, axis=0) if psds else None
        else:
            freqs, psd_mean = welch(epoch, fs=sfreq, nperseg=int(sfreq * 2))

        if freqs is None or psd_mean is None or len(freqs) == 0:
            for band in BANDS.keys():
                feats[f"{band}_power"] = 0.0
                feats[f"{band}_rel_power"] = 0.0
            feats.update({"spec_entropy": 0.0, "peak_freq": 0.0, "spectral_edge_95": 0.0})
            return feats

        # ------------------------------------------------------
        # Total Power & Spectral Entropy
        # ------------------------------------------------------
        total_power = float(trapezoid(psd_mean, freqs)) + 1e-12
        feats["spec_entropy"] = spectral_entropy(psd_mean)

        # ------------------------------------------------------
        # Band-Specific Powers
        # ------------------------------------------------------
        band_powers = {}
        for band, (fmin, fmax) in BANDS.items():
            p = bandpower_from_psd(freqs, psd_mean, fmin, fmax)
            band_powers[band] = p
            feats[f"{band}_power"] = float(p)
            feats[f"{band}_rel_power"] = float(p / total_power) if total_power > 0 else 0.0

        # ------------------------------------------------------
        # Peak Frequency
        # ------------------------------------------------------
        try:
            peak_idx = int(np.argmax(psd_mean))
            feats["peak_freq"] = float(freqs[peak_idx])
        except Exception:
            feats["peak_freq"] = 0.0

        # ------------------------------------------------------
        # Spectral Edge (95%)
        # ------------------------------------------------------
        feats["spectral_edge_95"] = spectral_edge(freqs, psd_mean, pct=0.95)

        # ------------------------------------------------------
        # NEW: Spectral Ratio Features (Novel Additions)
        # ------------------------------------------------------
        def safe_ratio(a, b):
            """Return ratio a/b safely, avoiding div by zero."""
            return float(a / b) if b != 0 else 0.0

        feats["theta_alpha_ratio"] = safe_ratio(band_powers.get("theta", 0.0), band_powers.get("alpha", 0.0))
        feats["alpha_beta_ratio"] = safe_ratio(band_powers.get("alpha", 0.0), band_powers.get("beta", 0.0))
        feats["delta_theta_ratio"] = safe_ratio(band_powers.get("delta", 0.0), band_powers.get("theta", 0.0))
        feats["drowsiness_index"] = safe_ratio(
            band_powers.get("delta", 0.0) + band_powers.get("theta", 0.0),
            band_powers.get("alpha", 0.0),
        )
        feats["sigma_beta_ratio"] = safe_ratio(band_powers.get("sigma", 0.0), band_powers.get("beta", 0.0))

        # ------------------------------------------------------
        # Logging Summary
        # ------------------------------------------------------
        logger.debug(
            f"FreqFeats OK: peak={feats['peak_freq']:.2f}Hz | "
            f"Entropy={feats['spec_entropy']:.3f} | "
            f"Ratios: θ/α={feats['theta_alpha_ratio']:.2f}, α/β={feats['alpha_beta_ratio']:.2f}"
        )

    except Exception as e:
        # ------------------------------------------------------
        # Error Handling
        # ------------------------------------------------------
        logger.error(f"extract_freq_features failed: {e}")
        for band in BANDS.keys():
            feats[f"{band}_power"] = 0.0
            feats[f"{band}_rel_power"] = 0.0
        feats.update({
            "spec_entropy": 0.0,
            "peak_freq": 0.0,
            "spectral_edge_95": 0.0,
            "theta_alpha_ratio": 0.0,
            "alpha_beta_ratio": 0.0,
            "delta_theta_ratio": 0.0,
            "drowsiness_index": 0.0,
            "sigma_beta_ratio": 0.0,
            "error_flag": str(e)
        })

    return feats