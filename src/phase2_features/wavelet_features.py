# ==============================================================
# Wavelet-Based Energy Feature Extraction
# ==============================================================
import numpy as np
import pywt
from typing import Dict, Union, Optional
from loguru import logger


def extract_wavelet_energy(
    epoch: np.ndarray,
    sfreq: float,
    wavelet: str = "db4",
    level: Optional[int] = None
) -> Dict[str, Union[float, str]]:
    """
    Compute wavelet energy features for an EEG epoch.

    Parameters
    ----------
    epoch : np.ndarray
        1D (samples,) or 2D (channels x samples)
    sfreq : float
        Sampling frequency (e.g. 512 Hz)
    wavelet : str
        Mother wavelet (default = 'db4')
    level : Optional[int]
        Decomposition level (auto-detected if None)

    Returns
    -------
    feats : dict
        Absolute & relative energy per level
        {'wavelet_energy_L0', ..., 'wavelet_energy_Ln_rel'}
    """
    feats: Dict[str, Union[float, str]] = {}

    try:
        # --- Step 1: Flatten or average multi-channel ---
        if getattr(epoch, "ndim", 1) == 2:
            x = np.mean(epoch, axis=0)
            logger.debug(f"Wavelet: Averaged {epoch.shape[0]} channels.")
        else:
            x = epoch

        # --- Step 2: Get valid wavelet object ---
        try:
            wavelet_obj = pywt.Wavelet(wavelet)  # type: ignore[attr-defined]
        except Exception as e:
            logger.warning(f"Invalid wavelet '{wavelet}', fallback to 'db4': {e}")
            wavelet_obj = pywt.Wavelet("db4")  # type: ignore[attr-defined]

        # --- Step 3: Compute decomposition level ---
        max_level = pywt.dwt_max_level(len(x), wavelet_obj.dec_len)
        if level is None:
            level = max_level

        # --- Step 4: Decompose signal ---
        coeffs = pywt.wavedec(x, wavelet_obj, level=level)
        energies = [float(np.sum(np.square(c))) for c in coeffs]
        total_energy = sum(energies) + 1e-12

        # --- Step 5: Build feature dict ---
        for i, e in enumerate(energies):
            feats[f"wavelet_energy_L{i}"] = float(e)
            feats[f"wavelet_energy_L{i}_rel"] = float(e / total_energy)

        logger.debug(f"Wavelet levels={len(energies)}, total={total_energy:.3e}")

    except Exception as e:
        logger.error(f"Wavelet extraction failed: {e}")
        feats["wavelet_error"] = str(e)

    return feats
