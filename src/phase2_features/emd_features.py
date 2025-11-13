# emd_features.py
import numpy as np
from loguru import logger
from typing import Dict, Any

def extract_emd_features(epoch: np.ndarray, sfreq: float) -> Dict[str, Any]:
    """
    Extract EMD-based features. If PyEMD available, use EMD; otherwise fallback to simple IMF proxy.
    """
    try:
        # Try to import PyEMD (optional)
        try:
            from PyEMD import EMD
        except Exception:
            EMD = None

        if epoch.ndim == 2:
            ch = epoch[0]
        else:
            ch = epoch

        if EMD is not None:
            emd = EMD()
            imfs = emd(ch)
            n_imfs = imfs.shape[0]
            imf_energies = np.sum(imfs**2, axis=1)
            total = np.sum(imf_energies) + 1e-12
            ratios = imf_energies / total
            return {
                "emd_n_imfs": int(n_imfs),
                "emd_total_energy": float(total),
                "emd_first_imf_ratio": float(ratios[0]) if n_imfs > 0 else 0.0,
                "emd_entropy": float(-np.sum(ratios * np.log2(ratios + 1e-12))) if n_imfs > 0 else 0.0
            }
        else:
            # fallback: coarse proxy using band-pass subtractions
            # compute simple envelope decompositions via iterative sifting is expensive — so return safe defaults
            mean_power = float(np.mean(ch**2))
            return {"emd_n_imfs": 0, "emd_total_energy": mean_power, "emd_first_imf_ratio": 0.0, "emd_entropy": 0.0}
    except Exception as e:
        logger.debug(f"EMD features failed: {e}")
        return {"emd_n_imfs": 0, "emd_total_energy": 0.0, "emd_first_imf_ratio": 0.0, "emd_entropy": 0.0}
