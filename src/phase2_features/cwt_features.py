# cwt_features.py
import numpy as np
from scipy import signal
from loguru import logger
from typing import Dict, Any

def scale_entropy(energy):
    p = energy / (np.sum(energy) + 1e-12)
    return -np.sum(p * np.log2(p + 1e-12))

def extract_cwt_features(epoch: np.ndarray, sfreq: float) -> Dict[str, Any]:
    """
    Continuous wavelet transform (Morlet) - aggregated band/scale energies.
    """
    try:
        if epoch.ndim == 2:
            ch = epoch[0]  # lead with first channel (or average across channels)
        else:
            ch = epoch

        widths = np.arange(1, min(128, int(sfreq * 2)))
        cwtm = signal.cwt(ch, signal.morlet2, widths, w=5.0)
        # energy per scale
        energy = np.sum(np.abs(cwtm)**2, axis=1)
        energy = energy + 1e-12
        total = np.sum(energy)
        dominant_scale = float(np.argmax(energy))
        out = {
            "cwt_total_energy": float(total),
            "cwt_dominant_scale": dominant_scale,
            "cwt_scale_entropy": float(scale_entropy(energy))
        }
        return out
    except Exception as e:
        logger.debug(f"CWT features failed: {e}")
        return {"cwt_total_energy": 0.0, "cwt_dominant_scale": 0.0, "cwt_scale_entropy": 0.0}
