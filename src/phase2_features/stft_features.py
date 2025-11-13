# stft_features.py
import numpy as np
import scipy.signal as sps
from loguru import logger
from typing import Dict, Any

def spectral_entropy(psd):
    psd = np.maximum(psd, 1e-12)
    psd_norm = psd / np.sum(psd)
    return -np.sum(psd_norm * np.log2(psd_norm + 1e-12))

def spectral_rolloff(freqs, psd, rolloff_pct=0.85):
    cumsum = np.cumsum(psd)
    threshold = rolloff_pct * cumsum[-1]
    idx = np.searchsorted(cumsum, threshold)
    return float(freqs[min(idx, len(freqs)-1)])

def spectral_flatness(psd):
    psd = np.maximum(psd, 1e-12)
    geo = np.exp(np.mean(np.log(psd)))
    arith = np.mean(psd)
    return float(geo / (arith + 1e-12))

def extract_stft_features(epoch: np.ndarray, sfreq: float) -> Dict[str, Any]:
    """
    epoch: shape (n_channels, n_samples) or (n_samples,)
    returns dictionary of STFT-derived features (per-epoch, aggregated across channels)
    """
    try:
        if epoch.ndim == 2:
            ch_data = epoch
        else:
            ch_data = epoch[np.newaxis, :]

        # We'll compute per-channel then average
        feats_per_ch = []
        for ch in ch_data:
            f, t, Z = sps.stft(ch, fs=sfreq, nperseg=int(sfreq*2), noverlap=int(sfreq*1.5))
            psd = np.mean(np.abs(Z)**2, axis=1)  # avg across time frames
            if np.sum(psd) <= 0:
                psd = np.ones_like(psd) * 1e-12
            feats_per_ch.append({
                "stft_entropy": spectral_entropy(psd),
                "stft_rolloff": spectral_rolloff(f, psd),
                "stft_flatness": spectral_flatness(psd),
                "stft_bandwidth": float(np.sqrt(np.sum(psd * (f - np.sum(f*psd)/np.sum(psd))**2) / np.sum(psd)))
            })

        # average across channels
        keys = feats_per_ch[0].keys()
        out = {}
        for k in keys:
            out[k] = float(np.mean([fc[k] for fc in feats_per_ch]))
        return out
    except Exception as e:
        logger.debug(f"STFT features failed: {e}")
        return {
            "stft_entropy": 0.0,
            "stft_rolloff": 0.0,
            "stft_flatness": 0.0,
            "stft_bandwidth": 0.0
        }
