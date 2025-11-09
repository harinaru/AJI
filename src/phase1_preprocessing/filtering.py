import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from loguru import logger

def _butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq if lowcut is not None else 0.0
    high = highcut / nyq if highcut is not None else 1.0
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass(data, fs, low=0.3, high=35.0, order=5):
    """
    data: ndarray (n_channels, n_samples)
    """
    if low is None or high is None:
        return data
    b, a = _butter_bandpass(low, high, fs, order=order)
    filtered = filtfilt(b, a, data, axis=1)
    return filtered

def apply_notch(data, fs, freqs=[50.0], Q=30.0):
    """Apply notch filters for each frequency in freqs."""
    out = data.copy()
    for f0 in freqs:
        b, a = iirnotch(f0, Q, fs)
        out = filtfilt(b, a, out, axis=1)
    return out

def apply_filters(data, sfreq, low=0.3, high=35.0, notch_freqs=None):
    """
    Apply notch then bandpass. Returns filtered data and meta info.
    """
    try:
        filtered = data.astype(float)
        if notch_freqs:
            filtered = apply_notch(filtered, sfreq, freqs=notch_freqs)
        filtered = apply_bandpass(filtered, sfreq, low=low, high=high, order=4)
        meta = {"low": low, "high": high, "notch": notch_freqs}
        return filtered, meta
    except Exception as e:
        logger.exception(f"Filtering error: {e}")
        raise

def detect_artifacts_threshold(data, sfreq, thresh_uV=200.0, window_s=1.0):
    """
    Simple artifact detector: returns boolean mask of length n_samples (True if bad).
    Marks samples where absolute amplitude exceeds thresh_uV across any channel in sliding windows.
    Note: threshold uses same units as data (user must ensure units). We record results for QC.
    """
    n_channels, n_samples = data.shape
    abs_data = np.abs(data)
    max_across = np.max(abs_data, axis=0)
    # assume data in microvolts? This is dataset dependent. Use thresh directly.
    bad_mask = max_across > thresh_uV
    return bad_mask
