# ===============================================================
# src/phase1_preprocessing/data_loader.py
# v4.3.7_stable – NumPy 2.0 safe + memory-optimized EDF/MAT/CSV loader
# ===============================================================
import os
import numpy as np
from loguru import logger
from typing import List, Dict, Any, Tuple, Union

# Optional imports
try:
    import mne
except Exception:
    mne = None

try:
    import scipy.io
except Exception:
    scipy = None


# ---------------------------------------------------------------
def discover_subjects(raw_dir: str) -> List[str]:
    """Discover EDF subjects in the raw folder."""
    subjects = [
        os.path.splitext(f)[0]
        for f in os.listdir(raw_dir)
        if os.path.isfile(os.path.join(raw_dir, f))
        and f.lower().endswith(".edf")
    ]
    logger.info(f"discover_subjects: found {len(subjects)} EDF subjects")
    return subjects


# ---------------------------------------------------------------
# EDF reader (safe / adaptive)
# ---------------------------------------------------------------
def read_edf_mne(edf_path: str) -> Tuple[np.ndarray, float, List[str]]:
    """
    Safely load EDF via MNE (handles huge files and NumPy 2.0).
    Returns:
        data: np.ndarray (n_channels, n_samples)
        sfreq: float
        ch_names: list
    """
    if mne is None:
        raise ModuleNotFoundError("Please install mne (conda-forge recommended).")

    # Lazy metadata load
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    sfreq = float(raw.info.get("sfreq", 100.0))
    ch_names = raw.info.get("ch_names", [])

    # Downsample if too high frequency (to prevent OOM)
    if sfreq > 200:
        raw.resample(200)
        sfreq = 200.0
        logger.info(f"⏬ Downsampled {os.path.basename(edf_path)} → 200 Hz")

    # ✅ Safe NumPy 2.0 array conversion
    data = np.asarray(raw.get_data(), dtype=np.float32)
    logger.info(f"✅ EDF loaded safely → shape={data.shape}, sfreq={sfreq:.1f} Hz")
    return data, sfreq, ch_names


# ---------------------------------------------------------------
def read_mat_file(mat_path: str) -> Tuple[np.ndarray, float, List[str]]:
    """Load Matlab .mat files robustly."""
    if scipy is None:
        raise ModuleNotFoundError("scipy is required to read .mat files.")

    mat = scipy.io.loadmat(mat_path)
    data: Union[np.ndarray, None] = None

    # Try common variable names first
    for key in ["data", "signal"]:
        if key in mat:
            data = mat[key]
            break

    # Fallback: pick largest 2D array
    if data is None:
        candidates = [v for v in mat.values() if isinstance(v, np.ndarray) and v.ndim == 2]
        if candidates:
            data = max(candidates, key=lambda a: a.size)
        else:
            raise ValueError(f"No 2D numeric array found in {mat_path}")

    sfreq = float(mat.get("sfreq", mat.get("sampling_rate", 100.0)))
    data = np.asarray(data, dtype=np.float32)
    logger.info(f"✅ MAT loaded safely → shape={data.shape}, sfreq={sfreq}")
    return data, sfreq, []


# ---------------------------------------------------------------
def read_txt_csv(path: str) -> Tuple[np.ndarray, float, List[str]]:
    """Read numeric text/CSV data safely."""
    import pandas as pd
    df = pd.read_csv(path, header=None)
    arr = df.to_numpy(dtype=np.float32)

    # Transpose if rows >> cols (time-major)
    if arr.shape[0] > arr.shape[1]:
        arr = arr.T

    sfreq = 100.0  # Default if unknown
    logger.info(f"✅ CSV/TXT loaded safely → shape={arr.shape}, sfreq={sfreq}")
    return arr, sfreq, []


# ---------------------------------------------------------------
def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def read_st_annotations(st_path: str) -> List[Dict[str, Union[float, str]]]:
    """Parse .st or .edf.st annotation files."""
    anns = []
    if not os.path.exists(st_path):
        return anns

    with open(st_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.replace(",", " ").split()
            if len(parts) >= 2 and _is_number(parts[0]) and _is_number(parts[1]):
                start, val2 = float(parts[0]), float(parts[1])
                end = val2 if val2 > start else start + val2
                label = " ".join(parts[2:]) if len(parts) > 2 else ""
                anns.append({"start": start, "end": end, "label": label})

    logger.info(f"read_st_annotations: {len(anns)} entries parsed from {os.path.basename(st_path)}")
    return anns


# ---------------------------------------------------------------
def load_subject_normalized(subid: str, raw_dir: str) -> Dict[str, Any]:
    """
    Load a subject (EDF/MAT/CSV/TXT) and return standardized structure:
    {
        'subject': subid,
        'data': ndarray (n_channels, n_samples),
        'sfreq': float,
        'ch_names': list,
        'annotations': list[dict],
        'source_path': str
    }
    """
    candidates = []
    for ext in [".edf", ".mat", ".csv", ".txt"]:
        path = os.path.join(raw_dir, subid + ext)
        if os.path.exists(path):
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(f"No supported file for subject {subid} in {raw_dir}")

    chosen = candidates[0]
    ext = os.path.splitext(chosen)[1].lower()
    annotations, ch_names = [], []
    logger.info(f"Loading subject {subid} → {chosen}")

    # Load appropriate file type
    if ext == ".edf":
        data, sfreq, ch_names = read_edf_mne(chosen)
        for cand in [chosen + ".st", os.path.join(raw_dir, subid + ".st")]:
            if os.path.exists(cand):
                annotations = read_st_annotations(cand)
                break
    elif ext == ".mat":
        data, sfreq, ch_names = read_mat_file(chosen)
    else:
        data, sfreq, ch_names = read_txt_csv(chosen)

    # Sanity checks — enforce (channels, samples)
    data = np.atleast_2d(np.asarray(data, dtype=np.float32))
    if data.shape[0] > data.shape[1]:
        data = data.T

    logger.success(f"✅ Loaded {subid} | shape={data.shape} | sfreq={sfreq:.2f} Hz")

    return {
        "subject": subid,
        "data": data,
        "sfreq": float(sfreq),
        "ch_names": ch_names,
        "annotations": annotations,
        "source_path": chosen,
    }
