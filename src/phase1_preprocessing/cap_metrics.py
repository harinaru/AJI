# src/phase1_preprocessing/cap_metrics.py
import numpy as np
from typing import Sequence, Dict, Any, Optional
from loguru import logger


def _to_arr_safe(x):
    if x is None:
        return np.empty((0,), dtype=float)
    try:
        arr = np.asarray(x, dtype=float)
        return arr.flatten()
    except Exception:
        return np.empty((0,), dtype=float)


def _normalize_hyp(hyp) -> Optional[np.ndarray]:
    """
    Ensure hyp is a numpy array with shape (N, K).
    If hyp is 1-D (N,), convert to (N,1).
    If hyp is list-like, convert to np.ndarray.
    Returns None if hyp is None or empty.
    """
    if hyp is None:
        return None
    try:
        h = np.asarray(hyp)
        if h.size == 0:
            return None
        if h.ndim == 1:
            # reshape 1D to (N,1) so downstream indexing hyp[:,0] works
            h = h.reshape(-1, 1)
        return h
    except Exception as e:
        logger.debug(f"[CAP] normalize_hyp failed: {e}")
        return None


def compute_terzano_metrics(
    time_tot: Optional[Sequence[float]] = None,
    duration: Optional[Sequence[float]] = None,
    type_ar: Optional[Sequence[int]] = None,
    hyp: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Safe Terzano CAP metrics. All args optional (won't raise on missing data).
    If `hyp` is provided as 1-D it will be coerced to 2-D (N, 1).
    """

    startA = _to_arr_safe(time_tot)
    durA = _to_arr_safe(duration)

    n = min(len(startA), len(durA))
    startA, durA = startA[:n], durA[:n]

    if type_ar is None:
        type_ar = [0] * n
    else:
        type_ar = list(type_ar)[:n]

    endA = startA + durA

    # Normalize hyp safely
    hyp_norm = _normalize_hyp(hyp)

    NREMtime = 0.0
    if hyp_norm is not None:
        try:
            # allow if hyp_norm has more columns; only the first column is sleep stage
            stages = hyp_norm[:, 0]
            # treat stage==5 or stage==0 as non-NREM (based on your code)
            nrem_epochs = int(np.sum((stages != 5) & (stages != 0)))
            NREMtime = float(nrem_epochs * 30.0)
        except Exception as e:
            logger.debug(f"[CAP] NREMtime calc failed: {e}")
            NREMtime = 0.0

    if len(startA) == 0:
        return {
            "CAPtime": 0.0, "CAPrate": 0.0, "CAPdensity": 0.0,
            "CAPsequences": 0, "NREMtime": NREMtime,
            "A1_time": 0.0, "A2_time": 0.0, "A3_time": 0.0,
            "A1_ratio": 0.0, "A2_ratio": 0.0, "A3_ratio": 0.0,
            "A1_sequences": 0, "A2_sequences": 0, "A3_sequences": 0,
        }

    inter = startA[1:] - endA[:-1]

    seq_starts = []
    seq_ends = []

    if len(inter) > 0:
        s = 0
        for i, gap in enumerate(inter):
            if gap >= 60:
                seq_starts.append(s)
                seq_ends.append(i)
                s = i + 1
        seq_starts.append(s)
        seq_ends.append(len(inter))

    valid_sequences = [(s, e) for s, e in zip(seq_starts, seq_ends) if (e - s + 1) >= 2]

    CAPtime = 0.0
    for s, last in valid_sequences:
        seq_dur = float(endA[last] - startA[s])
        CAPtime += max(0.0, seq_dur)

    if CAPtime <= 0:
        CAPtime = float(np.sum(durA))

    A1_time = float(np.sum(durA[np.array(type_ar) == 1]))
    A2_time = float(np.sum(durA[np.array(type_ar) == 2]))
    A3_time = float(np.sum(durA[np.array(type_ar) == 3]))

    A1_seq = A2_seq = A3_seq = 0
    for s, e in valid_sequences:
        block = np.array(type_ar[s:e+1])
        if 1 in block: A1_seq += 1
        if 2 in block: A2_seq += 1
        if 3 in block: A3_seq += 1

    A1_ratio = A1_time / CAPtime if CAPtime > 0 else 0.0
    A2_ratio = A2_time / CAPtime if CAPtime > 0 else 0.0
    A3_ratio = A3_time / CAPtime if CAPtime > 0 else 0.0

    CAPrate = CAPtime / NREMtime if NREMtime > 0 else 0.0
    CAPdensity = CAPtime / (NREMtime / 60.0) if NREMtime > 0 else 0.0

    metrics = {
        "CAPtime": round(CAPtime, 3),
        "CAPrate": round(CAPrate, 5),
        "CAPdensity": round(CAPdensity, 5),
        "CAPsequences": len(valid_sequences),
        "NREMtime": round(NREMtime, 3),

        "A1_time": round(A1_time, 3),
        "A2_time": round(A2_time, 3),
        "A3_time": round(A3_time, 3),

        "A1_ratio": round(A1_ratio, 5),
        "A2_ratio": round(A2_ratio, 5),
        "A3_ratio": round(A3_ratio, 5),

        "A1_sequences": int(A1_seq),
        "A2_sequences": int(A2_seq),
        "A3_sequences": int(A3_seq),
    }

    logger.info(f"[CAP] Metrics → {metrics}")
    return metrics