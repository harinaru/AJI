# ===============================================================
# src/phase1_preprocessing/cap_metrics.py
# v5.1_research_final — Extended Terzano CAP metrics with subtype analysis
# ===============================================================
"""
Compute Terzano CAP metrics from A-phase times/durations and hypnogram.
Port of CAP.m (Terzano rules) to Python, adapted for safe use with list/array inputs.
Now extended with per-subtype CAP (A1/A2/A3) duration and ratio metrics.
"""

import numpy as np
from typing import Sequence, Dict, Any, Optional
from loguru import logger


def compute_terzano_metrics(
    time_tot: Sequence[float],
    duration: Sequence[float],
    hyp: Optional[np.ndarray],
    type_ar: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """
    Args:
      time_tot : list or np.ndarray
          A-phase start times (in seconds, absolute or relative)
      duration : list or np.ndarray
          A-phase durations (seconds)
      hyp : np.ndarray
          Hypnogram [n_epochs × 2], hyp[:,0]=stage (0–5), hyp[:,1]=epoch_start_seconds
      type_ar : list or np.ndarray, optional
          CAP subtype labels (1=A1, 2=A2, 3=A3). Can be None.

    Returns
    -------
      Dict[str, float]:
        - CAPtime: total CAP time (s)
        - CAPrate: CAPtime / NREMtime
        - CAPsequences: number of CAP sequences
        - CAPdensity: CAPtime per minute of NREM sleep
        - NREMtime: total NREM duration (s)
        - A1_time, A2_time, A3_time: total subtype durations (s)
        - A1_ratio, A2_ratio, A3_ratio: ratios per subtype
        - A1_seq, A2_seq, A3_seq: counts per subtype
    """

    # === Ensure numeric arrays ===
    time_tot_arr = np.asarray(time_tot, dtype=float)
    duration_arr = np.asarray(duration, dtype=float)
    type_arr = np.asarray(type_ar, dtype=int) if type_ar is not None else np.zeros_like(duration_arr, dtype=int)

    if time_tot_arr.size < 2:
        logger.info("⚠️ Not enough A-phases for CAP sequence computation.")
        return {
            "CAPtime": 0.0,
            "CAPrate": 0.0,
            "CAPsequences": 0,
            "CAPdensity": 0.0,
            "NREMtime": 0.0,
            "A1_time": 0.0, "A2_time": 0.0, "A3_time": 0.0,
            "A1_ratio": 0.0, "A2_ratio": 0.0, "A3_ratio": 0.0,
            "A1_seq": 0, "A2_seq": 0, "A3_seq": 0,
        }

    # === Compute CAP A/B phase relationships ===
    startA = time_tot_arr
    startB = time_tot_arr + duration_arr
    durationB = startA[1:] - startB[:-1]

    count = 0
    cap_seq_count = 0
    CAPf, CAPs = [], []

    for k in range(len(durationB)):
        if durationB[k] < 60:
            count += 1
        else:
            if count >= 2:
                cap_seq_count += 1
                CAPf.append(k)
                CAPs.append(k - count)
            count = 0
        if k == len(durationB) - 1 and count >= 2:
            cap_seq_count += 1
            CAPf.append(k + 1)
            CAPs.append(k - count + 1)

    CAPf = np.asarray(CAPf, dtype=int)
    CAPs = np.asarray(CAPs, dtype=int)

    # === Compute NREM time ===
    NREMtime = 0.0
    if hyp is not None and isinstance(hyp, np.ndarray) and hyp.size > 0:
        nrem_epochs = np.sum((hyp[:, 0] != 5) & (hyp[:, 0] != 0))
        NREMtime = float(nrem_epochs * 30.0)

    # === Compute CAP time ===
    CAPtime = float(np.sum(duration_arr)) if duration_arr.size > 0 else 0.0

    # === Subtype breakdown ===
    A1_time = float(np.sum(duration_arr[type_arr == 1])) if np.any(type_arr == 1) else 0.0
    A2_time = float(np.sum(duration_arr[type_arr == 2])) if np.any(type_arr == 2) else 0.0
    A3_time = float(np.sum(duration_arr[type_arr == 3])) if np.any(type_arr == 3) else 0.0

    A1_seq = int(np.sum(type_arr == 1))
    A2_seq = int(np.sum(type_arr == 2))
    A3_seq = int(np.sum(type_arr == 3))

    total_cap = CAPtime if CAPtime > 0 else 1.0
    A1_ratio = round(A1_time / total_cap, 4)
    A2_ratio = round(A2_time / total_cap, 4)
    A3_ratio = round(A3_time / total_cap, 4)

    # === Derive metrics ===
    CAPrate = CAPtime / NREMtime if NREMtime > 0 else 0.0
    CAPdensity = CAPtime / (NREMtime / 60.0) if NREMtime > 0 else 0.0

    metrics = {
        "CAPtime": round(CAPtime, 3),
        "CAPrate": round(CAPrate, 5),
        "CAPdensity": round(CAPdensity, 5),
        "CAPsequences": int(cap_seq_count),
        "NREMtime": round(NREMtime, 3),
        "A1_time": round(A1_time, 3),
        "A2_time": round(A2_time, 3),
        "A3_time": round(A3_time, 3),
        "A1_ratio": A1_ratio,
        "A2_ratio": A2_ratio,
        "A3_ratio": A3_ratio,
        "A1_seq": A1_seq,
        "A2_seq": A2_seq,
        "A3_seq": A3_seq,
    }

    logger.info("✅ CAP metrics computed successfully:")
    logger.info(f"   CAPtime={CAPtime:.2f}s | CAPrate={CAPrate:.3f} | CAPdensity={CAPdensity:.3f}")
    logger.info(f"   Subtypes: A1={A1_time:.2f}s ({A1_ratio*100:.1f}%), "
                f"A2={A2_time:.2f}s ({A2_ratio*100:.1f}%), "
                f"A3={A3_time:.2f}s ({A3_ratio*100:.1f}%) | CAPseq={cap_seq_count}")

    return metrics
