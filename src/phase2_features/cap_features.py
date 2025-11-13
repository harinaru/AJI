# src/phase2_features/cap_features.py
"""
Per-epoch CAP microstructure features.
Provides a simple, robust extractor that works with:
 - epoch_idx (int)
 - epoch_times: np.ndarray shape (n_epochs, 2) with columns [start_s, end_s]
 - epoch_labels (optional)
 - cap_events: list of dicts with keys 'start_s','end_s','subtype' (if available)
 - cap_metrics: dict (global metrics) optional
"""
import numpy as np
from typing import Optional, List, Dict, Any

def extract_cap_features(epoch_idx: int,
                         epoch_times: Optional[np.ndarray],
                         epoch_labels: Optional[np.ndarray],
                         cap_events: Optional[List[Dict[str, Any]]] = None,
                         cap_metrics: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    # defaults
    res = {
        "cap_present": 0,
        "cap_subtype": "NA",
        "cap_a_ratio": 0.0,
        "cap_A_start": np.nan,
        "cap_A_duration": np.nan,
    }

    if epoch_times is None or epoch_times.size == 0:
        # fallback to label-based (binary)
        if epoch_labels is not None and len(epoch_labels) > epoch_idx and epoch_labels[epoch_idx] == 1:
            res["cap_present"] = 1
        return res

    try:
        e_start = float(epoch_times[epoch_idx, 0])
        e_end = float(epoch_times[epoch_idx, 1])
    except Exception:
        e_start = epoch_idx * 5.0
        e_end = e_start + 30.0

    if cap_events:
        overlaps = []
        for ev in cap_events:
            a = float(ev.get("start_s", 0.0))
            b = float(ev.get("end_s", 0.0))
            if not (b <= e_start or a >= e_end):
                overlaps.append(ev)
        if overlaps:
            # mark presence, take the first overlapping event for subtype and timing
            total_overlap = 0.0
            for ev in overlaps:
                a = max(e_start, float(ev.get("start_s", 0.0)))
                b = min(e_end, float(ev.get("end_s", 0.0)))
                total_overlap += max(0.0, b - a)
            res["cap_present"] = 1
            res["cap_a_ratio"] = float(total_overlap / (e_end - e_start))
            first = overlaps[0]
            res["cap_subtype"] = first.get("subtype", "A?")
            res["cap_A_start"] = float(first.get("start_s", np.nan))
            res["cap_A_duration"] = float(first.get("duration", np.nan))
    else:
        # fallback: use epoch_labels if provided (1 == event)
        if epoch_labels is not None and len(epoch_labels) > epoch_idx and int(epoch_labels[epoch_idx]) == 1:
            res["cap_present"] = 1
            res["cap_a_ratio"] = 1.0

    # global metrics (if provided) are not per-epoch but we return None here; feature manager will attach them
    return res
