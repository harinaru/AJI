# src/phase2_features/cap_features.py
"""
Per-epoch CAP microstructure features (robust).
Returns per-epoch dict including subtype breakdown (A1/A2/A3)
"""
import numpy as np
from typing import Optional, List, Dict, Any

def _safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

def extract_cap_features(epoch_idx: int,
                         epoch_times: Optional[np.ndarray],
                         epoch_labels: Optional[np.ndarray],
                         cap_events: Optional[List[Dict[str, Any]]] = None,
                         cap_metrics: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Per-epoch features returned:
     - cap_present (0/1)
     - cap_subtype (dominant e.g., 'A1'/'A2'/'A3'/'A?')
     - cap_a_ratio (seconds overlap / epoch_len)
     - cap_A_start (first overlapping A start absolute or NaN)
     - cap_A_duration (first overlapping A duration or NaN)
     - cap_A1_time, cap_A2_time, cap_A3_time (overlap sec per subtype)
     - cap_A1_ratio, cap_A2_ratio, cap_A3_ratio (per epoch / epoch_len)
    """
    epoch_len = 30.0
    res = {
        "cap_present": 0,
        "cap_subtype": "A?",
        "cap_a_ratio": 0.0,
        "cap_A_start": np.nan,
        "cap_A_duration": np.nan,
        "cap_A1_time": 0.0,
        "cap_A2_time": 0.0,
        "cap_A3_time": 0.0,
        "cap_A1_ratio": 0.0,
        "cap_A2_ratio": 0.0,
        "cap_A3_ratio": 0.0,
    }

    try:
        if epoch_times is None or epoch_times.size == 0:
            e_start = float(epoch_idx * 5.0)
            e_end = e_start + epoch_len
        else:
            e_start = float(epoch_times[epoch_idx, 0])
            e_end = float(epoch_times[epoch_idx, 1])
            epoch_len = float(e_end - e_start) if (e_end > e_start) else epoch_len
    except Exception:
        e_start = float(epoch_idx * 5.0)
        e_end = e_start + epoch_len

    overlapping = []
    if cap_events:
        for ev in cap_events:
            try:
                a = float(ev.get("start_s", np.nan))
                # duration may be present in 'duration' or end_s - start_s
                d = ev.get("duration", None)
                if d is None:
                    b = float(ev.get("end_s", a))
                    d = b - a
                else:
                    b = a + float(d)
                if not (b <= e_start or a >= e_end):
                    overlapping.append({
                        "start_s": a,
                        "end_s": b,
                        "duration": float(d),
                        "subtype": ev.get("subtype", "A?")
                    })
            except Exception:
                continue

    # fallback to epoch labels: if label==1 treat as CAP over full epoch
    if not overlapping and epoch_labels is not None and len(epoch_labels) > epoch_idx:
        try:
            lab = int(epoch_labels[epoch_idx])
            if lab == 1:
                overlapping.append({
                    "start_s": e_start,
                    "end_s": e_end,
                    "duration": float(e_end - e_start),
                    "subtype": "A?"
                })
        except Exception:
            pass

    if not overlapping:
        return res

    total_overlap = 0.0
    subtype_totals = {"A1": 0.0, "A2": 0.0, "A3": 0.0, "A?": 0.0}
    first_ev = overlapping[0]
    res["cap_A_start"] = _safe_float(first_ev.get("start_s", np.nan))
    res["cap_A_duration"] = _safe_float(first_ev.get("duration", np.nan))

    for ev in overlapping:
        a = max(e_start, float(ev.get("start_s", e_start)))
        b = min(e_end, float(ev.get("end_s", e_end)))
        over = max(0.0, b - a)
        total_overlap += over
        st = str(ev.get("subtype", "A?")).upper()
        if st.startswith("A1"):
            subtype_totals["A1"] += over
        elif st.startswith("A2"):
            subtype_totals["A2"] += over
        elif st.startswith("A3"):
            subtype_totals["A3"] += over
        else:
            subtype_totals["A?"] += over

    res["cap_present"] = 1
    res["cap_a_ratio"] = float(total_overlap / epoch_len) if epoch_len > 0 else 0.0

    res["cap_A1_time"] = float(subtype_totals["A1"])
    res["cap_A2_time"] = float(subtype_totals["A2"])
    res["cap_A3_time"] = float(subtype_totals["A3"])

    res["cap_A1_ratio"] = float(subtype_totals["A1"] / epoch_len) if epoch_len > 0 else 0.0
    res["cap_A2_ratio"] = float(subtype_totals["A2"] / epoch_len) if epoch_len > 0 else 0.0
    res["cap_A3_ratio"] = float(subtype_totals["A3"] / epoch_len) if epoch_len > 0 else 0.0

    # dominant subtype
    dominant = "A?"
    dom_val = -1.0
    for k in ("A1", "A2", "A3", "A?"):
        v = subtype_totals.get(k, 0.0)
        if v > dom_val:
            dom_val = v
            dominant = k
    if dom_val <= 0.0:
        dominant = "A?"
    res["cap_subtype"] = dominant

    return res
