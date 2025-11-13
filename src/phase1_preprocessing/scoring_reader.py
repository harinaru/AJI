# ===============================================================
# scoring_reader.py — v6.0_stable_final
# Ultra-robust REMlogic CAP parser (CAP Sleep Database compatible)
# ===============================================================

import os
import re
import json
import numpy as np
from typing import List, Tuple, Optional
from loguru import logger


# ---------------------------------------------------------------
# Safe time parser
# ---------------------------------------------------------------
def _parse_time_hms(token: str) -> Optional[float]:
    """
    Accepts hh:mm:ss or hh.mm.ss → converts to seconds.
    Returns None if the token is invalid.
    """
    if not isinstance(token, str):
        return None

    token = token.replace(".", ":").strip()
    m = re.match(r"^(\d{1,2}):(\d{2}):(\d{2})$", token)
    if not m:
        return None

    try:
        h, m_, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return h * 3600 + m_ * 60 + s
    except Exception:
        return None


# ---------------------------------------------------------------
# MAIN PARSER (single entry point)
# ---------------------------------------------------------------
def read_remlogic_txt(
    path: str,
    output_json: Optional[str] = None
) -> Tuple[np.ndarray, List[float], List[float], List[int]]:
    """
    Parses REMlogic scorer text files used in the CAP Sleep Database.

    Returns:
        hyp       → Nx2 array: stage_code, epoch_start_sec
        time_tot  → A-phase start times (sec)
        duration  → A-phase durations (sec)
        type_ar   → A-phase subtype (1,2,3)
    """

    # ----------------- Auto-match EDF to TXT -----------------
    if path.lower().endswith(".edf"):
        txt_candidate = os.path.splitext(path)[0] + ".txt"
        if not os.path.exists(txt_candidate):
            raise FileNotFoundError(f"Missing scorer file for: {path}")
        path = txt_candidate

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    # ----------------- Containers -----------------
    hyp_rows: List[Tuple[int, float]] = []
    time_tot: List[float] = []
    duration: List[float] = []
    type_ar: List[int] = []

    # ----------------- Patterns -----------------
    time_regex = re.compile(r"\b\d{1,2}[:\.]\d{2}[:\.]\d{2}\b")
    cap_regex = re.compile(r"(MCAP|CAP|A[\s\-]?PHASE)[\s\-]*A?([123])?", re.I)

    # Sleep stage mapping
    stage_map = {
        "SLEEP-S0": 0, "SLEEP-S1": 1, "SLEEP-S2": 2,
        "SLEEP-S3": 3, "SLEEP-S4": 4,
        "SLEEP-REM": 5,
        "SLEEP-UNSCORED": 7
    }

    # ---------------------------------------------------------
    # Load lines
    # ---------------------------------------------------------
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f if line.strip()]

    found_any_cap = False

    # =========================================================
    # PASS 1 — TABULAR FORMAT (the real CAP Sleep Database)
    # =========================================================
    for line in lines:
        if not time_regex.search(line):
            continue

        parts = re.split(r"\t+|\s{2,}", line)
        if len(parts) < 3:
            parts = line.split()
        if len(parts) < 3:
            continue

        # ---- Find time token ----
        time_tok: Optional[str] = None
        time_idx: Optional[int] = None

        for idx, p in enumerate(parts):
            if time_regex.match(p):
                time_tok = p
                time_idx = idx
                break

        if time_tok is None:
            continue

        tsec = _parse_time_hms(time_tok)
        if tsec is None:
            continue

        # ---- Stage detection ----
        stage_val: Optional[int] = None
        for p in parts:
            up = p.upper()
            if up in stage_map:
                stage_val = stage_map[up]
                break
            if up.startswith("S") and len(up) >= 2 and up[1].isdigit():
                stage_val = int(up[1])
                break

        if stage_val is not None:
            hyp_rows.append((stage_val, tsec))

        # ---- CAP / A-phase detection ----
        cap_match = cap_regex.search(line)
        if not cap_match:
            continue

        subtype = int(cap_match.group(2)) if cap_match.group(2) else 0

        # Duration => nearest numeric token after event
        dur: Optional[float] = None
        for tok in parts:
            if re.fullmatch(r"\d+(\.\d+)?", tok):
                v = float(tok)
                if 0 < v < 600:     # duration in seconds
                    dur = v
                    break

        if dur is None:
            continue

        time_tot.append(tsec)
        duration.append(dur)
        type_ar.append(subtype)
        found_any_cap = True

    # =========================================================
    # PASS 2 — OLD SLEEP-STAGE HEADER MODE
    # =========================================================
    if not found_any_cap:
        header_seen = False

        for line in lines:
            if not header_seen:
                if line.lower().startswith("sleep stage"):
                    header_seen = True
                continue

            parts = re.split(r"\s{2,}|\t+", line)
            if len(parts) < 3:
                continue

            time_tok = next((p for p in parts if time_regex.match(p)), None)
            if time_tok is None:
                continue

            tsec = _parse_time_hms(time_tok)
            if tsec is None:
                continue

            # Stage parsing
            stage_val = None
            for p in parts:
                up = p.upper().strip()
                if up in stage_map:
                    stage_val = stage_map[up]
                    break

            if stage_val is not None:
                hyp_rows.append((stage_val, tsec))

            # CAP parsing
            mcap = cap_regex.search(line)
            if not mcap:
                continue

            subtype = int(mcap.group(2)) if mcap.group(2) else 0

            dur = None
            for tok in parts:
                if re.fullmatch(r"\d+(\.\d+)?", tok):
                    v = float(tok)
                    if 0 < v < 600:
                        dur = v
                        break

            if dur:
                time_tot.append(tsec)
                duration.append(dur)
                type_ar.append(subtype)

    # ---------------------------------------------------------
    # BUILD hyp array
    # ---------------------------------------------------------
    hyp_array = (
        np.array(hyp_rows, dtype=float)
        if hyp_rows else
        np.empty((0, 2), dtype=float)
    )

    if hyp_array.size:
        hyp_array[:, 1] -= hyp_array[0, 1]

    # ---------------------------------------------------------
    # Save JSON (optional)
    # ---------------------------------------------------------
    if output_json:
        cap_events = [
            {
                "subtype": f"A{s}" if s else "A?",
                "start_s": float(t - hyp_array[0,1]),
                "end_s": float((t + d) - hyp_array[0,1]),
                "duration": float(d),
            }
            for t, d, s in zip(time_tot, duration, type_ar)
        ]

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(cap_events, f, indent=2)

    logger.success(
        f"Parsed {len(hyp_rows)} hyp rows, {len(time_tot)} CAP events "
        f"from {os.path.basename(path)}"
    )

    return hyp_array, time_tot, duration, type_ar
