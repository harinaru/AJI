# src/phase1_preprocessing/scoring_reader.py
# v6.1_stable_final — robust REMlogic CAP parser
import os
import re
import json
import numpy as np
from typing import List, Tuple, Optional, Sequence
from loguru import logger


def _parse_time_hms(token: str) -> Optional[float]:
    """Parse HH:MM:SS or HH.MM.SS into seconds (float)."""
    if not isinstance(token, str):
        return None
    token = token.replace(".", ":").strip()
    m = re.match(r"^(\d{1,2}):(\d{2}):(\d{2})$", token)
    if not m:
        return None
    try:
        hh = int(m.group(1))
        mm = int(m.group(2))
        ss = int(m.group(3))
        return float(hh * 3600 + mm * 60 + ss)
    except Exception:
        return None


def read_remlogic_txt(
    path: str,
    output_json: Optional[str] = None
) -> Tuple[np.ndarray, List[float], List[float], List[int]]:
    """
    Parse REMlogic .txt and extract hypnogram rows and CAP A-phase events.

    Returns:
       hyp_array (Nx2 float) : columns -> [stage_code, epoch_start_s] normalized so first epoch start == 0.0
       time_tot (list[float]) : A-phase start times (seconds, absolute from file)
       duration (list[float]) : A-phase durations (seconds)
       type_ar (list[int])   : integer subtype (1..3) or 0 if unknown

    Notes:
    - hyp_array[:,1] is shifted so the first epoch start becomes 0.0.
    - time_tot/duration/type_ar remain absolute (not shifted) so they can be passed into compute_terzano_metrics directly.
    """
    # If .edf passed, try corresponding .txt
    if path.lower().endswith(".edf"):
        txt_candidate = os.path.splitext(path)[0] + ".txt"
        if not os.path.exists(txt_candidate):
            raise FileNotFoundError(f"Missing scorer file for: {path}")
        path = txt_candidate

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    logger.info(f"Parsing scorer file → {path}")

    # collect
    hyp_rows: List[Tuple[float, float]] = []  # (stage_code, epoch_start_sec)
    time_tot: List[float] = []  # A-phase starts (absolute seconds)
    duration: List[float] = []  # durations seconds
    type_ar: List[int] = []     # subtype ints

    # regexes
    time_regex = re.compile(r"\b\d{1,2}[:\.]\d{2}[:\.]\d{2}\b")
    cap_regex = re.compile(r"(MCAP|CAP|A[\s\-]?PHASE)[\s\-]*A?([123])?", flags=re.IGNORECASE)

    # common stage string -> numeric mapping
    stage_map = {
        "SLEEP-S0": 0, "S0": 0,
        "SLEEP-S1": 1, "S1": 1,
        "SLEEP-S2": 2, "S2": 2,
        "SLEEP-S3": 3, "S3": 3,
        "SLEEP-S4": 4, "S4": 4,
        "SLEEP-REM": 5, "REM": 5,
        "SLEEP-UNSCORED": 7
    }

    # read lines and keep non-empty
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw_lines = [ln.rstrip("\n\r") for ln in f]
    lines = [ln.strip() for ln in raw_lines if ln and ln.strip()]

    found_any_cap = False

    # PASS 1: modern/tabular lines (tabs or big whitespace)
    for line in lines:
        # find time token in line
        m_time = time_regex.search(line)
        if not m_time:
            continue
        time_tok = m_time.group(0)
        tsec = _parse_time_hms(time_tok)
        if tsec is None:
            continue

        # split into tokens (tabs or 2+ spaces). fallback to whitespace split
        parts = re.split(r"\t+|\s{2,}", line)
        if len(parts) < 3:
            parts = line.split()

        # stage detection (search tokens for any stage label)
        stage_val = None
        for p in parts:
            if not p:
                continue
            up = p.upper().strip()
            if up in stage_map:
                stage_val = stage_map[up]
                break
            # also handle tokens like 'S4' or 'SLEEP-S4' with punctuation
            cleaned = re.sub(r"[^\w\-]", "", up)
            if cleaned in stage_map:
                stage_val = stage_map[cleaned]
                break
        if stage_val is not None:
            hyp_rows.append((float(stage_val), float(tsec)))

        # CAP detection
        m_cap = cap_regex.search(line)
        if not m_cap:
            continue
        subtype = 0
        try:
            subtype = int(m_cap.group(2)) if m_cap.group(2) else 0
        except Exception:
            subtype = 0

        # numeric duration detection: first numeric token < 600 (seconds) near the event
        dur_val = None
        for tok in parts:
            tok = tok.strip()
            if re.fullmatch(r"\d+(\.\d+)?", tok):
                try:
                    v = float(tok)
                    # REMlogic exports often have integer durations (seconds)
                    if 0 < v < 600:
                        dur_val = float(v)
                        break
                except Exception:
                    continue

        # As a secondary heuristic: if no numeric token found, look for number immediately after CAP keyword in line
        if dur_val is None:
            after_cap = line[m_cap.end():].strip()
            first_num_match = re.search(r"\b(\d+(\.\d+)?)\b", after_cap)
            if first_num_match:
                v = float(first_num_match.group(1))
                if 0 < v < 600:
                    dur_val = float(v)

        if dur_val is None:
            # couldn't determine a duration — skip CAP event but keep searching
            continue

        time_tot.append(float(tsec))
        duration.append(float(dur_val))
        type_ar.append(int(subtype))
        found_any_cap = True

    # PASS 2: legacy fixed-column/HEADER format fallback
    if not found_any_cap:
        logger.warning("Falling back to OLD REMlogic parser mode (header-based parse)...")
        header_seen = False
        for line in lines:
            if not header_seen:
                # look for header marker (varies by export): common token 'Sleep Stage' or 'Sleep Stage' start
                if line.lower().startswith("sleep stage") or "scoring session" in line.lower():
                    header_seen = True
                continue
            parts = re.split(r"\s{2,}|\t+", line)
            if len(parts) < 2:
                continue
            # find any time token in parts
            time_tok = next((p for p in parts if time_regex.search(str(p))), None)
            if not time_tok:
                continue
            tsec = _parse_time_hms(time_regex.search(time_tok).group(0)) if time_regex.search(time_tok) else None  # pyright: ignore[reportOptionalMemberAccess]
            if tsec is None:
                continue
            # stage detection
            stage_val = None
            for p in parts:
                up = str(p).upper().strip()
                if up in stage_map:
                    stage_val = stage_map[up]
                    break
                cleaned = re.sub(r"[^\w\-]", "", up)
                if cleaned in stage_map:
                    stage_val = stage_map[cleaned]
                    break
            if stage_val is not None:
                hyp_rows.append((float(stage_val), float(tsec)))

            m_cap = cap_regex.search(line)
            if not m_cap:
                continue
            subtype = int(m_cap.group(2)) if m_cap.group(2) else 0
            # detect numeric duration token
            dur_val = None
            for tok in parts:
                if re.fullmatch(r"\d+(\.\d+)?", str(tok)):
                    v = float(tok)
                    if 0 < v < 600:
                        dur_val = float(v)
                        break
            if dur_val is not None:
                time_tot.append(float(tsec))
                duration.append(float(dur_val))
                type_ar.append(int(subtype))

    # Build hyp_array and normalize start to zero
    if hyp_rows:
        hyp_array = np.array(hyp_rows, dtype=float)
        # Sort by start time just in case parser produced out-of-order rows
        if hyp_array.shape[0] > 1:
            order = np.argsort(hyp_array[:, 1])
            hyp_array = hyp_array[order, :]
        # remember original first time for offsetting cap events later
        original_first_epoch_time = float(hyp_array[0, 1])
        # normalize start times to zero
        hyp_array[:, 1] = hyp_array[:, 1] - original_first_epoch_time
    else:
        # ensure a 2-D empty array (shape (0,2)) to avoid downstream indexing errors
        hyp_array = np.empty((0, 2), dtype=float)
        original_first_epoch_time = 0.0

    # If requested, write cap events JSON using times relative to hyp offset
    if output_json:
        cap_events = []
        for t_abs, d, s in zip(time_tot, duration, type_ar):
            # convert absolute start to relative to hyp offset
            cap_events.append({
                "subtype": f"A{s}" if s else "A?",
                "start_s": float(t_abs - original_first_epoch_time),
                "end_s": float(t_abs + d - original_first_epoch_time),
                "duration": float(d),
            })
        try:
            with open(output_json, "w", encoding="utf-8") as fo:
                json.dump(cap_events, fo, indent=2)
        except Exception:
            logger.debug("Could not save cap annotations JSON")

    logger.success(f"Parsed {hyp_array.shape[0]} hyp rows, {len(time_tot)} CAP events from {os.path.basename(path)}")
    return hyp_array, time_tot, duration, type_ar
