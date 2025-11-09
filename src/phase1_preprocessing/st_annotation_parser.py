import re
import json
import numpy as np
from loguru import logger

def parse_st_file(st_path: str, output_json: str):
    """
    Parses SleepTech .st annotation file and saves to JSON.
    Expected .st format lines like:
      "CAP A1 1234 1264"
    where values are start_time end_time in seconds.
    """
    cap_events = []
    with open(st_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            match = re.match(r"CAP\s+(A[1-3])\s+(\d+\.?\d*)\s+(\d+\.?\d*)", line)
            if match:
                subtype, start_s, end_s = match.groups()
                cap_events.append({
                    "subtype": subtype,
                    "start_s": float(start_s),
                    "end_s": float(end_s)
                })

    with open(output_json, "w") as f:
        json.dump(cap_events, f, indent=2)
    logger.info(f"Parsed {len(cap_events)} CAP events → {output_json}")
    return cap_events


def map_caps_to_epochs(cap_events, epoch_length_s, epoch_step_s, n_epochs):
    """Returns arrays of cap_present, cap_subtype, cap_a_ratio."""
    cap_present = np.zeros(n_epochs)
    cap_subtype = np.array(["NA"] * n_epochs)
    cap_a_ratio = np.zeros(n_epochs)

    for i in range(n_epochs):
        start = i * epoch_step_s
        end = start + epoch_length_s
        overlaps = [e for e in cap_events if not (e["end_s"] < start or e["start_s"] > end)]
        if overlaps:
            cap_present[i] = 1
            cap_subtype[i] = overlaps[0]["subtype"]
            total = sum(min(e["end_s"], end) - max(e["start_s"], start) for e in overlaps)
            cap_a_ratio[i] = total / epoch_length_s
    return cap_present, cap_subtype, cap_a_ratio
