# src/phase1_preprocessing/st_annotation_parser.py

import re
import json
import numpy as np

from collections import Counter

from typing import (
    Dict,
    List,
    Tuple,
    Any,
)

from numpy.typing import NDArray

from loguru import logger


# =============================================================================
# FLEXIBLE CAP REGEX
# =============================================================================
CAP_PATTERN = re.compile(
    r"""
    (?:CAP\s+)?                    # optional CAP prefix
    (A1|A2|A3|A\?|NC|NONCAP)      # subtype
    \s+
    (\d+\.?\d*)                   # start
    \s+
    (\d+\.?\d*)                   # end
    """,
    re.IGNORECASE | re.VERBOSE,
)


# =============================================================================
# NORMALIZE SUBTYPE
# =============================================================================
def normalize_subtype(
    label: str,
) -> str:

    label = label.upper().strip()

    if label in ["A1", "A2", "A3"]:
        return label

    if label in ["NC", "NONCAP"]:
        return "NC"

    return "A?"


# =============================================================================
# PARSE .ST FILE
# =============================================================================
def parse_st_file(
    st_path: str,
    output_json: str,
) -> List[Dict[str, Any]]:
    """
    Parse SleepTech .st CAP annotations.

    Supports:
        A1
        A2
        A3
        NC
        NONCAP
        A?

    Returns
    -------
    List[dict]
    """

    cap_events: List[Dict[str, Any]] = []

    subtype_counter = Counter()

    with open(
        st_path,
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as f:

        for line_idx, line in enumerate(f):

            line = line.strip()

            if len(line) == 0:
                continue

            match = CAP_PATTERN.search(line)

            if match is None:
                continue

            subtype_raw, start_s, end_s = match.groups()

            subtype = normalize_subtype(
                subtype_raw
            )

            try:

                start_val = float(start_s)

                end_val = float(end_s)

            except Exception:

                continue

            if end_val <= start_val:
                continue

            duration = (
                end_val
                -
                start_val
            )

            cap_events.append({

                "subtype":
                    subtype,

                "start_s":
                    start_val,

                "end_s":
                    end_val,

                "duration":
                    duration,
            })

            subtype_counter[subtype] += 1

    # =========================================================================
    # SAVE JSON
    # =========================================================================
    with open(
        output_json,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            cap_events,
            f,
            indent=2,
        )

    # =========================================================================
    # LOGGING
    # =========================================================================
    logger.info(
        f"[CAP Parser] Parsed "
        f"{len(cap_events)} events "
        f"from {st_path}"
    )

    logger.info(
        "[CAP Parser] Event Distribution:"
    )

    for k, v in subtype_counter.items():

        logger.info(
            f"  {k}: {v}"
        )

    logger.info(
        f"[CAP Parser] Saved → {output_json}"
    )

    return cap_events


# =============================================================================
# MAP CAPS TO EPOCHS
# =============================================================================
def map_caps_to_epochs(
    cap_events: List[Dict[str, Any]],
    epoch_length_s: float,
    epoch_step_s: float,
    n_epochs: int,
) -> Tuple[
    NDArray[np.int32],
    NDArray[np.object_],
    NDArray[np.float32],
]:
    """
    Convert CAP events → epoch labels.

    Returns
    -------
    cap_present : np.ndarray
    cap_subtype : np.ndarray
    cap_a_ratio : np.ndarray
    """

    # =========================================================================
    # OUTPUT ARRAYS
    # =========================================================================
    cap_present: NDArray[np.int32] = np.zeros(
        n_epochs,
        dtype=np.int32,
    )

    cap_subtype: NDArray[np.object_] = np.array(
        ["NC"] * n_epochs,
        dtype=object,
    )

    cap_a_ratio: NDArray[np.float32] = np.zeros(
        n_epochs,
        dtype=np.float32,
    )

    # =========================================================================
    # EPOCH LOOP
    # =========================================================================
    for i in range(n_epochs):

        start = i * epoch_step_s

        end = start + epoch_length_s

        overlaps: List[Dict[str, Any]] = []

        for e in cap_events:

            if not (
                e["end_s"] <= start
                or
                e["start_s"] >= end
            ):
                overlaps.append(e)

        # ---------------------------------------------------------------------
        # NO CAP EVENTS
        # ---------------------------------------------------------------------
        if len(overlaps) == 0:

            cap_present[i] = 0

            cap_subtype[i] = "NC"

            cap_a_ratio[i] = 0.0

            continue

        cap_present[i] = 1

        # ---------------------------------------------------------------------
        # OVERLAP ACCUMULATION
        # ---------------------------------------------------------------------
        subtype_overlap: Dict[str, float] = {

            "A1": 0.0,
            "A2": 0.0,
            "A3": 0.0,
            "NC": 0.0,
            "A?": 0.0,
        }

        total_overlap = 0.0

        for e in overlaps:

            overlap = (
                min(
                    float(e["end_s"]),
                    end,
                )
                -
                max(
                    float(e["start_s"]),
                    start,
                )
            )

            overlap = max(
                0.0,
                float(overlap),
            )

            subtype = str(
                e["subtype"]
            )

            if subtype not in subtype_overlap:

                subtype = "A?"

            subtype_overlap[subtype] += overlap

            total_overlap += overlap

        # ---------------------------------------------------------------------
        # DOMINANT SUBTYPE
        # ---------------------------------------------------------------------
        dominant = sorted(
            subtype_overlap.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )[0][0]

        cap_subtype[i] = dominant

        # ---------------------------------------------------------------------
        # CAP RATIO
        # ---------------------------------------------------------------------
        cap_a_ratio[i] = np.float32(
            total_overlap
            /
            max(epoch_length_s, 1e-8)
        )

    # =========================================================================
    # FINAL LOGGING
    # =========================================================================
    unique, counts = np.unique(
        cap_subtype,
        return_counts=True,
    )

    logger.info(
        "[Epoch CAP Distribution]"
    )

    for u, c in zip(unique, counts):

        logger.info(
            f"  {u}: {c}"
        )

    return (
        cap_present,
        cap_subtype,
        cap_a_ratio,
    )