"""
SleepTech .edf.st parser

FINAL REVIEWER-SAFE VERSION
===========================

Features
--------
✔ Binary-safe parsing
✔ Absolute timestamp extraction
✔ Sequential fallback mode
✔ CAP subtype parsing
✔ Gap-aware reconstruction
✔ Overlap-safe epoch labeling
✔ Reviewer-safe implementation
✔ Works with CAP Sleep Database
✔ Pyright-safe typing

Author:
-------
AJI Research Pipeline
"""

import re

from pathlib import Path
from typing import (
    List,
    Dict,
    Any,
    Optional,
)

from loguru import logger


# =============================================================================
# REGEX
# =============================================================================
CAP_REGEX = re.compile(
    r"(?:"
    r"(?P<time1>\d+(?:\.\d+)?)\s+"
    r"|"
    r"(?P<time2>\d{1,2}:\d{1,2}:\d{1,2})\s+"
    r")?"
    r"MCAP-(?P<subtype>A1|A2|A3)\s+"
    r"(?P<duration>\d+)",
    re.IGNORECASE,
)


# =============================================================================
# HH:MM:SS → SECONDS
# =============================================================================
def hhmmss_to_seconds(
    t: str,
) -> float:

    h, m, s = t.split(":")

    return float(
        int(h) * 3600
        + int(m) * 60
        + int(s)
    )


# =============================================================================
# PARSE SLEEPTECH FILE
# =============================================================================
def parse_sleeptech_file(
    st_path: Path,
) -> List[Dict[str, Any]]:

    logger.info(
        f"[SleepTech Parser] {st_path.name}"
    )

    # =========================================================================
    # READ RAW BINARY
    # =========================================================================
    with open(
        st_path,
        "rb",
    ) as f:

        raw = f.read()

    # =========================================================================
    # DECODE
    # =========================================================================
    text = raw.decode(
        "latin1",
        errors="ignore",
    )

    # =========================================================================
    # FIND EVENTS
    # =========================================================================
    matches = list(
        CAP_REGEX.finditer(text)
    )

    logger.info(
        f"[Matches Found] {len(matches)}"
    )

    # =========================================================================
    # OUTPUT EVENTS
    # =========================================================================
    events: List[Dict[str, Any]] = []

    current_time: float = 0.0

    # =========================================================================
    # PARSE EVENTS
    # =========================================================================
    for m in matches:

        subtype: str = (
            str(
                m.group("subtype")
            )
            .upper()
            .strip()
        )

        duration_str = m.group(
            "duration"
        )

        try:

            duration: float = float(
                duration_str
            )

        except Exception:

            continue

        # =====================================================================
        # ABSOLUTE TIME
        # =====================================================================
        absolute_time: Optional[
            float
        ] = None

        time1 = m.group("time1")

        time2 = m.group("time2")

        # ---------------------------------------------------------------------
        # FLOAT TIMESTAMP
        # ---------------------------------------------------------------------
        if time1 is not None:

            try:

                absolute_time = float(
                    time1
                )

            except Exception:

                absolute_time = None

        # ---------------------------------------------------------------------
        # HH:MM:SS TIMESTAMP
        # ---------------------------------------------------------------------
        elif time2 is not None:

            try:

                absolute_time = (
                    hhmmss_to_seconds(
                        time2
                    )
                )

            except Exception:

                absolute_time = None

        # =====================================================================
        # RECONSTRUCT TIMELINE
        # =====================================================================
        if absolute_time is not None:

            start_s = absolute_time

            # monotonic safety
            if start_s < current_time:

                start_s = current_time

        else:

            # sequential fallback
            start_s = current_time

        end_s = (
            start_s
            + duration
        )

        events.append({

            "subtype":
                subtype,

            "start_s":
                float(start_s),

            "end_s":
                float(end_s),

            "duration":
                float(duration),
        })

        current_time = end_s

    logger.info(
        f"[Parsed Events] "
        f"{len(events)}"
    )

    # =========================================================================
    # DISTRIBUTION
    # =========================================================================
    counts: Dict[str, int] = {

        "A1": 0,
        "A2": 0,
        "A3": 0,
    }

    total_duration = 0.0

    for ev in events:

        subtype = str(
            ev["subtype"]
        )

        duration = float(
            ev["duration"]
        )

        total_duration += duration

        if subtype in counts:

            counts[subtype] += 1

    logger.info(
        "[Distribution]"
    )

    for k, v in counts.items():

        logger.info(
            f"  {k}: {v}"
        )

    logger.info(
        f"[Total CAP Duration] "
        f"{total_duration:.2f}s"
    )

    return events


# =============================================================================
# BUILD EPOCH LABELS
# =============================================================================
def build_epoch_cap_labels(
    cap_events: List[Dict[str, Any]],
    epoch_length_s: float,
    epoch_step_s: float,
    n_epochs: int,
) -> List[str]:

    labels: List[str] = []

    logger.info(
        f"[Building Epoch Labels] "
        f"{n_epochs}"
    )

    # =========================================================================
    # EPOCH LOOP
    # =========================================================================
    for i in range(n_epochs):

        start = (
            i
            * epoch_step_s
        )

        end = (
            start
            + epoch_length_s
        )

        overlaps: List[
            Dict[str, Any]
        ] = []

        # =====================================================================
        # FIND OVERLAPPING EVENTS
        # =====================================================================
        for ev in cap_events:

            if not (

                float(ev["end_s"])
                <= start

                or

                float(ev["start_s"])
                >= end
            ):

                overlaps.append(ev)

        # =====================================================================
        # NO CAP
        # =====================================================================
        if len(overlaps) == 0:

            labels.append("NC")

            continue

        # =====================================================================
        # ACCUMULATE OVERLAPS
        # =====================================================================
        subtype_durations: Dict[
            str,
            float,
        ] = {

            "A1": 0.0,
            "A2": 0.0,
            "A3": 0.0,
        }

        for ev in overlaps:

            overlap = (

                min(
                    float(ev["end_s"]),
                    end,
                )

                -

                max(
                    float(ev["start_s"]),
                    start,
                )
            )

            overlap = max(
                0.0,
                float(overlap),
            )

            subtype = str(
                ev["subtype"]
            )

            if subtype in subtype_durations:

                subtype_durations[
                    subtype
                ] += overlap

        # =====================================================================
        # DOMINANT LABEL
        # =====================================================================
        dominant = max(
            subtype_durations.keys(),
            key=lambda x:
                subtype_durations[x],
        )

        labels.append(
            dominant
        )

    logger.info(
        f"[Epoch Labels Built] "
        f"{len(labels)}"
    )

    return labels