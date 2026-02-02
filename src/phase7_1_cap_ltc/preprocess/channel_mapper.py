# preprocess/channel_mapper.py

"""
Phase 7.x — Channel Mapper
--------------------------
✔ Maps ch1...chN → true EEG channel names
✔ Handles subject-specific channel availability
✔ Missing channels handled safely
✔ Produces channel index + mask
✔ CRITICAL for Phase 7.1 stability
"""

from typing import Dict, List, Tuple
import logging


# ---------------------------------------------------------------------
# CANONICAL EEG CHANNEL ORDER (TARGET ORDER)
# ---------------------------------------------------------------------
CANONICAL_CHANNELS = [
    "Fp2-F4", "F4-C4", "C4-P4", "P4-O2",
    "Fp1-F3", "F3-C3", "C3-P3", "P3-O1",
    "F8-T4", "T4-T6", "F7-T3", "T3-T5",
    "ROC-LOC", "EMG1-EMG2",
]


# ---------------------------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------------------------
def map_channels_for_subject(
    available_channels: List[str],
    subject_id: str,
) -> Dict:
    """
    Maps subject-specific channels to canonical channel order.

    Args:
        available_channels:
            Example: ["Fp2-F4", "F4-C4", "C4-P4"]
        subject_id:
            Used only for logging

    Returns:
        dict with:
            - channel_names: ordered list aligned to canonical order
            - channel_mask: 1 if present, 0 if missing
            - channel_to_index: mapping for fast lookup
    """

    channel_names: List[str] = []
    channel_mask: List[int] = []
    channel_to_index: Dict[str, int] = {}

    for idx, ch in enumerate(CANONICAL_CHANNELS):
        if ch in available_channels:
            channel_names.append(ch)
            channel_mask.append(1)
            channel_to_index[ch] = idx
        else:
            channel_names.append("MISSING")
            channel_mask.append(0)

    n_present = sum(channel_mask)

    logging.info(
        f"[ChannelMapper] Subject={subject_id} | "
        f"Channels present={n_present}/{len(CANONICAL_CHANNELS)}"
    )

    if n_present < 4:
        logging.warning(
            f"[ChannelMapper] Subject={subject_id} has very few channels "
            f"({n_present}). Model robustness may degrade."
        )

    return {
        "channel_names": channel_names,
        "channel_mask": channel_mask,
        "channel_to_index": channel_to_index,
    }


# ---------------------------------------------------------------------
# CSV HEADER PARSER (ch1 → channel count)
# ---------------------------------------------------------------------
def infer_channel_count_from_columns(columns: List[str]) -> int:
    """
    Infer number of channels from CSV headers like ch1_mean, ch2_std...

    Args:
        columns: list of CSV column names

    Returns:
        int: number of detected channels
    """

    channel_ids = set()

    for col in columns:
        if col.startswith("ch"):
            try:
                idx = int(col.split("_")[0][2:])
                channel_ids.add(idx)
            except Exception:
                continue

    return max(channel_ids) if channel_ids else 0


# ---------------------------------------------------------------------
# FALLBACK CHANNEL GENERATOR
# ---------------------------------------------------------------------
def generate_placeholder_channels(n_channels: int) -> List[str]:
    """
    Used when true channel names are unknown.

    Example: ["ch1", "ch2", ..., "chN"]
    """

    return [f"ch{i}" for i in range(1, n_channels + 1)]
