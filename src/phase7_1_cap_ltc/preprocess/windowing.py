# src/phase7_1_cap_ltc/preprocess/windowing.py

"""
Phase 7.0b — Temporal Windowing (FINAL, A?-SAFE)
-----------------------------------------------
✔ Converts prepared tensors → temporal windows
✔ Variable channels per subject
✔ Supports overlap (0 / 50 / 75%)
✔ Preserves CAP uncertainty (A?)
✔ Propagates original masks
✔ Robust to short sequences
✔ Reviewer-safe metadata output
"""

from pathlib import Path
from typing import Tuple, Optional
import json
import logging

import numpy as np

from .. import config


# =============================================================================
# SLIDING WINDOWS
# =============================================================================
def sliding_windows(
    data: np.ndarray,
    mask: np.ndarray,
    window_size: int,
    overlap: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        data: (T, ...)
        mask: (T,)
    Returns:
        windows: (N, window_size, ...)
        win_mask: (N, window_size)
    """
    assert 0.0 <= overlap < 1.0

    step = max(int(window_size * (1 - overlap)), 1)
    T = data.shape[0]

    windows = []
    masks = []

    for start in range(0, T - window_size + 1, step):
        end = start + window_size
        windows.append(data[start:end])
        masks.append(mask[start:end])

    if not windows:
        empty_shape = (0, window_size) + data.shape[1:]
        return (
            np.empty(empty_shape, dtype=data.dtype),
            np.empty((0, window_size), dtype=bool),
        )

    return np.stack(windows), np.stack(masks)


# =============================================================================
# SUBJECT-LEVEL WINDOWING
# =============================================================================
def window_subject_npz(
    subject_dir: Path,
    out_dir: Path,
    overlap: float,
    window_size: int,
) -> None:
    """
    Input:
        subject_dir/features.npz
    Output:
        out_dir/windows.npz
    """
    npz_path = subject_dir / "features.npz"

    if not npz_path.exists():
        logging.warning(
            f"[Windowing] Missing features.npz → {subject_dir.name}"
        )
        return

    data = np.load(npz_path)

    features = data["features"]  # (T, C, F)
    cap = data["cap"]            # (T,)
    mask = data["mask"]          # (T,)

    T = features.shape[0]

    if T < window_size:
        logging.warning(
            f"[Windowing] Too short ({T} < {window_size}) → skipped: {subject_dir.name}"
        )
        return

    # ------------------------------
    # VALIDATE CAP STATES ONCE
    # ------------------------------
    valid_states = set(config.CAP_MAPPING.values())
    unique_states = set(np.unique(cap).tolist())

    if not unique_states.issubset(valid_states):
        raise ValueError(
            f"[Windowing] Invalid CAP states in {subject_dir.name}: {unique_states}"
        )

    # ------------------------------
    # WINDOW FEATURES
    # ------------------------------
    feat_windows, win_mask = sliding_windows(
        data=features,
        mask=mask,
        window_size=window_size,
        overlap=overlap,
    )

    # ------------------------------
    # WINDOW CAP LABELS
    # ------------------------------
    cap_windows, _ = sliding_windows(
        data=cap.reshape(-1, 1),
        mask=mask,
        window_size=window_size,
        overlap=overlap,
    )
    cap_windows = cap_windows.squeeze(-1)

    if feat_windows.shape[0] == 0:
        logging.warning(
            f"[Windowing] No valid windows → skipped: {subject_dir.name}"
        )
        return

    # ------------------------------
    # SAVE
    # ------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_dir / "windows.npz",
        features=feat_windows,
        cap=cap_windows,
        mask=win_mask,
    )

    logging.info(
        f"[Windowing] {subject_dir.name} → {feat_windows.shape[0]} windows"
    )


# =============================================================================
# BATCH DRIVER
# =============================================================================
def run_windowing(
    prepared_root: Path,
    output_root: Path,
    overlap: float,
    window_size: Optional[int] = None,
) -> None:
    """
    Convert all prepared subjects into windowed NPZs.
    """
    logging.basicConfig(level=logging.INFO)

    if window_size is None:
        window_size = config.WINDOW_SIZE

    if not prepared_root.exists():
        raise FileNotFoundError(
            f"Prepared dir missing: {prepared_root}"
        )

    overlap_dir = output_root / f"overlap_{int(overlap * 100)}"
    overlap_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata (reviewer-safe)
    with open(overlap_dir / "meta.json", "w") as f:
        json.dump(
            {
                "window_size": window_size,
                "overlap": overlap,
                "cap_states": config.CAP_MAPPING,
            },
            f,
            indent=4,
        )

    for subject_dir in sorted(prepared_root.iterdir()):
        if not subject_dir.is_dir():
            continue

        window_subject_npz(
            subject_dir=subject_dir,
            out_dir=overlap_dir / subject_dir.name,
            overlap=overlap,
            window_size=window_size,
        )


# =============================================================================
# STANDALONE
# =============================================================================
if __name__ == "__main__":
    run_windowing(
        prepared_root=config.PREPARED_DIR,
        output_root=config.DATA_DIR,
        overlap=0.75,
        window_size=config.WINDOW_SIZE,
    )
