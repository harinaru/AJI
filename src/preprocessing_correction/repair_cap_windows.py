"""
Repair CAP labels inside existing windows.npz
WITHOUT recomputing EEG features.

FINAL SLEEPTECH VERSION
=======================

✔ SleepTech sequential parser
✔ Correct CAP reconstruction
✔ Preserves EEG features
✔ Preserves masks
✔ Rewrites ONLY CAP labels
✔ Creates windows_corrected.npz
✔ Progress bars
✔ Before/After statistics
✔ Reviewer-safe
✔ Pyright-safe
"""

from pathlib import Path
from collections import Counter
from typing import Dict, List

import numpy as np

from tqdm import tqdm
from loguru import logger

from .sleeptech_parser import (
    parse_sleeptech_file,
    build_epoch_cap_labels,
)


# =============================================================================
# LABEL MAP
# =============================================================================
CAP_MAPPING: Dict[str, int] = {

    "NC": 0,

    "A1": 1,
    "A2": 2,
    "A3": 3,

    "A?": 4,
}


LABEL_NAMES: Dict[int, str] = {

    0: "NC",
    1: "A1",
    2: "A2",
    3: "A3",
    4: "A?",
}


# =============================================================================
# PROCESS ONE SUBJECT
# =============================================================================
def repair_subject(
    subject_dir: Path,
    annotation_dir: Path,
) -> None:

    subject_id = subject_dir.name

    logger.info("=" * 80)
    logger.info(f"[Subject] {subject_id}")

    # =========================================================================
    # ORIGINAL WINDOWS
    # =========================================================================
    windows_path = (
        subject_dir
        / "windows.npz"
    )

    if not windows_path.exists():

        logger.warning(
            f"[Missing windows.npz] {windows_path}"
        )

        return

    # =========================================================================
    # SLEEPTECH ST FILE
    # =========================================================================
    st_file = (
        annotation_dir
        / f"{subject_id}.edf.st"
    )

    if not st_file.exists():

        logger.warning(
            f"[Missing ST File] {st_file}"
        )

        return

    logger.info(
        f"[ST File] {st_file.name}"
    )

    # =========================================================================
    # LOAD WINDOWS
    # =========================================================================
    try:

        data = np.load(
            windows_path,
            allow_pickle=True,
        )

    except Exception as e:

        logger.exception(
            f"[NPZ Load Failed] {e}"
        )

        return

    # =========================================================================
    # REQUIRED KEYS
    # =========================================================================
    required_keys = [
        "features",
        "cap",
        "mask",
    ]

    for key in required_keys:

        if key not in data:

            logger.error(
                f"[Missing Key] {key}"
            )

            return

    # =========================================================================
    # PRESERVE ORIGINAL DATA
    # =========================================================================
    features = data["features"]

    old_cap = data["cap"]

    masks = data["mask"]

    # =========================================================================
    # SHAPES
    # =========================================================================
    n_windows = int(
        old_cap.shape[0]
    )

    T = int(
        old_cap.shape[1]
    )

    total_epochs = (
        n_windows * T
    )

    logger.info(
        f"[Windows] {n_windows}"
    )

    logger.info(
        f"[Window Length] {T}"
    )

    logger.info(
        f"[Total Epochs] {total_epochs}"
    )

    # =========================================================================
    # PARSE SLEEPTECH FILE
    # =========================================================================
    cap_events = parse_sleeptech_file(
        st_file
    )

    logger.info(
        f"[CAP Events Parsed] {len(cap_events)}"
    )

    if len(cap_events) == 0:

        logger.warning(
            "[WARNING] No CAP events found"
        )

    # =========================================================================
    # BUILD GLOBAL EPOCH LABELS
    # =========================================================================
    full_cap_subtype = build_epoch_cap_labels(
        cap_events=cap_events,

        # -------------------------------------------------------------
        # 30s epochs
        # 75% overlap => 5s stride
        # -------------------------------------------------------------
        epoch_length_s=30.0,
        epoch_step_s=5.0,

        n_epochs=total_epochs,
    )

    # =========================================================================
    # CAP PRESENCE VECTOR
    # =========================================================================
    cap_present = np.array(
        [
            0 if x == "NC" else 1
            for x in full_cap_subtype
        ],
        dtype=np.int64,
    )

    # =========================================================================
    # PLACEHOLDER CAP RATIO
    # =========================================================================
    cap_ratio = np.zeros(
        total_epochs,
        dtype=np.float32,
    )

    logger.info(
        f"[Epoch Labels Built] {len(full_cap_subtype)}"
    )

    # =========================================================================
    # OUTPUT ARRAY
    # =========================================================================
    corrected_cap = np.zeros_like(
        old_cap,
        dtype=np.int64,
    )

    # =========================================================================
    # COUNTERS
    # =========================================================================
    before_counter: Counter = Counter()

    after_counter: Counter = Counter()

    # =========================================================================
    # WINDOW REPAIR LOOP
    # =========================================================================
    logger.info(
        "[Repairing Windows]"
    )

    for w in tqdm(
        range(n_windows),
        desc=f"{subject_id}",
    ):

        # ---------------------------------------------------------------------
        # OLD DISTRIBUTION
        # ---------------------------------------------------------------------
        unique_old, counts_old = np.unique(
            old_cap[w],
            return_counts=True,
        )

        for u, c in zip(
            unique_old,
            counts_old,
        ):

            before_counter[
                int(u)
            ] += int(c)

        # ---------------------------------------------------------------------
        # GLOBAL SLICE
        # ---------------------------------------------------------------------
        start_epoch = w * T

        end_epoch = (
            start_epoch + T
        )

        window_subtypes = (
            full_cap_subtype[
                start_epoch:end_epoch
            ]
        )

        # ---------------------------------------------------------------------
        # SAFETY
        # ---------------------------------------------------------------------
        if len(window_subtypes) != T:

            logger.warning(
                f"[Window Length Mismatch] "
                f"window={w}"
            )

            padded = np.array(
                ["NC"] * T,
                dtype=object,
            )

            n_copy = min(
                T,
                len(window_subtypes),
            )

            padded[:n_copy] = (
                window_subtypes[:n_copy]
            )

            window_subtypes = padded

        # ---------------------------------------------------------------------
        # CONVERT LABELS
        # ---------------------------------------------------------------------
        numeric_cap = np.array(

            [
                CAP_MAPPING.get(
                    str(x).upper(),
                    4,
                )
                for x in window_subtypes
            ],

            dtype=np.int64,
        )

        corrected_cap[w] = numeric_cap

        # ---------------------------------------------------------------------
        # NEW DISTRIBUTION
        # ---------------------------------------------------------------------
        unique_new, counts_new = np.unique(
            numeric_cap,
            return_counts=True,
        )

        for u, c in zip(
            unique_new,
            counts_new,
        ):

            after_counter[
                int(u)
            ] += int(c)

    # =========================================================================
    # SAVE CORRECTED FILE
    # =========================================================================
    output_path = (
        subject_dir
        / "windows_corrected.npz"
    )

    logger.info(
        "[Saving windows_corrected.npz]"
    )

    np.savez_compressed(

        output_path,

        # -------------------------------------------------------------
        # PRESERVE FEATURES
        # -------------------------------------------------------------
        features=features,

        # -------------------------------------------------------------
        # REPAIRED CAP
        # -------------------------------------------------------------
        cap=corrected_cap,

        # -------------------------------------------------------------
        # PRESERVE MASK
        # -------------------------------------------------------------
        mask=masks,
    )

    logger.info(
        f"[Saved] {output_path}"
    )

    # =========================================================================
    # BEFORE
    # =========================================================================
    logger.info("-" * 60)
    logger.info("[BEFORE DISTRIBUTION]")

    total_before = sum(
        before_counter.values()
    )

    for k in sorted(
        before_counter.keys()
    ):

        v = before_counter[k]

        pct = (
            100.0 * v
            / max(total_before, 1)
        )

        logger.info(
            f"{LABEL_NAMES.get(k, str(k)):>4} "
            f": {v:>8} "
            f"({pct:6.2f}%)"
        )

    # =========================================================================
    # AFTER
    # =========================================================================
    logger.info("-" * 60)
    logger.info("[AFTER DISTRIBUTION]")

    total_after = sum(
        after_counter.values()
    )

    for k in sorted(
        after_counter.keys()
    ):

        v = after_counter[k]

        pct = (
            100.0 * v
            / max(total_after, 1)
        )

        logger.info(
            f"{LABEL_NAMES.get(k, str(k)):>4} "
            f": {v:>8} "
            f"({pct:6.2f}%)"
        )

    # =========================================================================
    # CHANGE SUMMARY
    # =========================================================================
    logger.info("-" * 60)
    logger.info("[CHANGE SUMMARY]")

    for k in sorted(
        LABEL_NAMES.keys()
    ):

        before = before_counter.get(
            k,
            0,
        )

        after = after_counter.get(
            k,
            0,
        )

        diff = after - before

        logger.info(
            f"{LABEL_NAMES[k]:>4} "
            f": {before:>8} "
            f"→ {after:>8} "
            f"(Δ={diff:+})"
        )

    logger.info("=" * 80)


# =============================================================================
# RUN ALL
# =============================================================================
def run_repair() -> None:

    root = Path("data")

    overlap_dirs: List[Path] = [

        root / "overlap_0",
        root / "overlap_50",
        root / "overlap_75",
    ]

    annotation_dir = (
        root / "annotations"
    )

    logger.info("=" * 80)
    logger.info(
        "STARTING SLEEPTECH CAP REPAIR"
    )
    logger.info("=" * 80)

    for overlap_dir in overlap_dirs:

        if not overlap_dir.exists():

            logger.warning(
                f"[Missing Overlap Dir] "
                f"{overlap_dir}"
            )

            continue

        logger.info("\n")
        logger.info("=" * 80)
        logger.info(
            f"[Processing] {overlap_dir.name}"
        )
        logger.info("=" * 80)

        subject_dirs = sorted(

            [
                p for p
                in overlap_dir.iterdir()
                if p.is_dir()
            ]
        )

        logger.info(
            f"[Subjects] {len(subject_dirs)}"
        )

        for subject_dir in tqdm(
            subject_dirs,
            desc=overlap_dir.name,
        ):

            try:

                repair_subject(
                    subject_dir=subject_dir,
                    annotation_dir=annotation_dir,
                )

            except Exception as e:

                logger.exception(
                    f"[FAILED] "
                    f"{subject_dir} | {e}"
                )

    logger.info("=" * 80)
    logger.info(
        "CAP REPAIR COMPLETED"
    )
    logger.info("=" * 80)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":

    run_repair()