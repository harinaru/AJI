"""
Validate corrected CAP distributions.
"""

from pathlib import Path
from collections import Counter

import numpy as np

from loguru import logger


LABELS = {
    0: "NC",
    1: "A1",
    2: "A2",
    3: "A3",
    4: "A?",
}


def validate_overlap(
    overlap_dir: Path,
):

    counter = Counter()

    for subject_dir in overlap_dir.iterdir():

        if not subject_dir.is_dir():
            continue

        npz_path = (
            subject_dir
            / "windows_corrected.npz"
        )

        if not npz_path.exists():
            continue

        data = np.load(
            npz_path,
            allow_pickle=True,
        )

        cap = data["cap"]

        unique, counts = np.unique(
            cap,
            return_counts=True,
        )

        for u, c in zip(unique, counts):

            counter[int(u)] += int(c)

    logger.info("=" * 70)

    logger.info(
        f"[Validation] {overlap_dir.name}"
    )

    total = sum(counter.values())

    for k in sorted(counter):

        pct = (
            100.0 * counter[k]
            / total
        )

        logger.info(
            f"{LABELS[k]}: "
            f"{counter[k]} "
            f"({pct:.2f}%)"
        )

    logger.info("=" * 70)


def run_validation():

    root = Path("data")

    overlaps = [
        root / "overlap_0",
        root / "overlap_50",
        root / "overlap_75",
    ]

    for ov in overlaps:

        if ov.exists():

            validate_overlap(ov)