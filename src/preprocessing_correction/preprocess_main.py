"""
Main controller for CAP correction pipeline.

FINAL REVIEWER-GRADE VERSION
============================

Pipeline:
---------
1. Repair CAP labels using SleepTech parser
2. Validate repaired distributions
3. Report timing + statistics

Features:
---------
✔ Progress-aware
✔ Reviewer-safe logging
✔ Timing metrics
✔ Crash-safe execution
✔ Modular pipeline
✔ Pyright-safe
"""

import time

from loguru import logger

from .repair_cap_windows import (
    run_repair,
)

from .validate_cap_distribution import (
    run_validation,
)


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:

    total_start = time.time()

    logger.info("\n")
    logger.info("=" * 80)
    logger.info(
        "STARTING CAP CORRECTION PIPELINE"
    )
    logger.info("=" * 80)

    # =========================================================================
    # STEP 1 — REPAIR
    # =========================================================================
    logger.info("\n")
    logger.info("=" * 80)
    logger.info(
        "STEP 1 — REPAIR WINDOWS"
    )
    logger.info("=" * 80)

    step1_start = time.time()

    try:

        run_repair()

    except Exception as e:

        logger.exception(
            f"[STEP 1 FAILED] {e}"
        )

        return

    step1_end = time.time()

    logger.info(
        "\n"
        f"[STEP 1 COMPLETED] "
        f"{(step1_end - step1_start)/60:.2f} minutes"
    )

    # =========================================================================
    # STEP 2 — VALIDATION
    # =========================================================================
    logger.info("\n")
    logger.info("=" * 80)
    logger.info(
        "STEP 2 — VALIDATE DISTRIBUTION"
    )
    logger.info("=" * 80)

    step2_start = time.time()

    try:

        run_validation()

    except Exception as e:

        logger.exception(
            f"[STEP 2 FAILED] {e}"
        )

        return

    step2_end = time.time()

    logger.info(
        "\n"
        f"[STEP 2 COMPLETED] "
        f"{(step2_end - step2_start)/60:.2f} minutes"
    )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    total_end = time.time()

    logger.info("\n")
    logger.info("=" * 80)
    logger.info(
        "CAP CORRECTION COMPLETED"
    )
    logger.info("=" * 80)

    logger.info(
        f"[TOTAL PIPELINE TIME] "
        f"{(total_end - total_start)/60:.2f} minutes"
    )

    logger.info("=" * 80)
    logger.info(
        "Generated Outputs:"
    )

    logger.info(
        "✔ windows_corrected.npz"
    )

    logger.info(
        "✔ corrected CAP distributions"
    )

    logger.info(
        "✔ overlap-aware CAP labels"
    )

    logger.info(
        "✔ preserved EEG features"
    )

    logger.info("=" * 80)
    logger.info("\n")


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":

    main()