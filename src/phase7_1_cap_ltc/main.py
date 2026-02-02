"""
Phase 7 — CAP-LTC Master Controller (FINAL)
-------------------------------------------
Controls:
✔ Phase 7.0a: Phase-2 CSV → tensor preparation
✔ Phase 7.0b: Temporal windowing
✔ Phase 7.1: Training (multi-overlap)
✔ Phase 7.2: τ Analysis
✔ Phase 7.3: Ablation
✔ Inference (multi-overlap)
✔ Visualizations
"""

import argparse
import logging
from datetime import datetime

from . import config

# Phase 7 modules
from .preprocess.prepare_from_phase2 import prepare_all_subjects
from .preprocess.windowing import run_windowing
from .train import run_training
from .analysis.tau_analysis import run_tau_analysis
from .ablation import run_ablation
from .infer import run_inference
from .visualize import run_all_visualizations


# =============================================================================
# ARGUMENTS
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser("CAP-LTC Phase 7 Controller")

    parser.add_argument("--prep", action="store_true")
    parser.add_argument("--window", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--tau", action="store_true")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--infer", action="store_true")
    parser.add_argument("--viz", action="store_true")
    parser.add_argument("--all", action="store_true")

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================
def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    start_time = datetime.now()

    print("\n" + "=" * 80)
    print("CAP-LTC PHASE 7 PIPELINE STARTED")
    print(f"Start Time : {start_time}")
    print("=" * 80)

    config.summary()

    # ======================================================================
    # FULL PIPELINE
    # ======================================================================
    if args.all:
        # ---------------------------
        # Phase 7.0a — PREPARATION
        # ---------------------------
        print("▶▶ Phase 7.0a — PREPARATION")
        prepare_all_subjects()
        print("✔ Preparation complete\n")

        # ---------------------------
        # Phase 7.0b — WINDOWING
        # ---------------------------
        print("▶▶ Phase 7.0b — WINDOWING")
        for overlap in config.OVERLAPS:
            run_windowing(
                prepared_root=config.PREPARED_DIR,
                output_root=config.DATA_DIR,
                overlap=overlap,
                window_size=config.WINDOW_SIZE,
            )
        print("✔ Windowing complete\n")

        # ---------------------------
        # Phase 7.1 → 7.3 — PER OVERLAP
        # ---------------------------
        for overlap in config.OVERLAPS:
            print(f"\n▶▶▶ RUNNING OVERLAP = {int(overlap * 100)}%")

            # Update global config safely
            config.OVERLAP_PERCENT = overlap

            # Training
            print("▶ Phase 7.1 — TRAINING")
            run_training()

            # τ analysis
            print("▶ Phase 7.2 — τ ANALYSIS")
            run_tau_analysis()

            # Ablation
            print("▶ Phase 7.3 — ABLATION")
            run_ablation()

            # Inference
            print("▶ INFERENCE")
            run_inference()

            print(f"✔ Overlap {int(overlap * 100)}% completed")

        # --------------------------
        # VISUALIZATION
        # --------------------------
        print("\n▶▶ VISUALIZATIONS")
        run_all_visualizations()
        print("✔ Visualizations completed")

    # ======================================================================
    # PARTIAL MODES
    # ======================================================================
    else:
        executed = False

        if args.prep:
            prepare_all_subjects()
            executed = True

        if args.window:
            for overlap in config.OVERLAPS:
                run_windowing(
                    prepared_root=config.PREPARED_DIR,
                    output_root=config.DATA_DIR,
                    overlap=overlap,
                    window_size=config.WINDOW_SIZE,
                )
            executed = True

        if args.train:
            run_training()
            executed = True

        if args.tau:
            run_tau_analysis()
            executed = True

        if args.ablation:
            run_ablation()
            executed = True

        if args.infer:
            run_inference()
            executed = True

        if args.viz:
            run_all_visualizations()
            executed = True

        if not executed:
            print("⚠️ No action selected. Use --all or individual flags.")

    print("\n" + "=" * 80)
    print("PIPELINE FINISHED")
    print(f"Total Time : {datetime.now() - start_time}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
