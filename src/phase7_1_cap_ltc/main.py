"""
Phase 7 — CAP-LTC Master Controller (FINAL, Q1 JOURNAL-GRADE)
--------------------------------------------------------------
Controls:
✔ Phase 7.0a: Phase-2 CSV → tensor preparation
✔ Phase 7.0b: Temporal windowing
✔ Phase 7.1: Training (multi-overlap)
✔ Phase 7.2: τ Analysis
✔ Phase 7.3: Ablation
✔ Inference (multi-overlap)
✔ Reviewer-grade evaluation
✔ Journal-grade visualizations
✔ Latent embedding extraction
✔ t-SNE / UMAP analysis
✔ Integrated Gradients XAI
✔ Multi-seed robustness evaluation
✔ Runtime benchmarking
✔ Reviewer-safe logging
✔ Failure-safe execution
"""

import argparse
import logging

from datetime import datetime

from . import config

# =============================================================================
# PREPROCESS
# =============================================================================
from .preprocess.prepare_from_phase2 import (
    prepare_all_subjects,
)

from .preprocess.windowing import (
    run_windowing,
)

# =============================================================================
# TRAINING
# =============================================================================
from .train import run_training

# =============================================================================
# TAU ANALYSIS
# =============================================================================
from .analysis.tau_analysis import (
    run_tau_analysis,
)

# =============================================================================
# ABLATION
# =============================================================================
from .ablation import run_ablation

# =============================================================================
# INFERENCE
# =============================================================================
from .infer import run_inference

# =============================================================================
# EVALUATION
# =============================================================================
from .evaluate import run_evaluation

# =============================================================================
# VISUALIZATION
# =============================================================================
from .visualize import run_all_visualizations

# =============================================================================
# ADVANCED ANALYSIS
# =============================================================================
from .latent import run_latent_analysis

from .xai.xai_integrated_gradients import (
    run_integrated_gradients,
)

from .analysis.runtime_analysis import (
    run_runtime_analysis,
)

from .multi_seed import (
    run_multi_seed_evaluation,
)


# =============================================================================
# ARGUMENTS
# =============================================================================
def parse_args():

    parser = argparse.ArgumentParser(
        "CAP-LTC Phase 7 Controller"
    )

    # -------------------------------------------------------------------------
    # CORE
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--prep",
        action="store_true",
    )

    parser.add_argument(
        "--window",
        action="store_true",
    )

    parser.add_argument(
        "--train",
        action="store_true",
    )

    parser.add_argument(
        "--tau",
        action="store_true",
    )

    parser.add_argument(
        "--ablation",
        action="store_true",
    )

    parser.add_argument(
        "--infer",
        action="store_true",
    )

    parser.add_argument(
        "--eval",
        action="store_true",
    )

    parser.add_argument(
        "--viz",
        action="store_true",
    )

    # -------------------------------------------------------------------------
    # ADVANCED
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--latent",
        action="store_true",
    )

    parser.add_argument(
        "--xai",
        action="store_true",
    )

    parser.add_argument(
        "--runtime",
        action="store_true",
    )

    parser.add_argument(
        "--multiseed",
        action="store_true",
    )

    # -------------------------------------------------------------------------
    # FULL PIPELINE
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--all",
        action="store_true",
    )

    return parser.parse_args()


# =============================================================================
# SAFE EXECUTION WRAPPER
# =============================================================================
def safe_run(
    name: str,
    fn,
):

    print(f"\n▶ {name}")

    try:

        fn()

        print(f"✔ {name} completed")

    except Exception as e:

        logging.exception(
            f"[ERROR] {name} failed"
        )

        print(
            f"✖ {name} failed → {e}"
        )


# =============================================================================
# OVERLAP LOOP
# =============================================================================
def run_overlap_pipeline(
    overlap: float
):

    overlap_pct = int(
        overlap * 100
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"RUNNING OVERLAP = "
        f"{overlap_pct}%"
    )

    print(
        "=" * 70
    )

    # -------------------------------------------------------------------------
    # IMPORTANT:
    # SET GLOBAL OVERLAP
    # -------------------------------------------------------------------------
    config.OVERLAP_PERCENT = overlap

    # -------------------------------------------------------------------------
    # TRAINING
    # -------------------------------------------------------------------------
    safe_run(
        "Phase 7.1 — TRAINING",
        run_training,
    )

    # -------------------------------------------------------------------------
    # TAU ANALYSIS
    # -------------------------------------------------------------------------
    safe_run(
        "Phase 7.2 — τ ANALYSIS",
        run_tau_analysis,
    )

    # -------------------------------------------------------------------------
    # INFERENCE
    # -------------------------------------------------------------------------
    safe_run(
        "INFERENCE",
        run_inference,
    )

    # -------------------------------------------------------------------------
    # EVALUATION
    # -------------------------------------------------------------------------
    safe_run(
        "EVALUATION",
        run_evaluation,
    )

    # -------------------------------------------------------------------------
    # LATENT EXTRACTION
    # -------------------------------------------------------------------------
    safe_run(
        "LATENT EXTRACTION",
        run_latent_analysis,
    )

    # -------------------------------------------------------------------------
    # INTEGRATED GRADIENTS
    # -------------------------------------------------------------------------
    safe_run(
        "INTEGRATED GRADIENTS",
        run_integrated_gradients,
    )

    print(
        f"\n✔ OVERLAP "
        f"{overlap_pct}% COMPLETE"
    )


# =============================================================================
# MAIN
# =============================================================================
def main():

    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    start_time = datetime.now()

    print(
        "\n"
        + "=" * 80
    )

    print(
        "CAP-LTC PHASE 7 "
        "PIPELINE STARTED"
    )

    print(
        f"Start Time : "
        f"{start_time}"
    )

    print("=" * 80)

    # -------------------------------------------------------------------------
    # CONFIG SUMMARY
    # -------------------------------------------------------------------------
    config.summary()

    # =========================================================================
    # FULL PIPELINE
    # =========================================================================
    if args.all:

        # ---------------------------------------------------------------------
        # PREPARATION
        # ---------------------------------------------------------------------
        safe_run(
            "Phase 7.0a — PREPARATION",
            prepare_all_subjects,
        )

        # ---------------------------------------------------------------------
        # WINDOWING
        # ---------------------------------------------------------------------
        print(
            "\n▶▶ "
            "Phase 7.0b — WINDOWING"
        )

        for overlap in config.OVERLAPS:

            try:

                run_windowing(
                    prepared_root=config.PREPARED_DIR,
                    output_root=config.DATA_DIR,
                    overlap=overlap,
                    window_size=config.WINDOW_SIZE,
                )

                print(
                    f"✔ Windowing "
                    f"{int(overlap*100)}%"
                )

            except Exception as e:

                logging.exception(
                    "[ERROR] Windowing failed"
                )

                print(
                    f"✖ Windowing failed → {e}"
                )

        # ---------------------------------------------------------------------
        # MULTI-OVERLAP PIPELINE
        # ---------------------------------------------------------------------
        for overlap in config.OVERLAPS:

            run_overlap_pipeline(
                overlap
            )

        # ---------------------------------------------------------------------
        # ABLATION
        # ---------------------------------------------------------------------
        safe_run(
            "Phase 7.3 — ABLATION",
            run_ablation,
        )

        # ---------------------------------------------------------------------
        # MULTI-SEED
        # ---------------------------------------------------------------------
        safe_run(
            "MULTI-SEED EVALUATION",
            run_multi_seed_evaluation,
        )

        # ---------------------------------------------------------------------
        # RUNTIME ANALYSIS
        # ---------------------------------------------------------------------
        safe_run(
            "RUNTIME ANALYSIS",
            run_runtime_analysis,
        )

        # ---------------------------------------------------------------------
        # VISUALIZATION
        # ---------------------------------------------------------------------
        safe_run(
            "VISUALIZATIONS",
            run_all_visualizations,
        )

    # =========================================================================
    # PARTIAL MODES
    # =========================================================================
    else:

        executed = False

        # ---------------------------------------------------------------------
        # PREPARATION
        # ---------------------------------------------------------------------
        if args.prep:

            safe_run(
                "PREPARATION",
                prepare_all_subjects,
            )

            executed = True

        # ---------------------------------------------------------------------
        # WINDOWING
        # ---------------------------------------------------------------------
        if args.window:

            print("\n▶ WINDOWING")

            for overlap in config.OVERLAPS:

                try:

                    run_windowing(
                        prepared_root=config.PREPARED_DIR,
                        output_root=config.DATA_DIR,
                        overlap=overlap,
                        window_size=config.WINDOW_SIZE,
                    )

                    print(
                        f"✔ Windowing "
                        f"{int(overlap*100)}%"
                    )

                except Exception as e:

                    print(
                        f"✖ Windowing failed → {e}"
                    )

            executed = True

        # ---------------------------------------------------------------------
        # TRAINING
        # ---------------------------------------------------------------------
        if args.train:

            safe_run(
                "TRAINING",
                run_training,
            )

            executed = True

        # ---------------------------------------------------------------------
        # TAU ANALYSIS
        # ---------------------------------------------------------------------
        if args.tau:

            safe_run(
                "τ ANALYSIS",
                run_tau_analysis,
            )

            executed = True

        # ---------------------------------------------------------------------
        # ABLATION
        # ---------------------------------------------------------------------
        if args.ablation:

            safe_run(
                "ABLATION",
                run_ablation,
            )

            executed = True

        # ---------------------------------------------------------------------
        # INFERENCE
        # ---------------------------------------------------------------------
        if args.infer:

            safe_run(
                "INFERENCE",
                run_inference,
            )

            executed = True

        # ---------------------------------------------------------------------
        # EVALUATION
        # ---------------------------------------------------------------------
        if args.eval:

            safe_run(
                "EVALUATION",
                run_evaluation,
            )

            executed = True

        # ---------------------------------------------------------------------
        # VISUALIZATION
        # ---------------------------------------------------------------------
        if args.viz:

            safe_run(
                "VISUALIZATION",
                run_all_visualizations,
            )

            executed = True

        # ---------------------------------------------------------------------
        # LATENT
        # ---------------------------------------------------------------------
        if args.latent:

            safe_run(
                "LATENT EXTRACTION",
                run_latent_analysis,
            )

            executed = True

        # ---------------------------------------------------------------------
        # XAI
        # ---------------------------------------------------------------------
        if args.xai:

            safe_run(
                "INTEGRATED GRADIENTS",
                run_integrated_gradients,
            )

            executed = True

        # ---------------------------------------------------------------------
        # RUNTIME
        # ---------------------------------------------------------------------
        if args.runtime:

            safe_run(
                "RUNTIME ANALYSIS",
                run_runtime_analysis,
            )

            executed = True

        # ---------------------------------------------------------------------
        # MULTI-SEED
        # ---------------------------------------------------------------------
        if args.multiseed:

            safe_run(
                "MULTI-SEED EVALUATION",
                run_multi_seed_evaluation,
            )

            executed = True

        # ---------------------------------------------------------------------
        # NOTHING SELECTED
        # ---------------------------------------------------------------------
        if not executed:

            print(
                "\n⚠️ No action selected.\n"
                "Use --all or individual flags."
            )

    # =========================================================================
    # END
    # =========================================================================
    total_time = (
        datetime.now()
        - start_time
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "PIPELINE FINISHED"
    )

    print(
        f"Total Time : "
        f"{total_time}"
    )

    print("=" * 80 + "\n")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":

    main()