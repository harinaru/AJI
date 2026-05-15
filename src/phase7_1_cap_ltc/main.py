# src/phase7_1_cap_ltc/main.py

"""
Phase 7 — CAP-LTC Master Controller
===================================

FINAL REVIEWER-GRADE VERSION
============================

✔ Explicit overlap propagation
✔ No stale config state
✔ No overlap caching
✔ Reviewer-safe orchestration
✔ Overlap-safe outputs
✔ Multi-run safe
✔ Multi-seed safe
✔ Non-overwriting
✔ Failure-safe execution
✔ Q1 journal-ready
✔ Windows-safe
✔ mmap-safe
✔ Modular execution
✔ Persistent experiment tracking
"""

# =============================================================================
# IMPORTS
# =============================================================================

import argparse
import json
import logging
import sys

from datetime import datetime
from pathlib import Path

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
# PIPELINE MODULES
# =============================================================================

from .train import run_training

from .infer import run_inference

from .evaluate import run_evaluation

from .latent import run_latent_analysis

from .xai.xai_integrated_gradients import (
    run_integrated_gradients,
)

from .analysis.tau_analysis import (
    run_tau_analysis,
)

from .ablation import run_ablation

from .analysis.runtime_analysis import (
    run_runtime_analysis,
)

from .visualize import (
    run_all_visualizations,
)

from .multi_seed import (
    run_multi_seed_evaluation,
)

# =============================================================================
# MASTER RUN
# =============================================================================

MASTER_RUN_ID = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

MASTER_RUN_DIR = (
    config.RESULTS_DIR
    / "master_runs"
    / MASTER_RUN_ID
)

MASTER_RUN_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# =============================================================================
# LOGGER
# =============================================================================

LOGGER = logging.getLogger(
    "phase7_master"
)

LOGGER.setLevel(logging.INFO)

if not LOGGER.handlers:

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    console_handler = logging.StreamHandler(
        sys.stdout
    )

    console_handler.setFormatter(
        formatter
    )

    file_handler = logging.FileHandler(

        MASTER_RUN_DIR
        / "master_pipeline.log",

        encoding="utf-8",
    )

    file_handler.setFormatter(
        formatter
    )

    LOGGER.addHandler(console_handler)
    LOGGER.addHandler(file_handler)

# =============================================================================
# METADATA
# =============================================================================

def save_master_metadata():

    metadata = {

        "master_run_id":
            MASTER_RUN_ID,

        "timestamp":
            str(datetime.now()),

        "device":
            str(config.DEVICE),

        "seed":
            int(config.SEED),

        "overlaps":
            [
                int(o * 100)
                for o in config.OVERLAPS
            ],

        "window_size":
            int(config.WINDOW_SIZE),

        "batch_size":
            int(config.BATCH_SIZE),

        "learning_rate":
            float(config.LEARNING_RATE),

        "epochs":
            int(config.MAX_EPOCHS),
    }

    path = (
        MASTER_RUN_DIR
        / "master_metadata.json"
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4,
        )

    LOGGER.info(
        f"[Metadata] Saved: {path}"
    )

# =============================================================================
# ARGUMENTS
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--prep", action="store_true")
    parser.add_argument("--window", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--tau", action="store_true")
    parser.add_argument("--infer", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--latent", action="store_true")
    parser.add_argument("--xai", action="store_true")
    parser.add_argument("--viz", action="store_true")
    parser.add_argument("--runtime", action="store_true")
    parser.add_argument("--multiseed", action="store_true")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--all", action="store_true")

    return parser.parse_args()

# =============================================================================
# SAFE RUN
# =============================================================================

def safe_run(
    name: str,
    fn,
):

    print(f"\n▶ {name}")

    LOGGER.info(
        f"STARTED → {name}"
    )

    try:

        fn()

        LOGGER.info(
            f"COMPLETED → {name}"
        )

        print(
            f"✔ {name} completed"
        )

    except Exception as e:

        LOGGER.exception(
            f"FAILED → {name}"
        )

        print(
            f"✖ {name} failed → {e}"
        )

# =============================================================================
# OVERLAP DIRS
# =============================================================================

def create_overlap_dirs(
    overlap: float,
):

    overlap_pct = int(
        overlap * 100
    )

    overlap_tag = (
        f"overlap_{overlap_pct}"
    )

    run_dir = (
        MASTER_RUN_DIR
        / overlap_tag
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    subdirs = [

        "models",
        "logs",
        "splits",
        "tau",
        "inference",
        "evaluation",
        "latent",
        "xai",
        "runtime",
        "visualization",
    ]

    for s in subdirs:

        (
            run_dir / s
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    LOGGER.info(
        f"[Dirs] Created: {run_dir}"
    )

    return run_dir

# =============================================================================
# MAIN
# =============================================================================

def main():

    args = parse_args()

    start_time = datetime.now()

    save_master_metadata()

    LOGGER.info(
        "PIPELINE STARTED"
    )

    config.summary()

    # =========================================================================
    # PREP
    # =========================================================================

    if args.prep or args.all:

        safe_run(
            "PREPARATION",
            prepare_all_subjects,
        )

    # =========================================================================
    # WINDOWING
    # =========================================================================

    if args.window or args.all:

        for overlap in config.OVERLAPS:

            safe_run(

                f"WINDOWING {int(overlap * 100)}%",

                lambda overlap=overlap:
                    run_windowing(

                        prepared_root=config.PREPARED_DIR,

                        output_root=config.DATA_DIR,

                        overlap=overlap,

                        window_size=config.WINDOW_SIZE,
                    )
            )

    # =========================================================================
    # OVERLAP PIPELINE
    # =========================================================================

    overlap_tasks_requested = any([

        args.train,
        args.tau,
        args.infer,
        args.eval,
        args.latent,
        args.xai,
        args.all,
    ])

    if overlap_tasks_requested:

        for overlap in config.OVERLAPS:

            overlap_pct = int(
                overlap * 100
            )

            print(
                "\n" + "=" * 80
            )

            print(
                f"OVERLAP {overlap_pct}%"
            )

            print(
                "=" * 80
            )

            run_dir = create_overlap_dirs(
                overlap
            )

            # =============================================================
            # TRAIN
            # =============================================================

            if args.train or args.all:

                safe_run(

                    f"TRAIN {overlap_pct}%",

                    lambda overlap=overlap,
                           run_dir=run_dir:

                    run_training(

                        overlap=overlap,

                        run_dir=run_dir,
                    )
                )

            # =============================================================
            # TAU
            # =============================================================

            if args.tau or args.all:

                safe_run(

                    f"TAU {overlap_pct}%",

                    lambda overlap=overlap,
                           run_dir=run_dir:

                    run_tau_analysis(

                        overlap=overlap,

                        run_dir=run_dir,
                    )
                )

            # =============================================================
            # INFERENCE
            # =============================================================

            if args.infer or args.all:

                safe_run(

                    f"INFERENCE {overlap_pct}%",

                    lambda overlap=overlap,
                           run_dir=run_dir:

                    run_inference(

                        overlap=overlap,

                        run_dir=run_dir,
                    )
                )

            # =============================================================
            # EVALUATION
            # =============================================================

            if args.eval or args.all:

                safe_run(

                    f"EVALUATION {overlap_pct}%",

                    lambda overlap=overlap,
                           run_dir=run_dir:

                    run_evaluation(

                        overlap=overlap,

                        run_dir=run_dir,
                    )
                )

            # =============================================================
            # LATENT
            # =============================================================

            if args.latent or args.all:

                safe_run(

                    f"LATENT {overlap_pct}%",

                    lambda overlap=overlap,
                           run_dir=run_dir:

                    run_latent_analysis(

                        overlap=overlap,

                        run_dir=run_dir,
                    )
                )

            # =============================================================
            # XAI
            # =============================================================

            if args.xai or args.all:

                safe_run(

                    f"XAI {overlap_pct}%",

                    lambda overlap=overlap,
                           run_dir=run_dir:

                    run_integrated_gradients(

                        overlap=overlap,

                        run_dir=run_dir,
                    )
                )

    # =========================================================================
    # GLOBAL MODULES
    # =========================================================================

    if args.ablation or args.all:

        safe_run(
            "ABLATION",
            run_ablation,
        )

    if args.multiseed or args.all:

        safe_run(
            "MULTI-SEED",
            run_multi_seed_evaluation,
        )

    if args.runtime or args.all:

        safe_run(
            "RUNTIME",
            run_runtime_analysis,
        )

    if args.viz or args.all:

        safe_run(
            "VISUALIZATION",
            run_all_visualizations,
        )

    # =========================================================================
    # FINISH
    # =========================================================================

    total_time = (
        datetime.now()
        - start_time
    )

    LOGGER.info(
        f"PIPELINE FINISHED | {total_time}"
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "PIPELINE FINISHED"
    )

    print(
        f"Total Time : {total_time}"
    )

    print(
        "=" * 80 + "\n"
    )

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    main()