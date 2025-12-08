#!/usr/bin/env python3
"""
Clean entry point. All heavy lifting lives under src/.

Usage:
  python main.py --phase 1
  python main.py --phase 2
"""

import os

# =========================================================================
# 🔒 CRITICAL: Limit BLAS / MKL / OMP threads BEFORE ANY NUMPY/SCIPY IMPORT
# =========================================================================
# These prevent DLL import failures like: "DLL load failed: paging file is too small"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

# =========================================================================
# Windows multiprocessing stability flags
# =========================================================================
# Prevent recursive import issues + excessive preloading during multiprocessing
os.environ.setdefault("PYTHONFAULTHANDLER", "1")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

# =========================================================================
# IMPORTS (after env fixes)
# =========================================================================
import sys
import argparse

# ensure src/ is importable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# imports from src
from core.pipeline_controller import run_phase1_from_main, run_phase2_from_main
from utils.config_loader import load_config
from utils.logger import get_logger

# Initialize logger (safe now)
logger = get_logger()

# =========================================================================
# argument parsing
# =========================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Hypnos-6: Main controller")
    p.add_argument("--phase", type=int, default=1, help="Phase to run (1=preproc, 2=features)")
    p.add_argument("--config", type=str, default=os.path.join("config", "global_config.yaml"))
    p.add_argument("--raw_dir", type=str, default=os.path.join("data", "raw"))
    p.add_argument("--out_dir", type=str, default=os.path.join("data", "processed"))
    p.add_argument("--subjects", type=str, default="", help="Comma-separated subject ids")
    p.add_argument("--workers", type=int, default=1, help="Parallel workers")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()

# =========================================================================
# main controller
# =========================================================================
def main():
    args = parse_args()
    logger = get_logger()
    cfg = load_config(args.config)

    logger.info(f"Main: start phase {args.phase} (raw={args.raw_dir}, out={args.out_dir})")

    # -------------------------------------------------------------
    # Phase 1 → automatic Phase 2
    # -------------------------------------------------------------
    if args.phase == 1:
        subjects = [s.strip() for s in args.subjects.split(",")] if args.subjects else None
        logger.info("Starting Phase 1: Preprocessing...")

        run_phase1_from_main(cfg,
                             raw_dir=args.raw_dir,
                             out_dir=args.out_dir,
                             subjects=subjects,
                             debug=args.debug)

        logger.info("✅ Phase 1 completed. Starting Phase 2 (Feature Extraction)...")

        processed_dir = args.out_dir
        results_dir = os.path.join("results", "phase2_features")

        run_phase2_from_main(cfg, processed_dir=processed_dir, results_dir=results_dir)

    # -------------------------------------------------------------
    # Phase 2 only
    # -------------------------------------------------------------
    elif args.phase == 2:
        processed_dir = args.out_dir
        results_dir = os.path.join("results", "phase2_features")

        logger.info("Starting Phase 2: Feature Extraction only...")
        run_phase2_from_main(cfg, processed_dir=processed_dir, results_dir=results_dir)

    else:
        logger.error("Only phase=1 and phase=2 are implemented.")
        raise SystemExit(1)


# =========================================================================
# Entry point
# =========================================================================
if __name__ == "__main__":
    main()
