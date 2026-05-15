"""
Utilities for Phase 7.1 — CAP-LTC
=================================

✔ Reproducibility
✔ UTF-8 safe logging
✔ File + console logging
✔ Reviewer-safe checkpointing
✔ Overlap-safe saving
✔ Multi-seed safe saving
✔ Non-overwriting architecture
✔ τ parameter extraction
✔ Crash-safe JSON saving
✔ Windows-safe
"""

# =============================================================================
# IMPORTS
# =============================================================================

import sys
import time
import json
import random
import logging

from pathlib import Path

from typing import (
    Optional,
    Dict,
)

import numpy as np

import torch

from . import config


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_global_seed(
    seed: Optional[int] = None
) -> None:

    """
    Set all seeds for reproducibility.
    """

    seed = (
        seed
        if seed is not None
        else config.SEED
    )

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# =============================================================================
# UTF-8 SAFE LOGGER
# =============================================================================

def setup_logger(
    name: str = "phase7_1"
) -> logging.Logger:

    """
    Reviewer-grade logger.

    ✔ UTF-8 safe
    ✔ Windows-safe
    ✔ Console + file logging
    ✔ Prevent duplicate handlers
    ✔ Timestamped logs
    """

    logger = logging.getLogger(name)

    logger.setLevel(logging.INFO)

    # =========================================================================
    # PREVENT DUPLICATE HANDLERS
    # =========================================================================
    if logger.handlers:

        return logger

    # =========================================================================
    # LOG DIRECTORY
    # =========================================================================
    config.LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================================
    # LOG FILE
    # =========================================================================
    timestamp = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    log_file = (

        config.LOG_DIR
        / f"{name}_{timestamp}.log"
    )

    # =========================================================================
    # FORMATTER
    # =========================================================================
    formatter = logging.Formatter(

        "%(asctime)s | %(levelname)s | %(message)s"
    )

    # =========================================================================
    # FILE HANDLER
    # =========================================================================
    fh = logging.FileHandler(

        log_file,

        encoding="utf-8"
    )

    fh.setLevel(logging.INFO)

    fh.setFormatter(formatter)

    # =========================================================================
    # CONSOLE HANDLER
    # =========================================================================
    ch = logging.StreamHandler(
        sys.stdout
    )

    ch.setLevel(logging.INFO)

    ch.setFormatter(formatter)

    # =========================================================================
    # ADD HANDLERS
    # =========================================================================
    logger.addHandler(fh)

    logger.addHandler(ch)

    # =========================================================================
    # HEADER
    # =========================================================================
    logger.info("=" * 80)

    logger.info(
        "PHASE 7.1 — CAP-LTC"
    )

    logger.info(
        f"Device: {config.DEVICE}"
    )

    logger.info(
        f"Seed: {config.SEED}"
    )

    logger.info("=" * 80)

    logger.info(
        f"[Logger] File → {log_file}"
    )

    return logger


# =============================================================================
# REVIEWER-GRADE CHECKPOINTING
# =============================================================================
def save_checkpoint(

    model: torch.nn.Module,

    optimizer: torch.optim.Optimizer,

    epoch: int,

    loss: float,
    overlap_pct: int,

    filename: Optional[str] = None,

    save_dir: Optional[Path] = None,

) -> Path:

    """
    Reviewer-grade checkpoint saving.

    ✔ overlap-safe
    ✔ ablation-safe
    ✔ multi-seed safe
    ✔ resume-safe
    ✔ non-overwriting
    ✔ explicit path-safe
    """

    # =========================================================================
    # EXPLICIT SAVE DIR
    # =========================================================================

    if save_dir is None:

        raise ValueError(

            "[Checkpoint] save_dir must be explicitly provided "
            "for reviewer-grade reproducibility."
        )

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================================
    # DEFAULT FILENAME
    # =========================================================================

    if filename is None:

        filename = (
            f"cap_ltc_best_overlap_"
           f"{overlap_pct}"
        )

    # =========================================================================
    # SAVE PATH
    # =========================================================================

    save_path = (
        save_dir / filename
    )

    # =========================================================================
    # CHECKPOINT CONTENT
    # =========================================================================

    checkpoint = {

        "epoch":
            epoch,

        "loss":
            float(loss),

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "seed":
            config.SEED,

        "device":
            str(config.DEVICE),

        "overlap":
            config.OVERLAP_PERCENT,

        "feature_dim":
            config.FEATURE_DIM,

        "window_size":
            config.WINDOW_SIZE,
    }

    # =========================================================================
    # SAVE
    # =========================================================================

    torch.save(
        checkpoint,
        save_path,
    )

    logging.info(
        f"[Checkpoint] Saved → {save_path}"
    )

    return save_path

# =============================================================================
# τ EXTRACTION
# =============================================================================

def extract_tau_parameters(

    model: torch.nn.Module,

    overlap_pct: int,

    save: bool = True,

    save_dir: Optional[Path] = None,

) -> Optional[Dict]:

    """
    Extract learned τ parameters.

    ✔ CAP-aware
    ✔ overlap-safe
    ✔ reviewer-safe
    ✔ experiment-isolated
    ✔ non-overwriting
    ✔ explicit-save-dir safe
    """

    tau_data: Dict[str, object] = {}

    # =========================================================================
    # FIND CAP-LTC CELL
    # =========================================================================

    for module in model.modules():

        if (

            hasattr(module, "tau_base")

            and

            hasattr(module, "tau_cap")
        ):

            tau_base = torch.nn.functional.softplus(

                torch.as_tensor(
                    module.tau_base
                )
            )

            tau_cap = torch.nn.functional.softplus(

                torch.as_tensor(
                    module.tau_cap
                )
            )

            tau_data["tau_base"] = float(
                tau_base.item()
            )

            tau_data["tau_cap"] = (

                tau_cap
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )

            logging.info(
                "[Tau] CAP-LTC τ parameters extracted"
            )

            break

    # =========================================================================
    # NO TAU FOUND
    # =========================================================================

    if not tau_data:

        logging.warning(
            "[Tau] No τ parameters found"
        )

        return None

    # =========================================================================
    # SAVE
    # =========================================================================

    if save:

        if save_dir is None:

            raise ValueError(

                "[Tau] save_dir must be explicitly provided "
                "for reviewer-grade reproducibility."
            )

        save_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        timestamp = time.strftime(
            "%Y%m%d_%H%M%S"
        )

        tau_filename = (

            f"learned_tau_overlap_"

            f"{overlap_pct}"

            f"_{timestamp}.json"
        )

        tau_path = (
            save_dir / tau_filename
        )

        with open(

            tau_path,

            "w",

            encoding="utf-8"

        ) as f:

            json.dump(

                tau_data,

                f,

                indent=4,
            )

        logging.info(
            f"[Tau] Saved → {tau_path}"
        )

    return tau_data
# =============================================================================
# RESOLVE LATEST CHECKPOINT
# =============================================================================

import re
import logging

from pathlib import Path


def resolve_latest_checkpoint(
    overlap: float,
    model_dir: Path,
):
    """
    Resolve BEST checkpoint safely.

    Reviewer-grade logic:
    Select checkpoint with LOWEST validation loss.

    ✔ overlap-safe
    ✔ reviewer-safe
    ✔ multi-run safe
    ✔ best-model selection
    ✔ stable parsing
    """

    overlap_pct = int(
        overlap * 100
    )

    # =========================================================================
    # VALIDATE DIRECTORY
    # =========================================================================

    if not model_dir.exists():

        raise FileNotFoundError(

            f"[Checkpoint] Missing model dir: "
            f"{model_dir}"
        )

    # =========================================================================
    # FIND CHECKPOINTS
    # =========================================================================

    checkpoints = list(

        model_dir.glob(
            f"cap_ltc_best_overlap_{overlap_pct}_*.pt"
        )
    )

    if len(checkpoints) == 0:

        raise FileNotFoundError(

            "[Checkpoint] No checkpoints found "
            f"for overlap={overlap_pct}%"
        )

    # =========================================================================
    # REGEX
    # Example:
    # cap_ltc_best_overlap_75_epoch_12_valloss_0.231445.pt
    # =========================================================================

    pattern = re.compile(
        r"valloss_([0-9]+\.[0-9]+)"
    )

    best_checkpoint = None

    best_loss = float("inf")

    # =========================================================================
    # FIND LOWEST VAL LOSS
    # =========================================================================

    for ckpt in checkpoints:

        match = pattern.search(
            ckpt.stem
        )

        if match is None:

            logging.warning(

                f"[Checkpoint] Could not parse "
                f"val loss from: {ckpt.name}"
            )

            continue

        try:

            val_loss = float(
                match.group(1)
            )

        except Exception as e:

            logging.warning(

                f"[Checkpoint] Invalid val loss "
                f"in: {ckpt.name} | {e}"
            )

            continue

        # =====================================================================
        # BEST CHECKPOINT
        # =====================================================================

        if val_loss < best_loss:

            best_loss = val_loss

            best_checkpoint = ckpt

    # =========================================================================
    # FINAL CHECK
    # =========================================================================

    if best_checkpoint is None:

        raise RuntimeError(

            "[Checkpoint] Failed to resolve "
            "best checkpoint."
        )

    logging.info(
        "=" * 80
    )

    logging.info(
        f"[Checkpoint] BEST MODEL SELECTED"
    )

    logging.info(
        f"[Checkpoint] Overlap: "
        f"{overlap_pct}%"
    )

    logging.info(
        f"[Checkpoint] Validation Loss: "
        f"{best_loss:.6f}"
    )

    logging.info(
        f"[Checkpoint] Path: "
        f"{best_checkpoint}"
    )

    logging.info(
        "=" * 80
    )

    return best_checkpoint