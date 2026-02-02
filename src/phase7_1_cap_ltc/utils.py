"""
Utilities for Phase 7.1 — CAP-LTC
--------------------------------
• Reproducibility
• Logging (file + console)
• Checkpoint saving
• τ parameter extraction (CRITICAL: E1.1 / E6.1)
"""

import sys
import time
import json
import logging
import random
from typing import Optional, Dict

import numpy as np
import torch

from . import config


# =============================================================================
# REPRODUCIBILITY
# =============================================================================
def set_global_seed(seed: Optional[int] = None) -> None:
    """
    Set all random seeds for reproducibility.
    """
    seed = seed if seed is not None else config.SEED

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# LOGGING
# =============================================================================
def setup_logger(name: str = "phase7_1") -> logging.Logger:
    """
    Setup logger with file + console handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)   # ✅ FIX: no config.LOG_LEVEL

    if logger.handlers:
        return logger  # Prevent duplicate handlers

    log_file = config.LOG_DIR / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.log"

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info("=" * 80)
    logger.info("PHASE 7.1 — CAP-LTC")
    logger.info(f"Device: {config.DEVICE}")
    logger.info("=" * 80)

    return logger


# =============================================================================
# CHECKPOINTING
# =============================================================================
def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filename: Optional[str] = None,
) -> None:
    """
    Save training checkpoint safely.
    """
    filename = filename if filename is not None else config.CHECKPOINT_NAME
    path = config.MODEL_DIR / filename

    torch.save(
        {
            "epoch": epoch,
            "loss": loss,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )

    logging.info(f"[Checkpoint] Saved → {path}")


# =============================================================================
# τ EXTRACTION — CRITICAL (E1.1 / E6.1)
# =============================================================================
def extract_tau_parameters(
    model: torch.nn.Module,
    save: bool = True,
) -> Optional[Dict]:
    """
    Extract learned τ parameters from CAP-LTC cell.
    """

    tau_data: Dict[str, object] = {}

    for module in model.modules():
        if hasattr(module, "tau_base") and hasattr(module, "tau_cap"):

            tau_base = torch.nn.functional.softplus(
                torch.as_tensor(module.tau_base)
            )

            tau_cap = torch.nn.functional.softplus(
                torch.as_tensor(module.tau_cap)
            )

            tau_data["tau_base"] = float(tau_base.item())
            tau_data["tau_cap"] = tau_cap.detach().cpu().numpy().tolist()

            logging.info("[Tau] CAP-LTC τ parameters extracted")
            break

    if not tau_data:
        logging.warning("[Tau] No CAP-LTC τ parameters found")
        return None

    if save:
        config.TAU_DIR.mkdir(parents=True, exist_ok=True)
        path = config.TAU_DIR / "learned_tau.json"

        with open(path, "w") as f:
            json.dump(tau_data, f, indent=4)

        logging.info(f"[Tau] Saved learned τ → {path}")

    return tau_data
