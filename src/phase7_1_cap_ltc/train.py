"""
Phase 7.1 — CAP-LTC Training (FINAL, EARLY-STOPPING)
---------------------------------------------------
✔ Normality learning (controls only)
✔ tqdm progress bars
✔ Logging (console + file)
✔ Safe checkpointing (best only)
✔ Early stopping (reviewer-safe)
✔ τ extraction at end of training
"""

import time
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import config
from .utils import (
    set_global_seed,
    setup_logger,
    save_checkpoint,
    extract_tau_parameters,
)
from .dataset import TemporalEEGDataset, collate_fn
from .model.cap_ltc_model import CAP_LTC_Model


# =============================================================================
# TRAIN ONE EPOCH
# =============================================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> float:
    """
    Train model for one epoch.
    """
    model.train()
    epoch_loss = 0.0

    pbar = tqdm(
        loader,
        desc=f"[Train] Epoch {epoch}",
        ncols=100,
        leave=False,
    )

    for batch in pbar:
        features = batch["features"].to(device)   # (B, T, C, F)
        cap = batch["cap"].to(device)              # (B, T)
        mask = batch["mask"].to(device)            # (B, T)

        optimizer.zero_grad()

        # Forward
        recon, _ = model(features, cap, mask)

        # Masked reconstruction loss
        loss = criterion(
            recon[mask],
            features[mask],
        )

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.6f}")

    return epoch_loss / len(loader)


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================
def run_training() -> None:
    """
    Main Phase 7.1 training routine with early stopping.
    """

    # -------------------------------------------------------------------------
    # SETUP
    # -------------------------------------------------------------------------
    set_global_seed(config.SEED)
    logger = setup_logger("phase7_1_train")

    device = torch.device(config.DEVICE)
    logger.info(f"[Device] Using {device}")

    # -------------------------------------------------------------------------
    # DATASET
    # -------------------------------------------------------------------------
    dataset = TemporalEEGDataset(overlap=config.OVERLAP_PERCENT)

    if len(dataset) == 0:
        raise RuntimeError(
            "No training windows found. "
            "Check Phase 7.0 preparation + windowing."
        )

    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    logger.info(f"[Data] Training windows: {len(dataset)}")

    # -------------------------------------------------------------------------
    # MODEL
    # -------------------------------------------------------------------------
    model = CAP_LTC_Model().to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[Model] Trainable parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    criterion = nn.MSELoss()

    # -------------------------------------------------------------------------
    # TRAINING LOOP (EARLY STOPPING)
    # -------------------------------------------------------------------------
    best_loss = float("inf")
    patience = 5
    patience_counter = 0

    logger.info(
        f"[Training] Max epochs={config.MAX_EPOCHS}, "
        f"Early stopping patience={patience}"
    )

    for epoch in range(1, config.MAX_EPOCHS + 1):
        start = time.time()

        train_loss = train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
        )

        elapsed = time.time() - start

        logger.info(
            f"[Epoch {epoch:03d}] "
            f"Loss={train_loss:.6f} "
            f"Time={elapsed:.1f}s"
        )

        # -------------------------------------------------------------
        # CHECKPOINT + EARLY STOPPING
        # -------------------------------------------------------------
        if train_loss < best_loss - 1e-6:
            best_loss = train_loss
            patience_counter = 0

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=train_loss,
                filename=config.CHECKPOINT_NAME,
            )

            logger.info(
                f"[Checkpoint] New best loss={best_loss:.6f} "
                f"(epoch {epoch})"
            )
        else:
            patience_counter += 1
            logger.info(
                f"[EarlyStop] No improvement "
                f"({patience_counter}/{patience})"
            )

            if patience_counter >= patience:
                logger.info(
                    f"[EarlyStop] Triggered at epoch {epoch}. "
                    f"Stopping training."
                )
                break

    # -------------------------------------------------------------------------
    # τ EXTRACTION
    # -------------------------------------------------------------------------
    extract_tau_parameters(model, save=True)

    logger.info("=" * 80)
    logger.info("PHASE 7.1 COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)


# =============================================================================
# SCRIPT ENTRY
# =============================================================================
if __name__ == "__main__":
    run_training()
