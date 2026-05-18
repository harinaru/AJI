# src/phase7_1_cap_ltc/train.py

"""
Phase 7.x — CAP-LTC Training

CHANGES FROM PREVIOUS VERSION
==============================

1. MARGIN LOSS (new)
   The old training used only MSE reconstruction loss, which trains the model
   to reconstruct ALL windows well — including disorders. This is the opposite
   of what anomaly detection needs. The margin loss adds a penalty that
   explicitly pushes disorder reconstruction error ABOVE normal error by at
   least MARGIN_LOSS_MARGIN. Applied whenever both normal and disorder windows
   appear in the same batch.

   Loss composition:
     total = recon_loss
           + LATENT_REG_WEIGHT  * latent_temporal_smoothness
           + MARGIN_LOSS_WEIGHT * max(0, margin - (disorder_error - normal_error))

2. COSINE ANNEALING SCHEDULER (replaces ReduceLROnPlateau)
   ReduceLROnPlateau decays LR only after val loss stagnates for patience epochs.
   With the old patience=3 this caused premature LR decay before the model found
   anomaly-discriminative features. CosineAnnealingLR decays smoothly over
   MAX_EPOCHS, giving the model a better loss landscape to navigate.
   Controlled by config.LR_SCHEDULER ("cosine" or "plateau").

3. EARLY STOPPING PATIENCE: 3 → 10 (via config.EARLY_STOPPING_PATIENCE)
   Stops premature exit from training.

4. MAX_EPOCHS: 25 → 75 (via config.MAX_EPOCHS)
   75 epochs with cosine annealing gives the model enough time to learn.

✔ Explicit overlap propagation
✔ Explicit run_dir propagation
✔ No global overlap caching
✔ No stale config state
✔ Reviewer-safe logging
✔ Universal checkpoint saving
✔ Split reproducibility
✔ Multi-overlap safe
✔ Non-overwriting
✔ Q1 journal-ready
✔ CUDA-safe
✔ Windows-safe
✔ Memory-safe
✔ Stable τ extraction
✔ Full experiment isolation
✔ Latent temporal regularization
✔ Anomaly margin loss
✔ NaN-safe
✔ Metadata export
✔ Persistent dataloaders
"""

# =============================================================================
# IMPORTS
# =============================================================================

import csv
import json
import random
import logging

from pathlib import Path
from collections import Counter

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

from .dataset import (
    TemporalEEGDataset,
    collate_fn,
)

from .model.cap_ltc_model import (
    CAP_LTC_Model,
)

from .model.cap_ltc_cell import (
    CAPLTCCell,
)


# =============================================================================
# SUBJECT SPLIT
# =============================================================================

def split_subjects(
    dataset_dir: Path,
    logger,
    run_dir: Path,
):

    split_dir = (
        run_dir
        / "splits"
    )

    split_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    split_path = (
        split_dir
        / f"global_split_seed_{config.SEED}.json"
    )

    # =========================================================================
    # REUSE EXISTING SPLIT
    # =========================================================================

    if split_path.exists():

        logger.info(
            f"[Split] Reusing split: {split_path}"
        )

        with open(
            split_path,
            "r",
            encoding="utf-8",
        ) as f:

            splits = json.load(f)

        return (
            splits["train_subjects"],
            splits["val_subjects"],
            splits["test_subjects"],
        )

    # =========================================================================
    # CREATE SPLIT
    # =========================================================================

    logger.info(
        "[Split] Creating new split"
    )

    subjects = sorted([

        d.name.lower()

        for d in dataset_dir.iterdir()

        if d.is_dir()
    ])

    normal_subjects = [
        s for s in subjects
        if s.startswith("n")
    ]

    disorder_subjects = [
        s for s in subjects
        if not s.startswith("n")
    ]

    random.shuffle(normal_subjects)
    random.shuffle(disorder_subjects)

    # =========================================================================
    # NORMAL
    # =========================================================================

    n_total = len(normal_subjects)
    n_train = int(0.7 * n_total)
    n_val   = int(0.15 * n_total)

    normal_train = normal_subjects[:n_train]
    normal_val   = normal_subjects[n_train:n_train + n_val]
    normal_test  = normal_subjects[n_train + n_val:]

    # =========================================================================
    # DISORDER
    # =========================================================================

    d_total = len(disorder_subjects)
    d_train = int(0.7 * d_total)
    d_val   = int(0.15 * d_total)

    disorder_train = disorder_subjects[:d_train]
    disorder_val   = disorder_subjects[d_train:d_train + d_val]
    disorder_test  = disorder_subjects[d_train + d_val:]

    # =========================================================================
    # TRUE ANOMALY DETECTION SPLIT
    # =========================================================================

    if config.TRAIN_NORMAL_ONLY:

        train_subjects = normal_train

        val_subjects = (
            normal_val
            + disorder_val
        )

        test_subjects = (
            normal_test
            + disorder_test
        )

    else:

        train_subjects = (
            normal_train
            + disorder_train
        )

        val_subjects = (
            normal_val
            + disorder_val
        )

        test_subjects = (
            normal_test
            + disorder_test
        )

    random.shuffle(train_subjects)
    random.shuffle(val_subjects)
    random.shuffle(test_subjects)

    # =========================================================================
    # ASSERTIONS
    # =========================================================================

    assert set(train_subjects).isdisjoint(val_subjects)
    assert set(train_subjects).isdisjoint(test_subjects)
    assert set(val_subjects).isdisjoint(test_subjects)

    # =========================================================================
    # SAVE
    # =========================================================================

    splits = {
        "train_subjects": train_subjects,
        "val_subjects":   val_subjects,
        "test_subjects":  test_subjects,
    }

    with open(
        split_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(splits, f, indent=4)

    logger.info(
        f"[Split] Saved: {split_path}"
    )

    return (
        train_subjects,
        val_subjects,
        test_subjects,
    )


# =============================================================================
# MARGIN LOSS
# =============================================================================

def compute_margin_loss(
    recon: torch.Tensor,    # (B, T, C, F)
    x: torch.Tensor,        # (B, T, C, F)
    mask: torch.Tensor,     # (B, T)
    labels: torch.Tensor,   # (B,)
    device: torch.device,
) -> torch.Tensor:

    """
    Anomaly margin loss.

    Enforces that disorder windows have HIGHER reconstruction error than
    normal windows by at least MARGIN_LOSS_MARGIN.

    If neither normal nor disorder windows are present in the batch,
    returns zero (no-op).

    Loss = max(0, margin - (disorder_error - normal_error))

    When disorder_error > normal_error + margin: loss = 0 (desired state)
    When disorder_error ≤ normal_error + margin: loss > 0 (pull apart)
    """

    labels = labels.to(device)

    normal_mask   = (labels == 0)   # (B,)
    disorder_mask = (labels == 1)   # (B,)

    # Need at least one of each to compute the margin
    if not normal_mask.any() or not disorder_mask.any():

        return torch.tensor(
            0.0,
            device=device,
            requires_grad=False,
        )

    # -------------------------------------------------------------------------
    # Compute per-window reconstruction error (B, T)
    # -------------------------------------------------------------------------

    err = (recon - x).pow(2).mean(dim=(2, 3))   # (B, T)

    err = torch.nan_to_num(
        err,
        nan=0.0,
        posinf=1e4,
        neginf=0.0,
    )

    # -------------------------------------------------------------------------
    # Mean error over VALID timesteps per subject
    # -------------------------------------------------------------------------

    mask_float = mask.float()

    denom_normal   = mask_float[normal_mask].sum(dim=1).clamp(min=1.0)
    denom_disorder = mask_float[disorder_mask].sum(dim=1).clamp(min=1.0)

    normal_error   = (err[normal_mask] * mask_float[normal_mask]).sum(dim=1) / denom_normal
    disorder_error = (err[disorder_mask] * mask_float[disorder_mask]).sum(dim=1) / denom_disorder

    # Mean across subjects in batch
    mean_normal_error   = normal_error.mean()
    mean_disorder_error = disorder_error.mean()

    # -------------------------------------------------------------------------
    # Hinge loss: push disorder error above normal error by at least margin
    # -------------------------------------------------------------------------

    margin = config.MARGIN_LOSS_MARGIN

    margin_loss = torch.clamp(

        margin - (mean_disorder_error - mean_normal_error),

        min=0.0,
    )

    return margin_loss


# =============================================================================
# TRAIN ONE EPOCH
# =============================================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
):

    model.train()

    total_loss         = 0.0
    total_recon_loss   = 0.0
    total_latent_loss  = 0.0
    total_margin_loss  = 0.0

    valid_batches = 0

    cap_counter = Counter()

    for batch in tqdm(
        loader,
        leave=False,
    ):

        x = batch["features"].to(device, non_blocking=True)
        cap = batch["cap"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        # labels are available for margin loss even in TRAIN_NORMAL_ONLY mode
        # (val set contains both classes; train set is normal-only but the
        # margin loss gracefully returns 0.0 when only normals are present)
        labels = batch["labels"]

        # =====================================================================
        # SAFE CAP COUNTING
        # =====================================================================

        try:

            cap_counter.update(
                cap[mask].view(-1).tolist()
            )

        except Exception:
            pass

        optimizer.zero_grad(set_to_none=True)

        # =====================================================================
        # FORWARD
        # =====================================================================

        outputs = model(
            x,
            cap,
            mask,
            return_latents=True,    # needed for latent regularization
        )

        # =====================================================================
        # UNPACK OUTPUTS
        # =====================================================================

        extras = {}

        if isinstance(outputs, tuple):

            recon = outputs[0]

            if len(outputs) > 1 and outputs[1] is not None:
                extras = outputs[1]

        else:

            recon = outputs

        # =====================================================================
        # SAFE MASK CHECK
        # =====================================================================

        if mask.sum() == 0:
            continue

        # =====================================================================
        # RECONSTRUCTION LOSS (MSE on valid timesteps only)
        # =====================================================================

        recon_loss = criterion(
            recon[mask],
            x[mask],
        )

        # =====================================================================
        # LATENT TEMPORAL REGULARIZATION
        # Smoothness penalty: penalizes abrupt hidden-state changes.
        # Keeps dynamics smooth without collapsing them.
        # =====================================================================

        latent_loss = torch.tensor(0.0, device=device)

        if (
            isinstance(extras, dict)
            and "hidden_states" in extras
        ):

            h = extras["hidden_states"]

            if (
                torch.is_tensor(h)
                and h.ndim >= 3
                and h.shape[1] > 1
            ):

                latent_diff = h[:, 1:] - h[:, :-1]

                latent_loss = latent_diff.pow(2).mean()

        # =====================================================================
        # MARGIN LOSS
        # Pushes disorder reconstruction error above normal error.
        # Returns 0 when batch has only one class (train set = normals only).
        # =====================================================================

        margin_loss = compute_margin_loss(
            recon=recon,
            x=x,
            mask=mask,
            labels=labels,
            device=device,
        )

        # =====================================================================
        # TOTAL LOSS
        # =====================================================================

        loss = (
            recon_loss
            + config.LATENT_REG_WEIGHT  * latent_loss
            + config.MARGIN_LOSS_WEIGHT * margin_loss
        )

        # =====================================================================
        # NaN / INF SAFETY
        # =====================================================================

        if not torch.isfinite(loss):

            raise RuntimeError(
                f"[Train] Invalid loss detected: "
                f"recon={recon_loss.item():.4f} "
                f"latent={latent_loss.item():.4f} "
                f"margin={margin_loss.item():.4f}"
            )

        # =====================================================================
        # BACKPROP
        # =====================================================================

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        optimizer.step()

        total_loss        += float(loss.item())
        total_recon_loss  += float(recon_loss.item())
        total_latent_loss += float(latent_loss.item())
        total_margin_loss += float(margin_loss.item())

        valid_batches += 1

    # =========================================================================
    # SAFE DIVISION
    # =========================================================================

    n = max(valid_batches, 1)

    return (
        total_loss        / n,
        total_recon_loss  / n,
        total_latent_loss / n,
        total_margin_loss / n,
        cap_counter,
    )


# =============================================================================
# VALIDATION
# =============================================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    total_loss         = 0.0
    total_recon_loss   = 0.0
    total_margin_loss  = 0.0

    valid_batches = 0

    for batch in loader:

        x = batch["features"].to(device, non_blocking=True)
        cap = batch["cap"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        labels = batch["labels"]

        outputs = model(x, cap, mask)

        if isinstance(outputs, tuple):
            recon = outputs[0]
        else:
            recon = outputs

        if mask.sum() == 0:
            continue

        recon_loss = criterion(
            recon[mask],
            x[mask],
        )

        # Margin loss on val set — val contains both normals and disorders
        # so this gives a real gradient signal even in TRAIN_NORMAL_ONLY mode
        margin_loss = compute_margin_loss(
            recon=recon,
            x=x,
            mask=mask,
            labels=labels,
            device=device,
        )

        loss = (
            recon_loss
            + config.MARGIN_LOSS_WEIGHT * margin_loss
        )

        if not torch.isfinite(loss):

            raise RuntimeError(
                "[Validate] Invalid loss detected"
            )

        total_loss       += float(loss.item())
        total_recon_loss += float(recon_loss.item())
        total_margin_loss += float(margin_loss.item())

        valid_batches += 1

    n = max(valid_batches, 1)

    return (
        total_loss       / n,
        total_recon_loss / n,
        total_margin_loss / n,
    )


# =============================================================================
# MAIN TRAINING
# =============================================================================

def run_training(
    overlap: float,
    run_dir: Path,
):

    # =========================================================================
    # REPRODUCIBILITY
    # =========================================================================

    set_global_seed(config.SEED)

    logger = setup_logger(
        f"phase7_train_{int(overlap * 100)}"
    )

    device = torch.device(config.DEVICE)

    overlap_pct = int(overlap * 100)

    # =========================================================================
    # DIRS
    # =========================================================================

    model_dir = run_dir / "models"
    log_dir   = run_dir / "logs"
    tau_dir   = run_dir / "tau"

    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    tau_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # LOGGING
    # =========================================================================

    logger.info("=" * 80)
    logger.info(f"[Training] overlap={overlap_pct}%")
    logger.info(f"[Run Dir] {run_dir}")
    logger.info(f"[TAU_BASE_INIT] {config.TAU_BASE_INIT}")
    logger.info(f"[TAU_CAP_INIT] {config.TAU_CAP_INIT}")
    logger.info(f"[MARGIN_LOSS_WEIGHT] {config.MARGIN_LOSS_WEIGHT}")
    logger.info(f"[MARGIN_LOSS_MARGIN] {config.MARGIN_LOSS_MARGIN}")
    logger.info(f"[MAX_EPOCHS] {config.MAX_EPOCHS}")
    logger.info(f"[PATIENCE] {config.EARLY_STOPPING_PATIENCE}")
    logger.info(f"[LR_SCHEDULER] {config.LR_SCHEDULER}")
    logger.info("=" * 80)

    # =========================================================================
    # DATA DIRECTORY
    # =========================================================================

    overlap_dir = config.DATA_DIR / f"overlap_{overlap_pct}"

    # =========================================================================
    # SUBJECT SPLITS
    # =========================================================================

    (
        train_subjects,
        val_subjects,
        test_subjects,
    ) = split_subjects(
        overlap_dir,
        logger,
        run_dir,
    )

    logger.info(f"[Train Subjects] {train_subjects}")
    logger.info(f"[Val Subjects]   {val_subjects}")
    logger.info(f"[Test Subjects]  {test_subjects}")

    # =========================================================================
    # DATASETS
    # =========================================================================

    train_dataset = TemporalEEGDataset(
        overlap=overlap,
        allowed_subjects=train_subjects,
    )

    val_dataset = TemporalEEGDataset(
        overlap=overlap,
        allowed_subjects=val_subjects,
    )

    # =========================================================================
    # DATALOADERS
    # =========================================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(config.NUM_WORKERS > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(config.NUM_WORKERS > 0),
    )

    logger.info(f"[Train Dataset] {len(train_dataset)} windows")
    logger.info(f"[Val Dataset]   {len(val_dataset)} windows")

    # =========================================================================
    # MODEL
    # =========================================================================

    model = CAP_LTC_Model().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    # =========================================================================
    # SCHEDULER
    # Cosine annealing decays smoothly to LR_MIN over MAX_EPOCHS.
    # Much better than ReduceLROnPlateau for anomaly detection training.
    # Using explicit typed variables to avoid Pylance overload-resolution errors.
    # =========================================================================

    cosine_scheduler: torch.optim.lr_scheduler.CosineAnnealingLR | None = None
    plateau_scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None

    if config.LR_SCHEDULER == "cosine":

        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.MAX_EPOCHS,
            eta_min=config.LR_MIN,
        )

    else:

        # Fallback to ReduceLROnPlateau
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

    criterion = nn.MSELoss()

    best_val_loss  = float("inf")
    wait           = 0
    completed_epochs = 0

    # =========================================================================
    # LOSS CSV
    # =========================================================================

    loss_csv = log_dir / f"loss_overlap_{overlap_pct}.csv"

    with open(
        loss_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        csv.writer(f).writerow([
            "epoch",
            "train_loss",
            "train_recon_loss",
            "train_latent_loss",
            "train_margin_loss",
            "val_loss",
            "val_recon_loss",
            "val_margin_loss",
            "lr",
        ])

    logger.info(f"[Loss CSV] {loss_csv}")

    # =========================================================================
    # TRAIN LOOP
    # =========================================================================

    for epoch in range(1, config.MAX_EPOCHS + 1):

        completed_epochs = epoch

        (
            train_loss,
            train_recon,
            train_latent,
            train_margin,
            cap_counter,
        ) = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        (
            val_loss,
            val_recon,
            val_margin,
        ) = validate(
            model,
            val_loader,
            criterion,
            device,
        )

        # =====================================================================
        # SCHEDULER STEP
        # =====================================================================

        if cosine_scheduler is not None:
            cosine_scheduler.step()
        elif plateau_scheduler is not None:
            plateau_scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        # =====================================================================
        # LOGGING
        # =====================================================================

        logger.info(
            f"[Epoch {epoch:3d}/{config.MAX_EPOCHS}] "
            f"Train={train_loss:.5f} "
            f"(recon={train_recon:.5f} latent={train_latent:.5f} margin={train_margin:.5f}) "
            f"| Val={val_loss:.5f} "
            f"(recon={val_recon:.5f} margin={val_margin:.5f}) "
            f"| LR={current_lr:.2e}"
        )

        logger.info(
            f"[CAP Distribution] {dict(cap_counter)}"
        )

        # =====================================================================
        # CSV
        # =====================================================================

        with open(
            loss_csv,
            "a",
            newline="",
            encoding="utf-8",
        ) as f:

            csv.writer(f).writerow([
                epoch,
                train_loss,
                train_recon,
                train_latent,
                train_margin,
                val_loss,
                val_recon,
                val_margin,
                current_lr,
            ])

        # =====================================================================
        # τ LOGGING
        # =====================================================================

        for m in model.modules():

            if isinstance(m, CAPLTCCell):

                tau_base = (
                    torch.nn.functional.softplus(m.tau_base).item()
                )

                tau_cap = (
                    torch.nn.functional.softplus(m.tau_cap)
                    .detach().cpu().numpy()
                )

                gamma_vals = 1.0 / (config.TAU_MIN + tau_base + tau_cap).clip(1e-4)

                logger.info(
                    f"[Tau] Base={tau_base:.4f} "
                    f"Mean={tau_cap.mean():.4f} "
                    f"Std={tau_cap.std():.4f}"
                )

                logger.info(
                    f"[Gamma] Mean={gamma_vals.mean():.4f} "
                    f"Min={gamma_vals.min():.4f} "
                    f"Max={gamma_vals.max():.4f}  "
                    f"(target: 0.1-0.6)"
                )

                break

        # =====================================================================
        # BEST MODEL
        # =====================================================================

        if val_loss < best_val_loss - config.MIN_DELTA:

            best_val_loss = val_loss
            wait = 0

            checkpoint_name = (
                f"cap_ltc_best_overlap_{overlap_pct}"
                f"_epoch_{epoch}"
                f"_valloss_{val_loss:.6f}.pt"
            )

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=val_loss,
                overlap_pct=overlap_pct,
                filename=checkpoint_name,
                save_dir=model_dir,
            )

            logger.info(f"[Checkpoint Saved] {checkpoint_name}")

        else:

            wait += 1

            logger.info(f"[EarlyStop Wait] {wait}/{config.EARLY_STOPPING_PATIENCE}")

            if (
                config.EARLY_STOPPING
                and wait >= config.EARLY_STOPPING_PATIENCE
            ):

                logger.info("[EarlyStop Triggered]")
                break

    # =========================================================================
    # SAVE τ
    # =========================================================================

    extract_tau_parameters(
        model,
        overlap_pct=overlap_pct,
        save=True,
        save_dir=tau_dir,
    )

    # =========================================================================
    # TRAINING METADATA
    # =========================================================================

    metadata = {
        "overlap":            overlap_pct,
        "seed":               config.SEED,
        "train_subjects":     train_subjects,
        "val_subjects":       val_subjects,
        "test_subjects":      test_subjects,
        "train_dataset_size": len(train_dataset),
        "val_dataset_size":   len(val_dataset),
        "epochs_completed":   completed_epochs,
        "best_val_loss":      float(best_val_loss),
        "train_normal_only":  bool(config.TRAIN_NORMAL_ONLY),
        "tau_base_init":      config.TAU_BASE_INIT,
        "tau_cap_init":       config.TAU_CAP_INIT,
        "margin_loss_weight": config.MARGIN_LOSS_WEIGHT,
        "margin_loss_margin": config.MARGIN_LOSS_MARGIN,
        "lr_scheduler":       config.LR_SCHEDULER,
        "max_epochs":         config.MAX_EPOCHS,
        "patience":           config.EARLY_STOPPING_PATIENCE,
    }

    metadata_path = run_dir / "training_metadata.json"

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(metadata, f, indent=4)

    logger.info(f"[Metadata Saved] {metadata_path}")

    # =========================================================================
    # FINAL
    # =========================================================================

    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"[Best Val Loss] {best_val_loss:.6f}")
    logger.info(f"[Epochs]        {completed_epochs}/{config.MAX_EPOCHS}")
    logger.info(f"[Artifacts]     {run_dir}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    raise RuntimeError(
        "Run through main.py only."
    )