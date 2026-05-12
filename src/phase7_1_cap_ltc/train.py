# src/phase7_1_cap_ltc/train.py

import csv
import random
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
from .dataset import TemporalEEGDataset, collate_fn
from .model.cap_ltc_model import CAP_LTC_Model
from .model.cap_ltc_cell import CAPLTCCell


# =============================================================================
# SUBJECT SPLIT
# =============================================================================
def split_subjects(dataset_dir):

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

    # ---------------------------------------------------------
    # NORMAL SPLIT
    # ---------------------------------------------------------
    n_total = len(normal_subjects)

    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)

    normal_train = normal_subjects[:n_train]
    normal_val = normal_subjects[n_train:n_train+n_val]
    normal_test = normal_subjects[n_train+n_val:]

    # ---------------------------------------------------------
    # DISORDER SPLIT
    # ---------------------------------------------------------
    d_total = len(disorder_subjects)

    d_train = int(0.7 * d_total)
    d_val = int(0.15 * d_total)

    disorder_train = disorder_subjects[:d_train]
    disorder_val = disorder_subjects[d_train:d_train+d_val]
    disorder_test = disorder_subjects[d_train+d_val:]

    # ---------------------------------------------------------
    # COMBINE
    # ---------------------------------------------------------
    train_subjects = (
        normal_train +
        disorder_train
    )

    val_subjects = (
        normal_val +
        disorder_val
    )

    test_subjects = (
        normal_test +
        disorder_test
    )

    random.shuffle(train_subjects)
    random.shuffle(val_subjects)
    random.shuffle(test_subjects)

    return (
        train_subjects,
        val_subjects,
        test_subjects,
    )

# =============================================================================
# TRAIN ONE EPOCH
# =============================================================================
def train_one_epoch(model, loader, optimizer, criterion, device):

    model.train()

    total_loss = 0.0
    cap_counter = Counter()

    for batch in tqdm(loader, leave=False):

        x = batch["features"].to(device)
        cap = batch["cap"].to(device)
        mask = batch["mask"].to(device)

        cap_counter.update(cap[mask].view(-1).tolist())

        optimizer.zero_grad()

        recon, _ = model(x, cap, mask)

        loss = criterion(recon[mask], x[mask])

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader), cap_counter


# =============================================================================
# VALIDATION
# =============================================================================
@torch.no_grad()
def validate(model, loader, criterion, device):

    model.eval()

    total_loss = 0.0

    for batch in loader:

        x = batch["features"].to(device)
        cap = batch["cap"].to(device)
        mask = batch["mask"].to(device)

        recon, _ = model(x, cap, mask)

        loss = criterion(recon[mask], x[mask])

        total_loss += loss.item()

    return total_loss / len(loader)


# =============================================================================
# MAIN TRAINING
# =============================================================================
def run_training():

    set_global_seed(config.SEED)

    logger = setup_logger("phase7_1_train")

    device = torch.device(config.DEVICE)

    overlap_dir = (
        config.DATA_DIR /
        f"overlap_{int(config.OVERLAP_PERCENT * 100)}"
    )

    # -------------------------------------------------------------------------
    # SUBJECT SPLITS
    # -------------------------------------------------------------------------
    train_subjects, val_subjects, test_subjects = split_subjects(
        overlap_dir
    )

    logger.info(f"Train Subjects: {train_subjects}")
    logger.info(f"Val Subjects: {val_subjects}")
    logger.info(f"Test Subjects: {test_subjects}")

    # -------------------------------------------------------------------------
    # DATASETS
    # -------------------------------------------------------------------------
    train_dataset = TemporalEEGDataset(
        overlap=config.OVERLAP_PERCENT,
        allowed_subjects=train_subjects,
    )

    val_dataset = TemporalEEGDataset(
        overlap=config.OVERLAP_PERCENT,
        allowed_subjects=val_subjects,
    )

    # -------------------------------------------------------------------------
    # LOADERS
    # -------------------------------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.NUM_WORKERS,
    )

    # -------------------------------------------------------------------------
    # MODEL
    # -------------------------------------------------------------------------
    model = CAP_LTC_Model().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    criterion = nn.MSELoss()

    overlap = int(config.OVERLAP_PERCENT * 100)

    best_val_loss = float("inf")

    wait = 0

    # -------------------------------------------------------------------------
    # CSV LOGGING
    # -------------------------------------------------------------------------
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    loss_csv = config.LOG_DIR / f"loss_overlap_{overlap}.csv"

    with open(loss_csv, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch",
            "train_loss",
            "val_loss"
        ])

    # -------------------------------------------------------------------------
    # TRAIN LOOP
    # -------------------------------------------------------------------------
    for epoch in range(1, config.MAX_EPOCHS + 1):

        train_loss, cap_counter = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        val_loss = validate(
            model,
            val_loader,
            criterion,
            device,
        )

        logger.info(
            f"[Epoch {epoch}] "
            f"Train={train_loss:.6f} "
            f"Val={val_loss:.6f}"
        )

        # ---------------------------------------------------------------------
        # SAVE CSV
        # ---------------------------------------------------------------------
        with open(loss_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                train_loss,
                val_loss,
            ])

        # ---------------------------------------------------------------------
        # TAU LOGGING
        # ---------------------------------------------------------------------
        for m in model.modules():

            if isinstance(m, CAPLTCCell):

                tau_base = (
                    torch.nn.functional.softplus(
                        m.tau_base
                    ).item()
                )

                tau_cap = (
                    torch.nn.functional.softplus(
                        m.tau_cap
                    ).detach().cpu().numpy()
                )

                logger.info(
                    f"[Tau] Base={tau_base:.4f} "
                    f"Mean={tau_cap.mean():.4f} "
                    f"Std={tau_cap.std():.4f}"
                )

                break

        # ---------------------------------------------------------------------
        # EARLY STOPPING
        # ---------------------------------------------------------------------
        if val_loss < best_val_loss:

            best_val_loss = val_loss

            wait = 0

            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_loss,
                f"cap_ltc_best_overlap_{overlap}.pt"
            )

        else:

            wait += 1

            if wait >= config.EARLY_STOPPING_PATIENCE:

                logger.info("[EarlyStop] Triggered")

                break

    extract_tau_parameters(model, save=True)

    logger.info("✅ TRAINING COMPLETE")


if __name__ == "__main__":
    run_training()