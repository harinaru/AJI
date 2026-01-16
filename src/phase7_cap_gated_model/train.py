"""
Phase 7 — CAP-Gated Temporal Model Training (FINAL)

✔ Delta prediction
✔ Truncated BPTT
✔ Controls-only training
✔ Stable loss
✔ Journal-grade reproducibility
"""

import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from .config import Config
from .dataset import TemporalEEGDataset, collate_fn
from .model import CAPGatedTemporalModel


# ======================================================
# DATA SPLIT (CONTROLS ONLY)
# ======================================================

def split_controls(dataset, train_ratio=0.8, seed=42):
    rng = np.random.default_rng(seed)
    idx = [i for i, s in enumerate(dataset.samples) if s["label"] == 0]
    rng.shuffle(idx)

    n_train = int(len(idx) * train_ratio)
    return idx[:n_train], idx[n_train:]


# ======================================================
# TRAIN / VALIDATE (TRUNCATED BPTT)
# ======================================================

def train_one_epoch(model, loader, optimizer, device, chunk_len=100):
    model.train()
    total_loss, n_chunks = 0.0, 0

    for batch in tqdm(loader, desc="Training", leave=False):
        features = batch["features"].to(device)
        cap = batch["cap_flags"].to(device)
        mask = batch["mask"].to(device)

        _, T, _ = features.shape

        for t0 in range(0, T, chunk_len):
            t1 = min(t0 + chunk_len, T)

            f = features[:, t0:t1]
            c = cap[:, t0:t1]
            m = mask[:, t0:t1]

            if m.sum() == 0:
                continue

            optimizer.zero_grad()
            loss = model.compute_loss(f, c, m)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_chunks += 1

    return total_loss / max(n_chunks, 1)


@torch.no_grad()
def validate(model, loader, device, chunk_len=100):
    model.eval()
    total_loss, n_chunks = 0.0, 0

    for batch in loader:
        features = batch["features"].to(device)
        cap = batch["cap_flags"].to(device)
        mask = batch["mask"].to(device)

        _, T, _ = features.shape

        for t0 in range(0, T, chunk_len):
            t1 = min(t0 + chunk_len, T)

            f = features[:, t0:t1]
            c = cap[:, t0:t1]
            m = mask[:, t0:t1]

            if m.sum() == 0:
                continue

            loss = model.compute_loss(f, c, m)
            total_loss += loss.item()
            n_chunks += 1

    return total_loss / max(n_chunks, 1)


# ======================================================
# TRAIN ONE CHANNEL
# ======================================================

def train_channel(channel, config, device):
    print("\n" + "=" * 80)
    print(f"🧠 TRAINING CHANNEL: {channel}")
    print("=" * 80)

    feature_names = list(
        np.loadtxt(
            config.PHASE5_FEATURE_DIR / f"{channel}_selected_features.csv",
            dtype=str,
            delimiter=",",
            skiprows=1,
        )
    )

    dataset = TemporalEEGDataset(
        config.PHASE7_DATA_DIR / f"{channel}_clean.csv",
        feature_names,
    )

    if len(dataset) == 0:
        print("⚠️ No valid subjects — skipping")
        return

    train_idx, val_idx = split_controls(dataset)
    if not train_idx or not val_idx:
        print("⚠️ Not enough controls — skipping")
        return

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = CAPGatedTemporalModel(
        input_dim=config.INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        output_dim=config.OUTPUT_DIM,
        alpha_A_init=config.ALPHA_A_INIT,
        alpha_B_init=config.ALPHA_B_INIT,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val = float("inf")
    history = []

    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.NUM_EPOCHS}")

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)

        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train": train_loss, "val": val_loss})

        print(f"Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                model.state_dict(),
                config.CHECKPOINT_DIR / f"{channel}_best.pt",
            )
            print("✅ Best model saved")

    with open(config.LOG_DIR / f"{channel}_train_log.json", "w") as f:
        json.dump(history, f, indent=2)


# ======================================================
# MAIN
# ======================================================

def main():
    config = Config()
    config.create_directories()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device: {device}")

    for ch in config.CHANNELS:
        train_channel(ch, config, device)

    print("\n🎉 PHASE 7 TRAINING COMPLETE")
