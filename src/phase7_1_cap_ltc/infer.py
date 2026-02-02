"""
Phase 7.x — Inference & Subject-Level Scoring (FINAL)
-----------------------------------------------------
✔ Subject-level risk profiling
✔ Normal-only reconstruction inference
✔ Temporal instability analysis
✔ CSV outputs (publication-ready)
"""

import logging
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import config
from .dataset import TemporalEEGDataset, collate_fn
from .metrics import (
    reconstruction_error,
    delta_error,
    instability_index,
)
from .model.cap_ltc_model import CAP_LTC_Model


# =============================================================================
# SUBJECT-LEVEL INFERENCE
# =============================================================================
@torch.no_grad()
def infer_subjects(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    """
    Run inference and compute subject-level metrics.
    """

    model.eval()
    rows: List[Dict] = []

    for batch in tqdm(loader, desc="[Inference]", ncols=100):
        features = batch["features"].to(device)   # [B, T, C, F]
        cap = batch["cap"].to(device)
        mask = batch["mask"].to(device)

        subject_ids = batch["subject_ids"]
        labels = batch["labels"]
        lengths = batch["lengths"]

        recon, _ = model(features, cap, mask)

        err_ts = reconstruction_error(recon, features, mask)   # [B, T]
        d_err_ts = delta_error(err_ts, mask)                    # [B, T-1]

        for i, sid in enumerate(subject_ids):
            L = lengths[i].item()

            err = err_ts[i, :L]
            d_err = d_err_ts[i, :L-1]
            cap_i = cap[i, :L]

            metrics = instability_index(
                err[None, :],
                d_err[None, :],
                mask[i, :L][None, :],
            )

            # Explicit instability index (reviewer-safe)
            instability = metrics["mean_error"] + metrics["mean_delta"]

            rows.append({
                "subject_id": sid,
                "label": int(labels[i]),
                "label_name": "Control" if labels[i] == 0 else "Disorder",
                "n_windows": int(L),
                "cap_ratio": float((cap_i > 0).float().mean().item()),
                "mean_error": metrics["mean_error"],
                "p90_error": metrics["p90_error"],
                "mean_delta": metrics["mean_delta"],
                "p90_delta": metrics["p90_delta"],
                "instability_index": instability,
            })

    return pd.DataFrame(rows)


# =============================================================================
# MAIN INFERENCE RUNNER
# =============================================================================
def run_inference() -> None:
    logging.basicConfig(level=logging.INFO)
    device = torch.device(config.DEVICE)

    # ------------------------------------------------------------------
    # LOAD MODEL
    # ------------------------------------------------------------------
    model = CAP_LTC_Model().to(device)

    ckpt_path = config.MODEL_DIR / config.CHECKPOINT_NAME
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}"
        )

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    logging.info(f"[Inference] Loaded model → {ckpt_path}")

    # ------------------------------------------------------------------
    # DATASET (ALL SUBJECTS)
    # ------------------------------------------------------------------
    dataset = TemporalEEGDataset(overlap=config.OVERLAP_PERCENT)

    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.NUM_WORKERS,
    )

    logging.info(f"[Data] Inference windows: {len(dataset)}")

    # ------------------------------------------------------------------
    # RUN INFERENCE
    # ------------------------------------------------------------------
    df = infer_subjects(model, loader, device)

    # ------------------------------------------------------------------
    # SAVE OUTPUTS
    # ------------------------------------------------------------------
    out_dir = config.RESULTS_DIR / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "subject_level_metrics.csv"
    df.to_csv(csv_path, index=False)

    logging.info(f"[Inference] Saved → {csv_path}")

    # Quick sanity summary
    summary = df.groupby("label_name")["instability_index"].mean()
    logging.info("[Summary] Mean instability index by group:")
    logging.info(f"\n{summary}")

    logging.info("[Inference] Completed successfully")


if __name__ == "__main__":
    run_inference()
