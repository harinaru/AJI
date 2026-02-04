"""
Phase 7.x — Inference & Subject-Level Scoring (FINAL, SAFE)
----------------------------------------------------------
✔ Subject-level instability profiling
✔ Saves window-level scores + labels
✔ Overlap-aware directory structure
✔ Visualization-compatible
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
from .metrics import reconstruction_error, delta_error, instability_index
from .model.cap_ltc_model import CAP_LTC_Model


# =============================================================================
# SUBJECT-LEVEL INFERENCE
# =============================================================================
@torch.no_grad()
def infer_subjects(model, loader, device):
    model.eval()

    rows: List[Dict] = []
    all_scores = []
    all_labels = []

    for batch in tqdm(loader, desc="[Inference]", ncols=100):
        x = batch["features"].to(device)
        cap = batch["cap"].to(device)
        mask = batch["mask"].to(device)

        subject_ids = batch["subject_ids"]
        labels = batch["labels"]
        lengths = batch["lengths"]

        recon, _ = model(x, cap, mask)

        err_ts = reconstruction_error(recon, x, mask)
        d_err_ts = delta_error(err_ts, mask)

        for i, sid in enumerate(subject_ids):
            L = lengths[i].item()
            err = err_ts[i, :L]
            d_err = d_err_ts[i, :L-1]

            metrics = instability_index(
                err[None, :],
                d_err[None, :],
                mask[i, :L][None, :],
            )

            instability = metrics["mean_error"] + metrics["mean_delta"]

            rows.append({
                "subject_id": sid,
                "label": int(labels[i]),
                "label_name": "Control" if labels[i] == 0 else "Disorder",
                "n_windows": int(L),
                "mean_error": metrics["mean_error"],
                "p90_error": metrics["p90_error"],
                "mean_delta": metrics["mean_delta"],
                "p90_delta": metrics["p90_delta"],
                "instability_index": instability,
            })

            # window-level score for ROC
            all_scores.append(instability)
            all_labels.append(int(labels[i]))

    return pd.DataFrame(rows), np.array(all_scores), np.array(all_labels)


# =============================================================================
# MAIN INFERENCE RUNNER
# =============================================================================
def run_inference():
    logging.basicConfig(level=logging.INFO)
    device = torch.device(config.DEVICE)

    overlap_pct = int(config.OVERLAP_PERCENT * 100)

    # -----------------------------
    # LOAD MODEL
    # -----------------------------
    ckpt = config.MODEL_DIR / f"cap_ltc_best_overlap_{overlap_pct}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

    model = CAP_LTC_Model().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])
    model.eval()

    logging.info(f"[Inference] Loaded model → {ckpt}")

    # -----------------------------
    # DATA
    # -----------------------------
    dataset = TemporalEEGDataset(overlap=config.OVERLAP_PERCENT)
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.NUM_WORKERS,
    )

    # -----------------------------
    # RUN
    # -----------------------------
    df, scores, labels = infer_subjects(model, loader, device)

    # -----------------------------
    # SAVE (OVERLAP SAFE)
    # -----------------------------
    out_dir = config.RESULTS_DIR / "inference" / f"overlap_{overlap_pct}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / "subject_level_metrics.csv", index=False)
    np.save(out_dir / "scores_labels.npy", np.array([scores, labels], dtype=object))

    logging.info(f"[Inference] Saved → {out_dir}")
    logging.info(df.groupby("label_name")["instability_index"].mean())

    logging.info("[Inference] COMPLETED")


if __name__ == "__main__":
    run_inference()
