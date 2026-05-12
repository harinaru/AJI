# src/phase7_1_cap_ltc/infer.py

import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import config
from .dataset import TemporalEEGDataset, collate_fn
from .metrics import reconstruction_error, delta_error
from .model.cap_ltc_model import CAP_LTC_Model


# =============================================================================
# SUBJECT SPLIT
# =============================================================================
def get_test_subjects(overlap_dir):

    subjects = sorted([
        d.name.lower()
        for d in overlap_dir.iterdir()
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

    random.seed(config.SEED)

    random.shuffle(normal_subjects)

    n_total = len(normal_subjects)

    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)

    heldout_normals = normal_subjects[n_train+n_val:]

    return heldout_normals + disorder_subjects


# =============================================================================
# INFERENCE
# =============================================================================
@torch.no_grad()
def run_inference():

    device = torch.device(config.DEVICE)

    overlap = int(config.OVERLAP_PERCENT * 100)

    overlap_dir = (
        config.DATA_DIR /
        f"overlap_{overlap}"
    )

    test_subjects = get_test_subjects(overlap_dir)

    # -------------------------------------------------------------------------
    # MODEL
    # -------------------------------------------------------------------------
    model = CAP_LTC_Model().to(device)

    ckpt = torch.load(
        config.MODEL_DIR /
        f"cap_ltc_best_overlap_{overlap}.pt",
        map_location=device,
    )

    model.load_state_dict(ckpt["model_state"])

    model.eval()

    # -------------------------------------------------------------------------
    # DATASET
    # -------------------------------------------------------------------------
    dataset = TemporalEEGDataset(
        overlap=config.OVERLAP_PERCENT,
        allowed_subjects=test_subjects,
    )

    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    subj_rows = []
    win_rows = []

    # -------------------------------------------------------------------------
    # INFERENCE LOOP
    # -------------------------------------------------------------------------
    for batch in tqdm(loader, desc="Inference"):

        x = batch["features"].to(device)
        cap = batch["cap"].to(device)
        mask = batch["mask"].to(device)

        labels = batch["labels"]
        sids = batch["subject_ids"]
        lengths = batch["lengths"]

        recon, latent = model(x, cap, mask)

        err = reconstruction_error(recon, x, mask)

        derr = delta_error(err, mask)

        for i, sid in enumerate(sids):

            L = lengths[i].item()

            e = err[i, :L].cpu()

            d = derr[i, :L-1].cpu()

            label = labels[i].item()

            label_name = (
                "Control"
                if label == 0
                else "Disorder"
            )

            subj_rows.append({
                "subject_id": sid,
                "label": label,
                "label_name": label_name,
                "mean_error": e.mean().item(),
                "mean_delta": d.mean().item(),
                "instability_index": d.std().item(),
                "p90_error": torch.quantile(e, 0.9).item(),
                "p90_delta": torch.quantile(d, 0.9).item(),
            })

            for t in range(L):

                win_rows.append({
                    "subject_id": sid,
                    "window_idx": t,
                    "cap_state": int(cap[i, t].item()),
                    "reconstruction_error": e[t].item(),
                    "delta_error": (
                        d[t].item()
                        if t < L - 1
                        else np.nan
                    ),
                })

    # -------------------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------------------
    out = (
        config.RESULTS_DIR /
        "inference" /
        f"overlap_{overlap}"
    )

    out.mkdir(parents=True, exist_ok=True)

    subj_df = pd.DataFrame(subj_rows)

    win_df = pd.DataFrame(win_rows)

    subj_df.to_csv(
        out / "subject_level_metrics.csv",
        index=False
    )

    win_df.to_csv(
        out / "window_level_metrics.csv",
        index=False
    )

    np.save(
        out / "scores_labels.npy",
        np.array([
            subj_df["instability_index"].to_numpy(),
            subj_df["label"].to_numpy(),
        ], dtype=object)
    )

    print("✅ INFERENCE COMPLETE")


if __name__ == "__main__":
    run_inference()