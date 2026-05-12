"""
Phase 7 — CAP-LTC Explainable AI (JOURNAL-GRADE, LOCKED)
======================================================

Generates 5 publication-grade XAI figures:

Fig X1: CAP-aligned temporal explanation (EEG → Error → XAI)
Fig X2: Signed Channel × Time Integrated Gradients
Fig X3: CAP-wise Attribution Effect Size (Cohen's d)
Fig X4: Channel × CAP Sensitivity (Absolute + Relative)
Fig X5: Attribution–Error Coupling (Validation)

Author: [Your Name]
"""

import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from pathlib import Path

from .. import config
from ..dataset import TemporalEEGDataset, collate_fn
from ..model.cap_ltc_model import CAP_LTC_Model
from .xai_utils import gradient_x_input, integrated_gradients

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# UTILS
# ------------------------------------------------------------------
def normalize_signed(x: torch.Tensor, eps=1e-8):
    return x / (x.abs().max() + eps)

def cohens_d(x: np.ndarray):
    if len(x) < 2:
        return 0.0
    return np.abs(x.mean()) / (x.std() + 1e-8)

def shade_cap(ax, cap):
    caps = np.unique(cap)
    colors = plt.get_cmap("Pastel1")(np.linspace(0, 1, len(caps)))
    ymin, ymax = ax.get_ylim()

    for c, col in zip(caps, colors):
        mask = cap == c
        ax.fill_between(
            np.arange(len(cap)),
            ymin, ymax,
            where=mask,
            color=col,
            alpha=0.15,
            label=f"CAP {int(c)}"
        )
    ax.set_ylim(ymin, ymax)

# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def run_xai():

    logger.info("========== STARTING JOURNAL-GRADE XAI ==========")

    device = torch.device(config.DEVICE)
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------
    # Load model
    # --------------------------------------------------------------
    model = CAP_LTC_Model().to(device)
    ckpt = torch.load(
        config.MODEL_DIR / f"cap_ltc_best_overlap_{int(config.OVERLAP_PERCENT*100)}.pt",
        map_location=device
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    logger.info("Model loaded")

    # --------------------------------------------------------------
    # Load single representative window
    # --------------------------------------------------------------
    ds = TemporalEEGDataset(overlap=config.OVERLAP_PERCENT)
    loader = DataLoader(ds, batch_size=1, collate_fn=collate_fn)
    batch = next(iter(loader))

    x = batch["features"].to(device)        # (1, T, C, F)
    cap = batch["cap"][0].cpu().numpy()     # (T,)
    mask = batch["mask"].to(device)

    T, C = x.shape[1], x.shape[2]

    # --------------------------------------------------------------
    # Forward + error
    # --------------------------------------------------------------
    with torch.no_grad():
        recon, _ = model(x, batch["cap"].to(device), mask)
        error = ((recon - x) ** 2).mean(dim=(2,3))[0].cpu()

    # --------------------------------------------------------------
    # XAI (computed ONCE)
    # --------------------------------------------------------------
    gx = gradient_x_input(model, x, batch["cap"].to(device), mask)[0]
    ig = integrated_gradients(model, x, batch["cap"].to(device), mask)[0]

    gx_t = normalize_signed(gx.mean(dim=(1,2))).cpu()
    ig_tc = normalize_signed(ig.mean(dim=2)).cpu()   # (T, C)
    ig_t = normalize_signed(ig_tc.mean(dim=1)).cpu()

    # ==============================================================
    # FIG X1 — TEMPORAL CAP-ALIGNED EXPLANATION
    # ==============================================================
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    axs[0].plot(x[0].mean(dim=(1,2)).cpu())
    axs[0].set_title("(A) Raw EEG (Channel-averaged)")
    shade_cap(axs[0], cap)

    axs[1].plot(error)
    axs[1].set_title("(B) Reconstruction Error")
    shade_cap(axs[1], cap)

    axs[2].plot(gx_t)
    axs[2].axhline(0, ls="--", c="k")
    axs[2].set_title("(C) Gradient × Input")
    shade_cap(axs[2], cap)

    axs[3].plot(ig_t)
    axs[3].axhline(0, ls="--", c="k")
    axs[3].set_title("(D) Integrated Gradients")
    shade_cap(axs[3], cap)

    axs[-1].set_xlabel("Time (samples)")
    plt.tight_layout()
    plt.savefig(config.FIGURE_DIR / "fig_x1_temporal_cap_aligned.png", dpi=300)
    plt.close()

    # ==============================================================
    # FIG X2 — SIGNED CHANNEL × TIME IG
    # ==============================================================
    plt.figure(figsize=(12, 5))
    sns.heatmap(
        ig_tc.T,
        cmap="RdBu_r",
        center=0,
        cbar_kws={"label": "Signed Attribution"}
    )
    plt.xlabel("Time")
    plt.ylabel("EEG Channel")
    plt.title("Fig X2: Channel × Time Attribution (Integrated Gradients)")
    plt.tight_layout()
    plt.savefig(config.FIGURE_DIR / "fig_x2_channel_time_ig.png", dpi=300)
    plt.close()

    # ==============================================================
    # FIG X3 — CAP-WISE EFFECT SIZE
    # ==============================================================
    caps = np.unique(cap)
    effects = []

    for c in caps:
        effects.append(cohens_d(ig_t[cap == c].numpy()))

    plt.figure(figsize=(7,4))
    plt.bar([f"CAP {int(c)}" for c in caps], effects)
    plt.ylabel("Effect Size (Cohen's d)")
    plt.title("Fig X3: CAP-wise Attribution Effect Size")
    plt.tight_layout()
    plt.savefig(config.FIGURE_DIR / "fig_x3_cap_effect_size.png", dpi=300)
    plt.close()

    # ==============================================================
    # FIG X4 — CHANNEL × CAP SENSITIVITY (ABS + REL)
    # ==============================================================
    abs_mat = np.zeros((C, len(caps)))
    for i, c in enumerate(caps):
        abs_mat[:, i] = np.abs(ig_tc[cap == c]).mean(axis=0)

    rel_mat = abs_mat / (abs_mat.sum(axis=1, keepdims=True) + 1e-8)

    fig, axs = plt.subplots(1, 2, figsize=(14,5))

    sns.heatmap(abs_mat, cmap="YlOrRd", ax=axs[0])
    axs[0].set_title("(A) Absolute Sensitivity")
    axs[0].set_xlabel("CAP State")
    axs[0].set_ylabel("Channel")

    sns.heatmap(rel_mat, cmap="magma", ax=axs[1])
    axs[1].set_title("(B) Relative (Row-normalized)")
    axs[1].set_xlabel("CAP State")

    plt.suptitle("Fig X4: Channel × CAP Sensitivity Matrix")
    plt.tight_layout()
    plt.savefig(config.FIGURE_DIR / "fig_x4_channel_cap_matrix.png", dpi=300)
    plt.close()

    # ==============================================================
    # FIG X5 — ATTRIBUTION–ERROR COUPLING
    # ==============================================================
    r = np.corrcoef(error.numpy(), ig_t.numpy())[0,1]

    plt.figure(figsize=(6,5))
    sns.regplot(x=error.numpy(), y=ig_t.numpy())
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Integrated Gradients")
    plt.title(f"Fig X5: Attribution–Error Coupling (r={r:.2f})")
    plt.tight_layout()
    plt.savefig(config.FIGURE_DIR / "fig_x5_attr_error_coupling.png", dpi=300)
    plt.close()

    logger.info("✅ ALL 5 JOURNAL-GRADE XAI FIGURES GENERATED")

# ------------------------------------------------------------------
if __name__ == "__main__":
    run_xai()