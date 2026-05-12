# src/phase7_1_cap_ltc/xai/xai_integrated_gradients.py

"""
Phase 7.x — Integrated Gradients XAI
====================================

Provides:
✔ Integrated Gradients attribution
✔ EEG temporal attribution maps
✔ Channel importance analysis
✔ Feature saliency extraction
✔ CAP-aware explainability
✔ Subject-level explainability
✔ Reviewer-grade neuroscience visualization

Outputs
-------
results/
└── xai/
    ├── integrated_gradients/
    │   ├── attribution_maps.npy
    │   ├── attribution_summary.csv
    │   ├── ig_temporal_heatmap.png
    │   ├── ig_channel_importance.png
    │   ├── ig_feature_importance.png
    │   └── ig_subject_examples.png
"""

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import DataLoader

from captum.attr import IntegratedGradients

from .. import config
from ..dataset import TemporalEEGDataset
from ..model.cap_ltc_model import CAP_LTC_Model


# =============================================================================
# STYLE
# =============================================================================
sns.set_style("whitegrid")

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 120,
})


# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================
XAI_DIR = (
    config.RESULTS_DIR
    / "xai"
    / "integrated_gradients"
)

XAI_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# LOAD MODEL
# =============================================================================
def load_model():

    model = CAP_LTC_Model()

    model_path = (
        config.MODEL_DIR
        / f"cap_ltc_best_overlap_{int(config.OVERLAP_PERCENT * 100)}.pt"
    )

    ckpt = torch.load(
        model_path,
        map_location=config.DEVICE,
    )

    model.load_state_dict(
        ckpt["model_state_dict"]
    )

    model.to(config.DEVICE)

    model.eval()

    logging.info(
        f"[IG] Loaded model: {model_path}"
    )

    return model


# =============================================================================
# LOAD DATA
# =============================================================================
def load_dataset():

    dataset = TemporalEEGDataset(
        overlap=config.OVERLAP_PERCENT,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    return loader


# =============================================================================
# FORWARD FUNCTION FOR CAPTUM
# =============================================================================
class IGWrapper(torch.nn.Module):

    def __init__(
        self,
        model,
        cap,
        mask,
    ):
        super().__init__()

        self.model = model
        self.cap = cap
        self.mask = mask

    def forward(
        self,
        x,
    ):

        recon, _ = self.model(
            x,
            self.cap,
            self.mask,
        )

        # -------------------------------------------------------------
        # Reconstruction anomaly score
        # -------------------------------------------------------------
        err = (
            (recon - x) ** 2
        ).mean()

        return err.unsqueeze(0)


# =============================================================================
# COMPUTE INTEGRATED GRADIENTS
# =============================================================================
def compute_integrated_gradients():

    logging.info(
        "[IG] Computing Integrated Gradients..."
    )

    model = load_model()

    loader = load_dataset()

    attribution_maps = []

    summary_rows = []

    for idx, batch in enumerate(loader):

        logging.info(
            f"[IG] Subject {idx+1}"
        )

        x = batch["x"].to(
            config.DEVICE
        )

        cap = batch["cap"].to(
            config.DEVICE
        )

        mask = batch["mask"].to(
            config.DEVICE
        )

        label = int(
            batch["label"][0]
        )

        subject_id = str(
            batch["subject_id"][0]
        )

        # -------------------------------------------------------------
        # Wrap model
        # -------------------------------------------------------------
        wrapper = IGWrapper(
            model,
            cap,
            mask,
        )

        ig = IntegratedGradients(
            wrapper
        )

        baseline = torch.zeros_like(x)

        attributions = ig.attribute(
            x,
            baselines=baseline,
            n_steps=32,
        )

        attr_np = (
            attributions
            .detach()
            .cpu()
            .numpy()
        )

        attribution_maps.append(
            attr_np
        )

        # -------------------------------------------------------------
        # Temporal importance
        # -------------------------------------------------------------
        temporal_importance = np.mean(
            np.abs(attr_np),
            axis=(0, 2, 3),
        )

        # -------------------------------------------------------------
        # Channel importance
        # -------------------------------------------------------------
        channel_importance = np.mean(
            np.abs(attr_np),
            axis=(0, 1, 3),
        )

        # -------------------------------------------------------------
        # Feature importance
        # -------------------------------------------------------------
        feature_importance = np.mean(
            np.abs(attr_np),
            axis=(0, 1, 2),
        )

        summary_rows.append({

            "subject_id":
                subject_id,

            "label":
                label,

            "mean_importance":
                float(
                    np.mean(
                        np.abs(attr_np)
                    )
                ),

            "max_importance":
                float(
                    np.max(
                        np.abs(attr_np)
                    )
                ),
        })

        # -------------------------------------------------------------
        # Save example plots
        # -------------------------------------------------------------
        if idx < 5:

            plot_temporal_heatmap(
                temporal_importance,
                subject_id,
            )

            plot_channel_importance(
                channel_importance,
                subject_id,
            )

            plot_feature_importance(
                feature_importance,
                subject_id,
            )

    # -------------------------------------------------------------------------
    # Save arrays
    # -------------------------------------------------------------------------
    np.save(
        XAI_DIR
        / "attribution_maps.npy",
        np.asarray(
            attribution_maps,
            dtype=object,
        ),
    )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        XAI_DIR
        / "attribution_summary.csv",
        index=False,
    )

    logging.info(
        "✅ Integrated Gradients complete"
    )


# =============================================================================
# TEMPORAL HEATMAP
# =============================================================================
def plot_temporal_heatmap(
    temporal_importance,
    subject_id,
):

    plt.figure(figsize=(8, 3))

    sns.heatmap(
        temporal_importance.reshape(1, -1),
        cmap="viridis",
        cbar=True,
    )

    plt.xlabel("Time")
    plt.yticks([])

    plt.title(
        f"Temporal Attribution — {subject_id}"
    )

    plt.tight_layout()

    plt.savefig(
        XAI_DIR
        / f"{subject_id}_temporal_heatmap.png",
        dpi=300,
    )

    plt.close()


# =============================================================================
# CHANNEL IMPORTANCE
# =============================================================================
def plot_channel_importance(
    channel_importance,
    subject_id,
):

    plt.figure(figsize=(7, 4))

    plt.bar(
        np.arange(
            len(channel_importance)
        ),
        channel_importance,
    )

    plt.xlabel("EEG Channel")
    plt.ylabel("Importance")

    plt.title(
        f"Channel Attribution — {subject_id}"
    )

    plt.tight_layout()

    plt.savefig(
        XAI_DIR
        / f"{subject_id}_channel_importance.png",
        dpi=300,
    )

    plt.close()


# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================
def plot_feature_importance(
    feature_importance,
    subject_id,
):

    plt.figure(figsize=(7, 4))

    plt.bar(
        np.arange(
            len(feature_importance)
        ),
        feature_importance,
    )

    plt.xlabel("Feature Index")
    plt.ylabel("Importance")

    plt.title(
        f"Feature Attribution — {subject_id}"
    )

    plt.tight_layout()

    plt.savefig(
        XAI_DIR
        / f"{subject_id}_feature_importance.png",
        dpi=300,
    )

    plt.close()


# =============================================================================
# MASTER
# =============================================================================
def run_integrated_gradients():

    logging.basicConfig(
        level=logging.INFO
    )

    compute_integrated_gradients()


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":

    run_integrated_gradients()