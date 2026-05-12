# src/phase7_1_cap_ltc/analysis/latent.py

"""
Phase 7.x — Latent Space Analysis Module
========================================

Provides:
✔ Latent embedding extraction
✔ Subject-level aggregation
✔ t-SNE visualization
✔ UMAP visualization
✔ Clinical group coloring
✔ CAP-aware embedding analysis
✔ Journal-grade export pipeline
✔ Reviewer-safe implementation

Outputs
-------
results/
└── latent/
    ├── latent_embeddings.npy
    ├── latent_labels.npy
    ├── latent_subject_ids.npy
    ├── tsne_embeddings.csv
    ├── umap_embeddings.csv
    ├── tsne_plot.png
    └── umap_plot.png
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import importlib

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE

# Optional dependency: `umap-learn` (import name: `umap`).
# Use runtime import to avoid static "import could not be resolved" errors.
umap: Optional[Any]
try:
    umap = importlib.import_module("umap")
    HAS_UMAP = True
except Exception:
    umap = None
    HAS_UMAP = False

import torch
from torch.utils.data import DataLoader

from . import config
from ..phase7_1_cap_ltc.model.cap_ltc_model import CAP_LTC_Model
from ..phase7_1_cap_ltc.dataset import TemporalEEGDataset


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
# OUTPUT DIR
# =============================================================================
LATENT_DIR = (
    config.RESULTS_DIR
    / "latent"
)

LATENT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# LOAD MODEL
# =============================================================================
def load_model() -> CAP_LTC_Model:

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
        f"[Latent] Loaded model: {model_path}"
    )

    return model


# =============================================================================
# LOAD DATASET
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
# SUBJECT LATENT AGGREGATION
# =============================================================================
def aggregate_latent(
    hidden_states: torch.Tensor,
) -> np.ndarray:
    """
    Aggregate temporal hidden states
    into a single subject embedding.

    Input
    -----
    hidden_states:
        [B*C, T, H]

    Returns
    -------
    embedding:
        [H]
    """

    latent = hidden_states.mean(dim=1)

    latent = latent.mean(dim=0)

    return (
        latent
        .detach()
        .cpu()
        .numpy()
    )


# =============================================================================
# EXTRACT LATENTS
# =============================================================================
def extract_latents():

    logging.info(
        "[Latent] Extracting embeddings..."
    )

    model = load_model()

    loader = load_dataset()

    embeddings_list: List[np.ndarray] = []
    labels_list: List[int] = []
    subject_ids_list: List[str] = []

    with torch.no_grad():

        for batch in loader:

            x = batch["x"].to(config.DEVICE)

            cap = batch["cap"].to(config.DEVICE)

            mask = batch["mask"].to(config.DEVICE)

            label = int(batch["label"][0])

            subject_id = str(
                batch["subject_id"][0]
            )

            recon, extras = model(
                x,
                cap,
                mask,
                return_latents=True,
            )

            hidden_states = extras[
                "hidden_states"
            ]

            embedding = aggregate_latent(
                hidden_states
            )

            embeddings_list.append(embedding)

            labels_list.append(label)

            subject_ids_list.append(subject_id)

    embeddings = np.stack(
        embeddings_list,
        axis=0,
    )

    labels = np.asarray(labels_list)

    subject_ids = np.asarray(
        subject_ids_list
    )

    np.save(
        LATENT_DIR / "latent_embeddings.npy",
        embeddings,
    )

    np.save(
        LATENT_DIR / "latent_labels.npy",
        labels,
    )

    np.save(
        LATENT_DIR / "latent_subject_ids.npy",
        subject_ids,
    )

    logging.info(
        f"[Latent] Saved embeddings: {embeddings.shape}"
    )

    return (
        embeddings,
        labels,
        subject_ids,
    )


# =============================================================================
# T-SNE
# =============================================================================
def run_tsne(
    embeddings: np.ndarray,
):

    logging.info(
        "[Latent] Running t-SNE..."
    )

    tsne = TSNE(
        n_components=2,
        perplexity=10,
        learning_rate="auto",
        init="pca",
        random_state=42,
    )

    z = tsne.fit_transform(
        embeddings
    )

    df = pd.DataFrame({
        "x": z[:, 0],
        "y": z[:, 1],
    })

    df.to_csv(
        LATENT_DIR / "tsne_embeddings.csv",
        index=False,
    )

    return z


# =============================================================================
# UMAP
# =============================================================================
def run_umap(
    embeddings: np.ndarray,
):

    if not HAS_UMAP or umap is None:

        logging.warning(
            "[Latent] UMAP not installed"
        )

        return None

    logging.info(
        "[Latent] Running UMAP..."
    )

    reducer = umap.UMAP(
        n_components=2,
        random_state=42,
    )

    z = reducer.fit_transform(
        embeddings
    )

    df = pd.DataFrame({
        "x": z[:, 0],
        "y": z[:, 1],
    })

    df.to_csv(
        LATENT_DIR / "umap_embeddings.csv",
        index=False,
    )

    return z


# =============================================================================
# PLOT EMBEDDINGS
# =============================================================================
def plot_embeddings(
    z: np.ndarray,
    labels: np.ndarray,
    title: str,
    save_name: str,
):

    label_names = np.where(
        labels == 0,
        "Control",
        "Disorder",
    )

    plot_df = pd.DataFrame({
        "x": z[:, 0],
        "y": z[:, 1],
        "Group": label_names,
    })

    plt.figure(figsize=(6, 5))

    sns.scatterplot(
        data=plot_df,
        x="x",
        y="y",
        hue="Group",
        s=90,
    )

    plt.title(title)

    plt.tight_layout()

    plt.savefig(
        LATENT_DIR / save_name,
        dpi=300,
    )

    plt.close()


# =============================================================================
# MASTER
# =============================================================================
def run_latent_analysis():

    logging.basicConfig(
        level=logging.INFO
    )

    embeddings, labels, _ = (
        extract_latents()
    )

    # -------------------------------------------------------------------------
    # t-SNE
    # -------------------------------------------------------------------------
    z_tsne = run_tsne(
        embeddings
    )

    plot_embeddings(
        z_tsne,
        labels,
        title="t-SNE Latent Embedding Space",
        save_name="tsne_plot.png",
    )

    # -------------------------------------------------------------------------
    # UMAP
    # -------------------------------------------------------------------------
    z_umap = run_umap(
        embeddings
    )

    if z_umap is not None:

        plot_embeddings(
            z_umap,
            labels,
            title="UMAP Latent Embedding Space",
            save_name="umap_plot.png",
        )

    logging.info(
        "✅ Latent analysis complete"
    )


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":

    run_latent_analysis()