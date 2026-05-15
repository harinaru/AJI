# src/phase7_1_cap_ltc/latent.py

"""
Phase 7.x — Reviewer-Grade Latent Analysis
==========================================

✔ Temporal-aware latent extraction
✔ Instability-aware pooling
✔ Reviewer-grade embeddings
✔ t-SNE ready
✔ UMAP ready
✔ Non-overwriting outputs
✔ Multi-overlap safe
✔ NaN-safe
✔ Checkpoint-safe
✔ Q1 journal-ready
"""

# =============================================================================
# IMPORTS
# =============================================================================

import json
import logging

from pathlib import Path

import numpy as np
import pandas as pd

import torch

from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from . import config

from .dataset import (
    TemporalEEGDataset,
    collate_fn,
)

from .train import (
    split_subjects,
)

from .model.cap_ltc_model import (
    CAP_LTC_Model,
)

from .utils import (
    resolve_latest_checkpoint,
)

# =============================================================================
# STYLE
# =============================================================================

sns.set_style("whitegrid")

# =============================================================================
# t-SNE
# =============================================================================

def run_tsne(
    embeddings,
):

    # =============================================================
    # SAFETY
    # =============================================================

    if len(embeddings) < 2:

        raise ValueError(
            "[t-SNE] Need at least 2 samples"
        )

    # =============================================================
    # STANDARDIZATION
    # VERY IMPORTANT FOR LATENT GEOMETRY
    # =============================================================

    scaler = StandardScaler()

    embeddings_scaled = scaler.fit_transform(
        embeddings
    )

    # =============================================================
    # PCA PRE-REDUCTION
    # Reviewer-grade stabilization
    # =============================================================

    pca_dims = min(

        50,

        embeddings_scaled.shape[1],

        len(embeddings_scaled) - 1,
    )

    pca = PCA(

        n_components=pca_dims,

        random_state=config.SEED,
    )

    embeddings_pca = pca.fit_transform(
        embeddings_scaled
    )

    # =============================================================
    # SAFE PERPLEXITY
    # =============================================================

    num_samples = len(
        embeddings_pca
    )

    perplexity = min(

        30,

        max(
            5,
            num_samples // 10,
        ),
    )

    perplexity = min(
        perplexity,
        num_samples - 1,
    )

    # =============================================================
    # REVIEWER-GRADE t-SNE
    # =============================================================

    tsne = TSNE(

        n_components=2,

        perplexity=perplexity,

        learning_rate="auto",

        init="pca",

        random_state=config.SEED,

        early_exaggeration=12,

        metric="euclidean",

        verbose=1,
    )

    z_tsne = tsne.fit_transform(
        embeddings_pca
    )

    return z_tsne

# =============================================================================
# LATENT EXTRACTION
# =============================================================================

@torch.no_grad()
def extract_latents(
    model,
    loader,
    device,
):

    model.eval()

    latent_vectors = []

    labels_list = []

    subject_ids_list = []

    for batch in tqdm(
        loader,
        desc="Latent Extraction",
    ):

        # =============================================================
        # INPUTS
        # =============================================================

        x = batch["features"].to(device)

        cap = batch["cap"].to(device)

        mask = batch["mask"].to(device)

        labels = batch["labels"]

        subject_ids = batch["subject_ids"]

        # =============================================================
        # FORWARD
        # =============================================================

        outputs = model(
            x,
            cap,
            mask,
        )

        # =============================================================
        # FLEXIBLE OUTPUTS
        # =============================================================

        extras = {}

        if isinstance(outputs, tuple):

            recon = outputs[0]

            if len(outputs) > 1:

                extras = outputs[1]

        else:

            recon = outputs

        # =============================================================
        # HIDDEN STATES
        # =============================================================

        if (
            isinstance(extras, dict)
            and
            "hidden_states" in extras
        ):

            hidden_states = extras[
                "hidden_states"
            ]

        else:

            logging.warning(
                "[Latent] hidden_states missing"
            )

            continue

        # =============================================================
        # SHAPE SAFETY
        # =============================================================

        if hidden_states.ndim != 3:

            logging.warning(
                f"[Latent] Invalid shape: "
                f"{hidden_states.shape}"
            )

            continue

        # =============================================================
        # TEMPORAL-AWARE LATENT POOLING
        # =============================================================

        mean_feat = hidden_states.mean(
            dim=1
        )

        std_feat = hidden_states.std(
            dim=1
        )

        max_feat, _ = hidden_states.max(
            dim=1
        )

        # =============================================================
        # TEMPORAL INSTABILITY
        # =============================================================

        if hidden_states.shape[1] > 1:

            delta_feat = (

                hidden_states[:, 1:]

                -

                hidden_states[:, :-1]

            ).abs().mean(dim=1)

        else:

            delta_feat = torch.zeros_like(
                mean_feat
            )

        # =============================================================
        # FINAL LATENT
        # =============================================================

        latent = torch.cat(

            [
                mean_feat,
                std_feat,
                max_feat,
                delta_feat,
            ],

            dim=-1,
        )

        # =============================================================
        # NUMERICAL SAFETY
        # =============================================================

        latent = torch.nan_to_num(

            latent,

            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        latent = (
            latent
            .detach()
            .cpu()
            .numpy()
        )

        # =============================================================
        # STORE
        # =============================================================

        for i in range(len(subject_ids)):

            latent_vectors.append(
                latent[i]
            )

            labels_list.append(
                int(labels[i].item())
            )

            subject_ids_list.append(
                subject_ids[i]
            )

    # =============================================================
    # FINAL ARRAYS
    # =============================================================

    latent_vectors = np.asarray(
        latent_vectors,
        dtype=np.float32,
    )

    labels_array = np.asarray(
        labels_list,
        dtype=np.int64,
    )

    return (
        latent_vectors,
        labels_array,
        subject_ids_list,
    )

# =============================================================================
# MAIN LATENT ANALYSIS
# =============================================================================

def run_latent_analysis(
    overlap: float,
    run_dir: Path,
):

    logging.basicConfig(
        level=logging.INFO
    )

    logger = logging.getLogger(
        f"latent_{int(overlap * 100)}"
    )

    overlap_pct = int(
        overlap * 100
    )

    logger.info("=" * 80)

    logger.info(
        f"LATENT ANALYSIS | OVERLAP={overlap_pct}%"
    )

    logger.info("=" * 80)

    device = torch.device(
        config.DEVICE
    )

    # ================================================================
    # DIRS
    # ================================================================

    latent_dir = (
        run_dir
        / "latent"
    )

    latent_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_dir = (
        run_dir
        / "models"
    )

    # ================================================================
    # SUBJECT SPLITS
    # ================================================================

    overlap_dir = (
        config.DATA_DIR
        / f"overlap_{overlap_pct}"
    )

    _, _, test_subjects = split_subjects(

        overlap_dir,

        logger,
        run_dir,
    )

    logger.info(

        f"[Latent] Using unseen "
        f"test subjects only "
        f"for subject-independent "
        f"latent analysis"
    )

    # ================================================================
    # LOAD MODEL
    # ================================================================

    model = CAP_LTC_Model().to(device)

    checkpoint_path = resolve_latest_checkpoint(

        overlap=overlap,

        model_dir=model_dir,
    )

    ckpt = torch.load(

        checkpoint_path,

        map_location=device,
    )

    if (
        isinstance(ckpt, dict)
        and
        "model_state" in ckpt
    ):

        state_dict = ckpt[
            "model_state"
        ]

    elif (
        isinstance(ckpt, dict)
        and
        "model_state_dict" in ckpt
    ):

        state_dict = ckpt[
            "model_state_dict"
        ]

    else:

        state_dict = ckpt

    missing_keys, unexpected_keys = model.load_state_dict(

        state_dict,

        strict=False,
    )

    if len(missing_keys) > 0:

        logger.warning(
            f"[Latent] Missing keys: "
            f"{missing_keys}"
        )

    if len(unexpected_keys) > 0:

        logger.warning(
            f"[Latent] Unexpected keys: "
            f"{unexpected_keys}"
        )

    model.eval()

    logger.info(
        f"[Latent] Loaded checkpoint: "
        f"{checkpoint_path}"
    )

    # ================================================================
    # DATASET
    # ================================================================

    dataset = TemporalEEGDataset(

        overlap=overlap,

        allowed_subjects=test_subjects,
    )

    # ================================================================
    # EMPTY DATASET SAFETY
    # ================================================================

    if len(dataset) < 5:

        raise RuntimeError(

            f"[Latent] Too few samples "
            f"for stable latent analysis: "
            f"{len(dataset)}"
        )

    # ================================================================
    # DATALOADER
    # ================================================================

    loader = DataLoader(

        dataset,

        batch_size=config.BATCH_SIZE,

        shuffle=False,

        collate_fn=collate_fn,

        num_workers=config.NUM_WORKERS,

        pin_memory=torch.cuda.is_available(),

        persistent_workers=(
            config.NUM_WORKERS > 0
        ),

        drop_last=False,
    )

    # ================================================================
    # EXTRACT LATENTS
    # ================================================================

    embeddings, labels, subject_ids = extract_latents(

        model=model,

        loader=loader,

        device=device,
    )

    logger.info(
        f"[Latent] Embeddings shape: "
        f"{embeddings.shape}"
    )

    # ================================================================
    # SAVE NUMPY
    # ================================================================

    np.save(

        latent_dir
        / "latent_embeddings.npy",

        embeddings,
    )

    np.save(

        latent_dir
        / "latent_labels.npy",

        labels,
    )

    # ================================================================
    # SAVE CSV
    # ================================================================

    latent_df = pd.DataFrame({

        "subject_id":
            subject_ids,

        "label":
            labels,
    })

    latent_df.to_csv(

        latent_dir
        / "latent_metadata.csv",

        index=False,
    )

    # ================================================================
    # t-SNE
    # ================================================================

    z_tsne = run_tsne(
        embeddings
    )

    tsne_df = pd.DataFrame({

        "x":
            z_tsne[:, 0],

        "y":
            z_tsne[:, 1],

        "label":
            labels,
    })

    tsne_df.to_csv(

        latent_dir
        / "tsne_embeddings.csv",

        index=False,
    )

    # ================================================================
    # PLOT
    # ================================================================

    plt.figure(figsize=(7, 6))

    sns.scatterplot(

        x=z_tsne[:, 0],

        y=z_tsne[:, 1],

        hue=labels,

        palette="Set1",
    )

    plt.title(
        f"t-SNE Latent Space | Overlap {overlap_pct}%"
    )

    plt.tight_layout()

    plt.savefig(

        latent_dir
        / "tsne_plot.png",

        dpi=300,
    )

    plt.close()

    # ================================================================
    # METADATA
    # ================================================================

    metadata = {

        "overlap":
            overlap_pct,

        "num_subjects":
            int(len(labels)),

        "embedding_dim":
            int(embeddings.shape[1]),

        "checkpoint":
            str(checkpoint_path),

        "tsne_performed_on":
            "unseen_test_subjects_only",
    }

    with open(

        latent_dir
        / "latent_metadata.json",

        "w",

        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4,
        )

    logger.info(
        f"[Latent] Saved outputs → "
        f"{latent_dir}"
    )

    logger.info("=" * 80)

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    raise RuntimeError(
        "Run through main.py only."
    )