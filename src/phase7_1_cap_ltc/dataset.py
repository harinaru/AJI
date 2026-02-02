"""
Phase 7.1 — Windowed Temporal EEG Dataset (FINAL, A?-SAFE)
----------------------------------------------------------
✔ One sample = one window
✔ Correct handling of (N, T, C, F) NPZs
✔ Variable channels per subject
✔ Preserves CAP uncertainty (A?)
✔ Reviewer-safe, training-stable
"""

from pathlib import Path
from typing import Dict, List, Tuple
import logging
import json
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from . import config


# =============================================================================
# DATASET
# =============================================================================
class TemporalEEGDataset(Dataset):
    """
    One dataset item = ONE window (T, C, F)
    """

    def __init__(self, overlap: float):
        self.overlap = overlap
        self.train_normal_only = config.TRAIN_NORMAL_ONLY

        # List of (npz_path, window_idx)
        self.samples: List[Tuple[Path, int]] = []
        self._build_index()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"[Dataset] No windows found for overlap={overlap}"
            )

        logging.info(
            f"[Dataset] Loaded {len(self.samples)} windows "
            f"(overlap={int(overlap * 100)}%)"
        )

    # ------------------------------------------------------------------
    def _build_index(self) -> None:
        """
        Build (file, window_idx) index.
        """
        overlap_dir = config.DATA_DIR / f"overlap_{int(self.overlap * 100)}"

        if not overlap_dir.exists():
            raise FileNotFoundError(
                f"[Dataset] Missing overlap directory: {overlap_dir}"
            )

        for subj_dir in sorted(overlap_dir.iterdir()):
            if not subj_dir.is_dir():
                continue

            npz_path = subj_dir / "windows.npz"
            if not npz_path.exists():
                continue

            data = np.load(npz_path)
            n_windows = data["features"].shape[0]

            # Optional: filter normals only
            subject_id = subj_dir.name.lower()
            is_normal = subject_id.startswith("n")

            if self.train_normal_only and not is_normal:
                continue

            for w in range(n_windows):
                self.samples.append((npz_path, w))

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict[str, Any]:

        npz_path, w = self.samples[idx]
        data = np.load(npz_path)

        features = torch.tensor(
            data["features"][w], dtype=torch.float32
        )  # (T, C, F)

        cap = torch.tensor(
            data["cap"][w], dtype=torch.long
        )  # (T,)

        mask = torch.tensor(
            data["mask"][w], dtype=torch.bool
        )  # (T,)

        # Safety check (A?-aware)
        if not torch.all((cap >= 0) & (cap < len(config.CAP_MAPPING))):
            raise ValueError(f"[Dataset] Invalid CAP labels in {npz_path}")

        subject_id = npz_path.parent.name
        label = 0 if subject_id.lower().startswith("n") else 1

        return {
            "features": features,      # (T, C, F)
            "cap": cap,                # (T,)
            "mask": mask,              # (T,)
            "subject_id": subject_id,  # str
            "label": label,            # int
        }


# =============================================================================
# COLLATE FUNCTION (UNCHANGED)
# =============================================================================
def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:

    max_T = max(b["features"].shape[0] for b in batch)
    max_C = max(b["features"].shape[1] for b in batch)
    F = batch[0]["features"].shape[-1]

    feats, caps, masks = [], [], []
    subject_ids, labels, lengths = [], [], []

    for b in batch:
        T, C, _ = b["features"].shape

        feat = torch.zeros(max_T, max_C, F)
        cap = torch.zeros(max_T, dtype=torch.long)
        mask = torch.zeros(max_T, dtype=torch.bool)

        feat[:T, :C] = b["features"]
        cap[:T] = b["cap"]
        mask[:T] = True

        feats.append(feat)
        caps.append(cap)
        masks.append(mask)

        subject_ids.append(b["subject_id"])
        labels.append(b["label"])
        lengths.append(T)

    return {
        "features": torch.stack(feats),   # (B, T, C, F)
        "cap": torch.stack(caps),
        "mask": torch.stack(masks),
        "labels": torch.tensor(labels),
        "subject_ids": subject_ids,
        "lengths": torch.tensor(lengths),
    }
