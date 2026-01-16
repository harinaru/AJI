"""
Phase 7 — Temporal EEG Dataset with CAP alignment

✔ Subject-wise sequences
✔ Temporal order preserved
✔ CAP-aligned
✔ Variable-length safe
✔ Padding + masking
✔ CONTROL-ONLY FEATURE NORMALIZATION (CRITICAL FIX)
✔ Journal-grade & stable
"""

from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class TemporalEEGDataset(Dataset):
    """
    One dataset sample = one subject (variable-length EEG sequence)
    """

    MIN_VALID_EPOCHS = 1

    def __init__(
        self,
        csv_path: Path,
        feature_names: List[str],
        allowed_subjects: List[str] | None = None,
        verbose: bool = True,
    ):
        self.csv_path = Path(csv_path)
        self.feature_names = feature_names
        self.verbose = verbose

        df = pd.read_csv(self.csv_path)

        if allowed_subjects is not None:
            df = df.loc[df["subject"].isin(allowed_subjects)].copy()

        required_cols = set(feature_names) | {"subject", "label"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if "cap_present" not in df.columns:
            df["cap_present"] = 0

        self.samples: List[Dict] = []

        # --------------------------------------------------
        # Build subject-wise sequences
        # --------------------------------------------------
        for subject_id, group in df.groupby("subject"):
            g = group.copy()

            if "epoch_idx" in g.columns:
                g = g.sort_values("epoch_idx")

            X = g[self.feature_names].to_numpy(dtype=np.float32)
            cap = g["cap_present"].to_numpy(dtype=np.float32)
            label = int(g["label"].iloc[0])

            valid_mask = (
                ~np.isnan(X).any(axis=1)
                & ~np.isinf(X).any(axis=1)
            )

            if valid_mask.sum() < self.MIN_VALID_EPOCHS:
                if self.verbose:
                    print(
                        f"⚠️ Skipping {subject_id} "
                        f"({valid_mask.sum()}/{len(X)} valid epochs)"
                    )
                continue

            X = X[valid_mask]
            cap = cap[valid_mask]

            self.samples.append(
                {
                    "subject_id": subject_id,
                    "features": torch.from_numpy(X),    # [T, F]
                    "cap_flags": torch.from_numpy(cap), # [T]
                    "length": X.shape[0],
                    "label": label,
                }
            )

        if not self.samples:
            if self.verbose:
                print(f"\n📊 Dataset loaded: {self.csv_path.name}")
                print("⚠️ No valid subjects in this channel")
            return

        # ==================================================
        # 🔥 CRITICAL FIX — CONTROL-ONLY NORMALIZATION
        # ==================================================

        control_features = [
            s["features"] for s in self.samples if s["label"] == 0
        ]

        if len(control_features) == 0:
            raise RuntimeError("No control subjects found for normalization.")

        all_control_feats = torch.cat(control_features, dim=0)  # [N, F]

        self.feature_mean = all_control_feats.mean(dim=0)
        self.feature_std = all_control_feats.std(dim=0)
        self.feature_std[self.feature_std < 1e-6] = 1.0

        # Apply normalization to ALL samples
        for s in self.samples:
            s["features"] = (s["features"] - self.feature_mean) / self.feature_std

        if self.verbose:
            self._print_summary()

    def _print_summary(self) -> None:
        lengths = [s["length"] for s in self.samples]
        labels = [s["label"] for s in self.samples]

        print(f"\n📊 Dataset loaded: {self.csv_path.name}")
        print(f"Subjects       : {len(self.samples)}")
        print(f"Controls       : {labels.count(0)}")
        print(f"Rare disorders : {labels.count(1)}")
        print(
            f"Seq length     : min={min(lengths)}, "
            f"max={max(lengths)}, mean={np.mean(lengths):.1f}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return (
            s["features"],     # [T, F] (NORMALIZED)
            s["cap_flags"],    # [T]
            s["length"],       # int
            s["label"],        # int
            s["subject_id"],   # str
        )


# ======================================================
# COLLATE FUNCTION
# ======================================================

def collate_fn(
    batch: List[
        Tuple[
            torch.Tensor,  # features [T, F]
            torch.Tensor,  # cap_flags [T]
            int,
            int,
            str,
        ]
    ]
) -> Dict[str, torch.Tensor | List[str]]:

    features_t, cap_t, lengths_t, labels_t, subject_ids = zip(*batch)

    features_pad = pad_sequence(
        list(features_t), batch_first=True, padding_value=0.0
    )

    cap_pad = pad_sequence(
        list(cap_t), batch_first=True, padding_value=0.0
    )

    lengths_tensor = torch.tensor(lengths_t, dtype=torch.long)
    T_max = features_pad.size(1)

    mask = torch.arange(T_max)[None, :] < lengths_tensor[:, None]

    return {
        "features": features_pad,
        "cap_flags": cap_pad,
        "mask": mask,
        "lengths": lengths_tensor,
        "labels": torch.tensor(labels_t),
        "subject_ids": list(subject_ids),
    }
