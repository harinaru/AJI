"""
Phase 7.0a — Phase-2 CSV → Tensor Preparation (FINAL, FIXED)
------------------------------------------------------------
✔ Uses *_channels_fixed.csv
✔ Infers REAL EEG channel names (Fp1, C3, O2, etc.)
✔ Variable channels per subject
✔ Missing channels padded with zeros
✔ Channel-wise normalization
✔ Explicit CAP uncertainty handling (A?)
✔ Outputs NPZ for windowing & training
"""

from pathlib import Path
from typing import List, Union
import json
import logging

import numpy as np
import pandas as pd

from .. import config
from .feature_selector import select_feature_columns
from .normalizer import normalize_features

# =============================================================================
# HELPERS
# =============================================================================
def infer_channels_from_columns(columns: List[str]) -> List[str]:
    """
    Infer EEG channel names from column prefixes.
    Example:
        Fp1_mean → Fp1
        C3_entropy → C3
    """
    channels = set()

    for col in columns:
        if "_" not in col:
            continue
        prefix = col.split("_")[0]
        if any(ch.isdigit() for ch in prefix):
            channels.add(prefix)

    return sorted(channels)


def encode_cap_labels(
    series: Union[pd.Series, pd.DataFrame]
) -> np.ndarray:
    """
    Map CAP labels: A1 / A2 / A3 / A? / NaN → int
    SAFE for missing, malformed, or unknown values.
    """

    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]

    def map_one(x):
        if pd.isna(x):
            return config.CAP_MAPPING["A?"]
        x = str(x).strip()
        return config.CAP_MAPPING.get(x, config.CAP_MAPPING["A?"])

    return series.apply(map_one).to_numpy(dtype=np.int64)

# =============================================================================
# SUBJECT PROCESSING
# =============================================================================
def prepare_subject_csv(
    csv_path: Path,
    out_dir: Path,
    subject_id: str,
) -> None:
    logging.info(f"[Prepare] {subject_id} ← {csv_path.name}")

    df = pd.read_csv(csv_path)

    # ------------------------------------------------------------------
    # CHANNEL INFERENCE
    # ------------------------------------------------------------------
    channel_names = infer_channels_from_columns(df.columns.tolist())

    if not channel_names:
        logging.warning(f"[Prepare] No channels found → skipped: {subject_id}")
        return

    logging.info(
        f"[Prepare] {subject_id} | Channels detected: {channel_names}"
    )

    # ------------------------------------------------------------------
    # FEATURE EXTRACTION
    # ------------------------------------------------------------------
    T = len(df)
    C = len(channel_names)
    F = config.FEATURE_DIM

    features = np.zeros((T, C, F), dtype=np.float32)

    for ci, ch in enumerate(channel_names):
        cols = select_feature_columns(df, ch)
        if not cols:
            continue

        vals = df[cols].to_numpy(dtype=np.float32)

        if vals.shape[1] > F:
            vals = vals[:, :F]
        elif vals.shape[1] < F:
            vals = np.pad(vals, ((0, 0), (0, F - vals.shape[1])))

        features[:, ci, :] = vals

    # ------------------------------------------------------------------
    # NORMALIZATION
    # ------------------------------------------------------------------
    if config.NORMALIZE_FEATURES:
        features = normalize_features(features)

    # ------------------------------------------------------------------
    # CAP LABELS (UNCERTAINTY SAFE)
    # ------------------------------------------------------------------
    if "cap_subtype" in df.columns:
        cap = encode_cap_labels(df["cap_subtype"])
    else:
        cap = np.full(T, config.CAP_MAPPING["NC"], dtype=np.int64)

    mask = np.ones(T, dtype=bool)

    # ------------------------------------------------------------------
    # SAVE OUTPUTS
    # ------------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_dir / "features.npz",
        features=features,
        cap=cap,
        mask=mask,
        channel_names=np.array(channel_names),
    )

    with open(out_dir / "meta.json", "w") as f:
        json.dump(
            {
                "subject_id": subject_id,
                "source_csv": csv_path.name,
                "n_channels": C,
                "channel_names": channel_names,
                "cap_states": config.CAP_MAPPING,
                "normalized": config.NORMALIZE_FEATURES,
            },
            f,
            indent=4,
        )

    logging.info(f"[Prepare] Saved → {out_dir}")

# =============================================================================
# BATCH DRIVER
# =============================================================================
def prepare_all_subjects(
    phase2_root: Path = config.PHASE2_FIXED_DIR,
    out_root: Path = config.PREPARED_DIR,
) -> None:
    """
    Process all subjects from phase2_features_fixed
    """
    logging.basicConfig(level=logging.INFO)

    if not phase2_root.exists():
        raise FileNotFoundError(
            f"Missing Phase-2 directory: {phase2_root}"
        )

    out_root.mkdir(parents=True, exist_ok=True)

    for subj_dir in sorted(phase2_root.iterdir()):
        if not subj_dir.is_dir():
            continue

        subject_id = subj_dir.name
        csv_files = list(subj_dir.glob("*_channels_fixed.csv"))

        if not csv_files:
            logging.warning(f"[Prepare] No CSVs for {subject_id}")
            continue

        prepare_subject_csv(
            csv_path=csv_files[0],
            out_dir=out_root / subject_id,
            subject_id=subject_id,
        )

    logging.info("[Prepare] Phase-2 → Phase-7 preparation COMPLETE")
