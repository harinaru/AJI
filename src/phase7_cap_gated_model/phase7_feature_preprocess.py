"""
Phase 7 — Feature Repair & Normalization (Standalone)

This module:
✔ Audits NaN / Inf / zero-heavy features
✔ Applies feature-wise imputation (NOT row drop)
✔ Uses robust normalization (median + IQR)
✔ Preserves CAP semantics (zeros where meaningful)
✔ Writes a cleaned CSV for training
"""

from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd


# --------------------------------------------------
# Robust statistics
# --------------------------------------------------

def robust_zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Median + IQR scaling (robust to outliers)
    """
    median = np.nanmedian(x)
    q1 = float(np.nanpercentile(x, 25))
    q3 = float(np.nanpercentile(x, 75))
    iqr = max(float(q3 - q1), eps)
    return (x - median) / iqr


# --------------------------------------------------
# Main preprocessing function
# --------------------------------------------------

def preprocess_channel_csv(
    input_csv: Path,
    output_csv: Path,
    feature_cols: List[str],
    zero_is_meaningful: List[str],
    verbose: bool = True,
) -> None:
    """
    Repairs + normalizes features in-place and saves a clean CSV.
    """

    df = pd.read_csv(input_csv)

    if verbose:
        print(f"\n🧪 Preprocessing: {input_csv.name}")
        print(f"Rows before: {len(df)}")

    for col in feature_cols:
        x = df[col].to_numpy(dtype=np.float64)

        # ---------------- NaN / Inf handling ----------------
        bad_mask = ~np.isfinite(x)

        if bad_mask.any():
            if col in zero_is_meaningful:
                # CAP ratios, counts → zero is valid
                x[bad_mask] = 0.0
            else:
                # Replace with median (robust)
                median = np.nanmedian(x)
                x[bad_mask] = median

        # ---------------- Zero-heavy logic ----------------
        zero_ratio = (x == 0).mean()

        if zero_ratio > 0.9:
            # Almost constant → leave as-is, but center
            x = x - np.mean(x)
        else:
            # Proper robust scaling
            x = robust_zscore(x)

        df[col] = x.astype(np.float32)

        if verbose:
            print(
                f"✔ {col:30s} | zero_ratio={zero_ratio:.2f}"
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    if verbose:
        print(f"✅ Saved cleaned file → {output_csv}")
        print(f"Rows after: {len(df)}")
