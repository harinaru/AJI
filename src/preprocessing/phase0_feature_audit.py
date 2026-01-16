"""
PHASE 0 — EEG FEATURE AUDIT & PREPROCESSING (RESEARCH-GRADE)

✔ Ignores CAP phase labels (A1/A2/A3/B)
✔ Normalizes ONLY numeric EEG features
✔ Preserves cap_present as control signal
✔ Produces audit reports (reviewer-safe)
✔ No leakage, no coercion, no assumptions
"""

from pathlib import Path
import pandas as pd
import numpy as np
import json

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

RAW_DATA_DIR = Path("results/phase4_channel_datasets")
OUTPUT_DATA_DIR = Path("results/phase4_features_cleaned")
REPORT_DIR = Path("results/feature_audit")

ZERO_STRUCTURAL_THRESHOLD = 0.6
MIN_VARIANCE = 1e-8
CLIP_VALUE = 10.0

OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# CORE FUNCTION
# --------------------------------------------------

def audit_and_preprocess_csv(csv_path: Path) -> None:
    print(f"\n🔍 Auditing: {csv_path.name}")

    df = pd.read_csv(csv_path)

    # ---------------- META / CONTROL COLUMNS ----------------
    META_COLS = {"subject", "label", "epoch_idx"}
    CAP_CONTROL_COLS = {"cap_present"}          # ✅ keep
    CAP_PHASE_COLS = {"cap_phase", "cap_type"}  # ❌ ignore if present

    # ---------------- FEATURE SELECTION ----------------
    feature_cols = []
    ignored_cols = []

    for col in df.columns:
        if col in META_COLS or col in CAP_CONTROL_COLS:
            continue

        # Explicitly ignore CAP phase labels or any non-numeric column
        if col in CAP_PHASE_COLS or not pd.api.types.is_numeric_dtype(df[col]):
            ignored_cols.append(col)
        else:
            feature_cols.append(col)

    if not feature_cols:
        raise RuntimeError(f"No numeric EEG features found in {csv_path.name}")

    # ---------------- AUDIT REPORT ----------------
    report = {
        "file": csv_path.name,
        "num_rows": len(df),
        "num_features_used": len(feature_cols),
        "ignored_columns": ignored_cols,
        "features": {}
    }
    base_cols = [c for c in df.columns if c in (META_COLS | CAP_CONTROL_COLS)]
    df_clean = df[base_cols].copy()
   
    # ---------------- FEATURE PROCESSING ----------------
    for col in feature_cols:
        x = df[col].to_numpy(dtype=np.float64)

        n = len(x)
        zero_ratio = float(np.mean(x == 0))
        nan_ratio = float(np.mean(np.isnan(x)))
        inf_ratio = float(np.mean(np.isinf(x)))

        valid = ~(np.isnan(x) | np.isinf(x))

        stats = {
            "zero_ratio": round(zero_ratio, 4),
            "nan_ratio": round(nan_ratio, 4),
            "inf_ratio": round(inf_ratio, 4),
            "mean": float(np.nanmean(x)),
            "std": float(np.nanstd(x)),
            "min": float(np.nanmin(x)),
            "max": float(np.nanmax(x)),
            "median": float(np.nanmedian(x)),
        }

        # ---------- SAFE FILLING ----------
        x_filled = x.copy()

        if valid.sum() == 0:
            x_filled[:] = 0.0
            fill_action = "all_invalid_zeroed"
        else:
            median = np.nanmedian(x[valid])

            if zero_ratio < ZERO_STRUCTURAL_THRESHOLD:
                x_filled[x == 0] = median
                fill_action = "zeros_replaced"
            else:
                fill_action = "zeros_preserved"

            x_filled[~valid] = median

        # ---------- ROBUST NORMALIZATION ----------
        q1 = np.percentile(x_filled, 25)
        q3 = np.percentile(x_filled, 75)
        iqr = q3 - q1

        if iqr < MIN_VARIANCE:
            x_norm = x_filled - np.median(x_filled)
            norm_method = "demean_only"
        else:
            x_norm = (x_filled - np.median(x_filled)) / iqr
            norm_method = "robust_zscore"

        x_norm = np.clip(x_norm, -CLIP_VALUE, CLIP_VALUE)

        df_clean[col] = x_norm.astype(np.float32)

        report["features"][col] = {
            **stats,
            "fill_action": fill_action,
            "normalization": norm_method
        }

    # ---------------- SAVE OUTPUTS ----------------
    df_clean.to_csv(OUTPUT_DATA_DIR / csv_path.name, index=False)

    with open(REPORT_DIR / f"{csv_path.stem}_audit.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"✅ Cleaned CSV saved → {OUTPUT_DATA_DIR / csv_path.name}")
    print(f"📄 Audit report → {REPORT_DIR / f'{csv_path.stem}_audit.json'}")


# --------------------------------------------------
# ENTRYPOINT
# --------------------------------------------------

def main():
    print(f"\n🔍 Looking for CSVs in: {RAW_DATA_DIR.resolve()}")

    csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))

    if not csv_files:
        raise RuntimeError("❌ No CSV files found")

    for csv in csv_files:
        audit_and_preprocess_csv(csv)

    print("\n🎉 PHASE 0 FEATURE AUDIT & PREPROCESSING COMPLETE")


if __name__ == "__main__":
    main()
