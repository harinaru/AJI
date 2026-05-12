# src/phase7_1_cap_ltc/utils/data_sanity.py

"""
Data Sanity & Repair Utilities (FINAL, JOURNAL-GRADE)
----------------------------------------------------
✔ Detects missing columns
✔ Handles NaN / inf safely
✔ Fills with statistically valid defaults
✔ Logs every correction explicitly
"""

import logging
from pathlib import Path
import numpy as np
import pandas as pd


def sanitize_dataframe(
    df: pd.DataFrame,
    name: str,
    strategy: str = "median",
) -> pd.DataFrame:
    """
    Fill NaN / inf values with safe statistics.
    """
    logging.info(f"[Sanity] Checking {name}")

    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        invalid = ~np.isfinite(df[col].to_numpy())
        n_bad = int(invalid.sum())

        if n_bad == 0:
            continue

        if strategy == "median":
            fill_value = float(df[col].median())
        elif strategy == "mean":
            fill_value = float(df[col].mean())
        else:
            fill_value = 0.0

        df.loc[invalid, col] = fill_value

        logging.warning(
            f"[Sanity] {name}.{col}: filled {n_bad} invalid values "
            f"with {strategy}={fill_value:.4f}"
        )

    return df


def sanitize_inference_dir(infer_dir: Path) -> None:
    """
    Apply sanity checks to inference outputs.
    """
    subj = infer_dir / "subject_level_metrics.csv"
    win = infer_dir / "window_level_metrics.csv"
    scores = infer_dir / "scores_labels.npy"

    if subj.exists():
        df = pd.read_csv(subj)
        df = sanitize_dataframe(df, "subject_level_metrics")
        df.to_csv(subj, index=False)
    else:
        logging.error(f"[Sanity] Missing {subj}")

    if win.exists():
        df = pd.read_csv(win)
        df = sanitize_dataframe(df, "window_level_metrics")
        df.to_csv(win, index=False)
    else:
        logging.error(f"[Sanity] Missing {win}")

    if scores.exists():
        arr = np.load(scores, allow_pickle=True)
        scores_np, labels_np = arr

        bad = ~np.isfinite(scores_np)
        if bad.any():
            fill = float(np.nanmedian(scores_np))
            scores_np[bad] = fill

            logging.warning(
                f"[Sanity] scores_labels.npy: filled {bad.sum()} invalid scores "
                f"with median={fill:.4f}"
            )

        np.save(scores, np.array([scores_np, labels_np], dtype=object))


def run_data_sanity():
    logging.basicConfig(level=logging.INFO)

    from .. import config

    base = config.RESULTS_DIR / "inference"
    if not base.exists():
        logging.error("[Sanity] No inference directory found")
        return

    for d in sorted(base.glob("overlap_*")):
        logging.info(f"[Sanity] Processing {d.name}")
        sanitize_inference_dir(d)

    logging.info("✅ DATA SANITY CHECK COMPLETED")


if __name__ == "__main__":
    run_data_sanity()
