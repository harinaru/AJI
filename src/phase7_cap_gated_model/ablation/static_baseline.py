"""
ABLATION — STATIC BASELINE
Gaussian normality (no temporal memory)

Question answered:
❓ Why not use static EEG statistics?
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from typing import List, cast

from ..config import Config


def main():
    print("\n" + "=" * 80)
    print(" BASELINE: STATIC GAUSSIAN NORMALITY ")
    print("=" * 80)

    config = Config()
    results: List[dict] = []

    for channel in config.CHANNELS:
        print(f"\n📊 Channel: {channel}")

        csv_path = config.PHASE4_DATA_DIR / f"{channel}.csv"

        # Explicit DataFrame typing (important for Pyright)
        df: pd.DataFrame = pd.read_csv(csv_path)

        # -------------------------------
        # Select numeric EEG features ONLY
        # -------------------------------
        numeric_df: pd.DataFrame = df.select_dtypes(include=[np.number])

        feature_cols = [
            c for c in numeric_df.columns
            if c not in {"label", "epoch_idx"}
        ]

        # -------------------------------
        # Controls only (fit normal model)
        # -------------------------------
        ctrl_df: pd.DataFrame = cast(pd.DataFrame, df[df["label"] == 0].reset_index(drop=True))

        X_ctrl: np.ndarray = (
            ctrl_df[feature_cols]
            .to_numpy(dtype=np.float64)
        )

        # Mean & covariance
        mu: np.ndarray = X_ctrl.mean(axis=0)
        cov: np.ndarray = np.cov(X_ctrl, rowvar=False)
        inv_cov: np.ndarray = np.linalg.pinv(cov)

        # -------------------------------
        # Subject-wise scoring
        # -------------------------------
        for subject_id, group in df.groupby("subject"):
            g: pd.DataFrame = group.copy()

            X: np.ndarray = (
                g[feature_cols]
                .to_numpy(dtype=np.float64)
            )

            # Mahalanobis distance per epoch
            dists = np.array([
                mahalanobis(x, mu, inv_cov) for x in X
            ])

            results.append({
                "channel": channel,
                "subject": subject_id,
                "label": int(g["label"].iloc[0]),
                "p90_mahalanobis": float(np.percentile(dists, 90))
            })

    # -------------------------------
    # Save results
    # -------------------------------
    out_df = pd.DataFrame(results)
    out_path = config.OUTPUT_DIR / "logs" / "static_baseline_results.csv"
    out_df.to_csv(out_path, index=False)

    print(f"\n📁 Static baseline saved to: {out_path}")
    print("❗ EXPECTED: poor burst sensitivity")


if __name__ == "__main__":
    main()
