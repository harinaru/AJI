from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from .normality_model import GaussianNormalityModel

# ================= CONFIG ================= #

CHANNELS = ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"]

PHASE4_DIR = Path("results/phase4_channel_datasets")
PHASE5_DIR = Path("results/phase5_feature_selection")

OUT_DIR = Path("results/phase6B_tail")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL = "label"
SUBJECT_COL = "subject"

# ========================================== #


def clean_features(df, feature_cols):
    """Median-impute + remove invalid rows"""
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    valid = ~X.isna().any(axis=1)
    return X.loc[valid].to_numpy(), valid


def main():
    print("\n==============================")
    print(" PHASE 6B — TAIL UNCERTAINTY ")
    print("==============================\n")

    stats_summary = {}

    for ch in CHANNELS:
        print(f"▶ Phase-6B | {ch}")

        df = pd.read_csv(PHASE4_DIR / f"{ch}.csv")
        feats = pd.read_csv(
            PHASE5_DIR / f"{ch}_selected_features.csv"
        )["selected_feature"].tolist()

        # -------- Train on CONTROLS ONLY -------- #
        ctrl_df = df[df[LABEL_COL] == 0].reset_index(drop=True)
        X_ctrl, mask = clean_features(ctrl_df, feats)
        ctrl_df = ctrl_df.loc[mask]

        model = GaussianNormalityModel()
        model.fit(X_ctrl)

        # -------- Score ALL epochs -------- #
        X_all, mask_all = clean_features(df, feats)
        df = df.loc[mask_all].reset_index(drop=True)
        df["epoch_uncertainty"] = model.score(X_all)

        # -------- SUBJECT-LEVEL TAIL METRICS -------- #
        subj = (
            df.groupby(SUBJECT_COL)
            .agg(
                uncertainty_mean=("epoch_uncertainty", "mean"),
                uncertainty_p90=("epoch_uncertainty",
                                 lambda x: np.percentile(x, 90)),
                uncertainty_p95=("epoch_uncertainty",
                                 lambda x: np.percentile(x, 95)),
                uncertainty_max=("epoch_uncertainty", "max"),
                uncertainty_std=("epoch_uncertainty", "std"),
                label=(LABEL_COL, "first"),
            )
            .reset_index()
        )

        out_csv = OUT_DIR / f"tail_uncertainty_{ch}.csv"
        subj.to_csv(out_csv, index=False)
        print(f"  ✓ Saved {out_csv.name}")

        # -------- Statistics (p90 is PRIMARY) -------- #
        ctrl = subj[subj["label"] == 0]["uncertainty_p90"].values
        rare = subj[subj["label"] != 0]["uncertainty_p90"].values

        if len(rare) > 0:
            u, p = mannwhitneyu(ctrl, rare, alternative="less")
            stats_summary[ch] = {
                "control_p90_mean": float(ctrl.mean()),
                "rare_p90_mean": float(rare.mean()),
                "p_value": float(p),
            }
            print(f"  ✓ p90 test p = {p:.4e}")

    with open(OUT_DIR / "phase6B_stats.json", "w") as f:
        json.dump(stats_summary, f, indent=2)

    print("\n✅ Phase-6B completed successfully\n")


if __name__ == "__main__":
    main()
