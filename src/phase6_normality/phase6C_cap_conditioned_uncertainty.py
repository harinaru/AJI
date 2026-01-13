from pathlib import Path
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import mannwhitneyu

from .normality_model import GaussianNormalityModel

# ---------------- CONFIG ---------------- #

CHANNELS = ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"]

PHASE4_DIR = Path("results/phase4_channel_datasets")
PHASE5_DIR = Path("results/phase5_feature_selection")
OUT_DIR = Path("results/phase6_normality")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL = "label"
SUBJECT_COL = "subject"
CAP_COL = "cap_present"

# --------------------------------------- #


def clean_features(df, feature_cols):
    X = df[feature_cols].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X = X.fillna(X.median())
    valid_mask = ~X.isna().any(axis=1)
    return X.loc[valid_mask].to_numpy(), valid_mask


def main():
    print("\n==============================")
    print(" PHASE 6C — CAP-CONDITIONED UNCERTAINTY ")
    print("==============================\n")

    cap_stats = {}

    for channel in CHANNELS:
        print(f"\n▶ Phase-6C | {channel}")

        # Load data
        df = pd.read_csv(PHASE4_DIR / f"{channel}.csv")
        feats = pd.read_csv(
            PHASE5_DIR / f"{channel}_selected_features.csv"
        )["selected_feature"].tolist()

        # ---------------- TRAIN NORMALITY MODEL (CONTROL ONLY) ---------------- #
        control_df = df[df[LABEL_COL] == 0].reset_index(drop=True)
        X_control, mask_ctrl = clean_features(control_df, feats)
        control_df = control_df.loc[mask_ctrl]

        model = GaussianNormalityModel()
        model.fit(X_control)

        # ---------------- SCORE ALL EPOCHS ---------------- #
        X_all, mask_all = clean_features(df, feats)
        df = df.loc[mask_all].reset_index(drop=True)
        df["epoch_uncertainty"] = model.score(X_all)

        # ---------------- CAP-CONDITIONED TAIL METRICS ---------------- #
        subject_metrics = []

        for subject, g in df.groupby(SUBJECT_COL):
            cap_epochs = g[g[CAP_COL] == 1]["epoch_uncertainty"]
            noncap_epochs = g[g[CAP_COL] == 0]["epoch_uncertainty"]

            if len(cap_epochs) < 5 or len(noncap_epochs) < 5:
                continue  # insufficient data

            p90_cap = np.percentile(cap_epochs, 90)
            p90_noncap = np.percentile(noncap_epochs, 90)

            subject_metrics.append({
                "subject": subject,
                "label": g[LABEL_COL].iloc[0],
                "p90_cap": p90_cap,
                "p90_noncap": p90_noncap,
                "cap_ratio": p90_cap / (p90_noncap + 1e-8)
            })

        subj_df = pd.DataFrame(subject_metrics)
        subj_df.to_csv(
            OUT_DIR / f"cap_tail_uncertainty_{channel}.csv",
            index=False
        )

        # ---------------- STATISTICAL TEST ---------------- #
        control_ratio = subj_df[subj_df["label"] == 0]["cap_ratio"]
        rare_ratio = subj_df[subj_df["label"] != 0]["cap_ratio"]

        if len(rare_ratio) > 0:
            u, p = mannwhitneyu(
                control_ratio,
                rare_ratio,
                alternative="less"
            )

            cap_stats[channel] = {
                "control_cap_ratio_mean": float(control_ratio.mean()),
                "rare_cap_ratio_mean": float(rare_ratio.mean()),
                "p_value": float(p),
                "n_control": int(len(control_ratio)),
                "n_rare": int(len(rare_ratio)),
            }

            print(
                f"  ✓ CAP ratio | Control={control_ratio.mean():.2f}, "
                f"Rare={rare_ratio.mean():.2f}, p={p:.4e}"
            )

    with open(OUT_DIR / "phase6C_cap_stats.json", "w") as f:
        json.dump(cap_stats, f, indent=2)

    print("\n🎉 Phase-6C completed successfully\n")


if __name__ == "__main__":
    main()
