from pathlib import Path
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns

from .normality_model import GaussianNormalityModel

# ---------------- CONFIG ---------------- #

CHANNELS = ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"]

PHASE4_DIR = Path("results/phase4_channel_datasets")
PHASE5_DIR = Path("results/phase5_feature_selection")
OUT_DIR = Path("results/phase6_normality")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL = "label"
SUBJECT_COL = "subject"

sns.set(style="whitegrid", font_scale=1.1)

# --------------------------------------- #


def clean_features(df, feature_cols):
    """
    Control-safe feature cleaning
    """
    X = df[feature_cols].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X = X.fillna(X.median())

    valid_mask = ~X.isna().any(axis=1)
    X = X.loc[valid_mask]

    return X.to_numpy(), valid_mask


def plot_uncertainty_box(subj_df, channel):
    """
    Control vs Rare uncertainty distribution (publication grade)
    """
    plot_df = subj_df.copy()
    plot_df["group"] = plot_df["label"].map(
        lambda x: "Screenable" if x == 0 else "Rare"
    )

    plt.figure(figsize=(6.5, 6))
    sns.boxplot(
        x="group",
        y="uncertainty_mean",
        data=plot_df,
        showfliers=False
    )
    sns.stripplot(
        x="group",
        y="uncertainty_mean",
        data=plot_df,
        color="black",
        alpha=0.5,
        size=4
    )

    n_ctrl = (plot_df["group"] == "Screenable").sum()
    n_rare = (plot_df["group"] == "Rare").sum()

    plt.title(
        f"{channel} — Subject-level Uncertainty\n"
        f"(Control n={n_ctrl}, Rare n={n_rare})",
        fontsize=13
    )
    plt.ylabel("Mean Mahalanobis Uncertainty")
    plt.xlabel("")

    plt.tight_layout()
    plt.savefig(
        OUT_DIR / f"uncertainty_box_{channel}.png",
        dpi=300
    )
    plt.close()


def plot_risk_coverage(ctrl_unc, channel, tau):
    """
    Risk–coverage curve (controls only)
    """
    taus = np.linspace(
        np.percentile(ctrl_unc, 60),
        np.percentile(ctrl_unc, 99.5),
        100
    )

    coverage = []
    risk = []

    for t in taus:
        committed = ctrl_unc < t
        cov = committed.mean()
        coverage.append(cov * 100)

        if committed.sum() > 1:
            risk.append(ctrl_unc[committed].std())
        else:
            risk.append(0)

    plt.figure(figsize=(7, 6))
    plt.plot(coverage, risk, linewidth=2)
    plt.scatter(
        (ctrl_unc < tau).mean() * 100,
        ctrl_unc[ctrl_unc < tau].std(),
        color="red",
        zorder=5,
        label=f"τ = {tau:.2f}"
    )

    plt.xlabel("Coverage (% controls committed)")
    plt.ylabel("Risk (uncertainty std)")
    plt.title(f"{channel} — Risk–Coverage Trade-off")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUT_DIR / f"risk_coverage_{channel}.png",
        dpi=300
    )
    plt.close()


def main():
    print("\n==============================")
    print(" PHASE 6A — BASELINE NORMALITY ")
    print("==============================\n")

    thresholds = {}
    stats_summary = {}

    for channel in tqdm(CHANNELS, desc="Channels"):
        print(f"\n▶ Phase-6 | {channel}")

        # Load data
        df = pd.read_csv(PHASE4_DIR / f"{channel}.csv")
        feats = pd.read_csv(
            PHASE5_DIR / f"{channel}_selected_features.csv"
        )["selected_feature"].tolist()

        # ---------------- CONTROL ONLY ---------------- #
        control_df = df[df[LABEL_COL] == 0].reset_index(drop=True)
        X_control, mask = clean_features(control_df, feats)
        control_df = control_df.loc[mask]

        # Train normality model
        model = GaussianNormalityModel()
        model.fit(X_control)

        # ---------------- SCORE ALL ---------------- #
        X_all, mask_all = clean_features(df, feats)
        df = df.loc[mask_all].reset_index(drop=True)

        df["epoch_uncertainty"] = model.score(X_all)

        # ---------------- SUBJECT LEVEL ---------------- #
        subj_df = (
            df.groupby(SUBJECT_COL)
            .agg(
                uncertainty_mean=("epoch_uncertainty", "mean"),
                uncertainty_std=("epoch_uncertainty", "std"),
                uncertainty_max=("epoch_uncertainty", "max"),
                label=(LABEL_COL, "first"),
            )
            .reset_index()
        )

        subj_df.to_csv(
            OUT_DIR / f"uncertainty_{channel}.csv",
            index=False
        )

        # ---------------- THRESHOLD ---------------- #
        ctrl_unc = subj_df[subj_df["label"] == 0]["uncertainty_mean"].values
        tau = np.percentile(ctrl_unc, 95)
        thresholds[channel] = float(tau)

        # ---------------- STATS ---------------- #
        rare_unc = subj_df[subj_df["label"] != 0]["uncertainty_mean"].values
        if len(rare_unc) > 0:
            u, p = mannwhitneyu(ctrl_unc, rare_unc, alternative="less")
            stats_summary[channel] = {
                "control_mean": float(ctrl_unc.mean()),
                "rare_mean": float(rare_unc.mean()),
                "p_value": float(p),
            }

        # ---------------- VISUALS ---------------- #
        plot_uncertainty_box(subj_df, channel)
        plot_risk_coverage(ctrl_unc, channel, tau)

        print(f"  ✓ τ = {tau:.3f}")

    # Save metadata
    with open(OUT_DIR / "thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)

    with open(OUT_DIR / "stats.json", "w") as f:
        json.dump(stats_summary, f, indent=2)

    print("\n🎉 Phase-6A completed (baseline diagnostics)\n")


if __name__ == "__main__":
    main()
