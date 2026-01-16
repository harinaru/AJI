"""
PHASE 8 — RESEARCH-GRADE VISUALIZATION
CAP-Gated Temporal Normality Model

Uses inference outputs only.
NO training. NO GPU required.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .config import Config


# ---------------------------------------------------------
# GLOBAL STYLE (JOURNAL SAFE)
# ---------------------------------------------------------

sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "figure.dpi": 300
})


# ---------------------------------------------------------
# FIGURE 1 — CONTROL vs RARE (p90 ERROR)
# ---------------------------------------------------------

def plot_control_vs_rare(summary_df, out_dir):
    plt.figure(figsize=(7, 5))

    sns.boxplot(
        data=summary_df,
        x="label_name",
        y="p90_error",
        palette=["#4CAF50", "#E53935"],
        showfliers=False
    )
    sns.stripplot(
        data=summary_df,
        x="label_name",
        y="p90_error",
        color="black",
        alpha=0.6,
        size=4
    )

    plt.title("Subject-Level Burst Uncertainty (p90 Error)")
    plt.xlabel("")
    plt.ylabel("Reconstruction Error (p90)")

    plt.tight_layout()
    plt.savefig(out_dir / "fig1_p90_control_vs_rare.png")
    plt.close()


# ---------------------------------------------------------
# FIGURE 2 — EXAMPLE ERROR TRAJECTORY WITH CAP
# ---------------------------------------------------------

def plot_example_trajectory(channel, inference_dir, out_dir):
    df = pd.read_csv(inference_dir / f"{channel}_subject_metrics.csv")

    # Pick one control + one rare
    control = df[df["label"] == 0].iloc[0]
    rare = df[df["label"] != 0].iloc[0]

    for subj, tag in [(control, "Control"), (rare, "Rare")]:
        errors = np.load(
            inference_dir / f"{subj.subject_id}_epoch_errors.npy"
        )
        cap = np.load(
            inference_dir / f"{subj.subject_id}_cap_flags.npy"
        )

        t = np.arange(len(errors))

        plt.figure(figsize=(10, 4))
        plt.plot(t, errors, label="Reconstruction Error", linewidth=1.5)

        plt.scatter(
            t[cap == 1],
            errors[cap == 1],
            color="red",
            s=12,
            alpha=0.6,
            label="CAP Epochs"
        )

        plt.title(f"{channel} — {tag} Subject Error Trajectory")
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        plt.legend()
        plt.tight_layout()

        plt.savefig(out_dir / f"fig2_{channel}_{tag}_trajectory.png")
        plt.close()


# ---------------------------------------------------------
# FIGURE 3 — CHANNEL-WISE COMPARISON
# ---------------------------------------------------------

def plot_channel_comparison(summary_df, out_dir):
    plt.figure(figsize=(9, 5))

    sns.barplot(
        data=summary_df,
        x="channel",
        y="p90_error",
        hue="label_name",
        ci="sd",
        palette=["#4CAF50", "#E53935"]
    )

    plt.title("Channel-wise Burst Uncertainty (p90)")
    plt.ylabel("Mean p90 Error")
    plt.xlabel("EEG Channel")

    plt.tight_layout()
    plt.savefig(out_dir / "fig3_channel_comparison.png")
    plt.close()


# ---------------------------------------------------------
# FIGURE 4 — CAP vs NON-CAP ERROR (KEY NOVELTY)
# ---------------------------------------------------------

def plot_cap_vs_noncap(summary_df, out_dir):
    plt.figure(figsize=(7, 5))

    sns.scatterplot(
        data=summary_df,
        x="cap_ratio",
        y="p90_error",
        hue="label_name",
        palette=["#4CAF50", "#E53935"]
    )

    plt.title("CAP Density vs Burst Instability")
    plt.xlabel("CAP Ratio")
    plt.ylabel("p90 Reconstruction Error")

    plt.tight_layout()
    plt.savefig(out_dir / "fig4_cap_vs_noncap_error.png")
    plt.close()


# ---------------------------------------------------------
# MAIN CONTROLLER
# ---------------------------------------------------------

def main():
    print("\n" + "=" * 80)
    print("PHASE 8 — VISUALIZATION")
    print("=" * 80)

    config = Config()

    inference_dir = config.OUTPUT_DIR / "inference"
    fig_dir = config.FIGURE_DIR
    fig_dir.mkdir(exist_ok=True)

    summary_path = inference_dir / "summary_all_channels.csv"
    summary_df = pd.read_csv(summary_path)

    print("📊 Generating figures...")

    plot_control_vs_rare(summary_df, fig_dir)
    plot_channel_comparison(summary_df, fig_dir)
    plot_cap_vs_noncap(summary_df, fig_dir)

    print("✅ Figures saved to:", fig_dir)
    print("🎉 PHASE 8 COMPLETED")


if __name__ == "__main__":
    main()
