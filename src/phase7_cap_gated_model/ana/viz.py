import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc

# ==========================================================
# CONFIG
# ==========================================================

BASE_DIR = Path("results/phase7_cap_gated_model")
DATA_DIR = BASE_DIR / "data"
FIG_DIR = BASE_DIR / "figures"

FIG_DIR.mkdir(parents=True, exist_ok=True)

channels = ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"]


# ==========================================================
# LOAD LABELS FROM ORIGINAL CLEAN FILE
# ==========================================================

def load_labels(channel):
    csv_path = DATA_DIR / f"{channel}_clean.csv"
    df = pd.read_csv(csv_path)

    # Ensure consistent types
    df["subject"] = df["subject"].astype(str)

    label_map = (
        df.groupby("subject")["label"]
        .first()
        .to_dict()
    )

    return label_map


# ==========================================================
# LOAD ALL CHANNEL SUMMARIES
# ==========================================================

dfs = {}

for ch in channels:
    summary_path = BASE_DIR / f"{ch}_epoch_reconstruction_summary.csv"

    summary_df = pd.read_csv(summary_path)

    # Normalize subject ID type
    summary_df["subject_id"] = summary_df["subject_id"].astype(str)

    label_map = load_labels(ch)

    label_df = pd.DataFrame(
        list(label_map.items()),
        columns=["subject_id", "label"]
    )

    summary_df = summary_df.merge(label_df, on="subject_id", how="left")

    dfs[ch] = summary_df


# ==========================================================
# FIGURE 1 — BOXPLOTS PER CHANNEL
# ==========================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, ch in enumerate(channels):
    df = dfs[ch]

    df = df.dropna(subset=["label"])

    controls = df[df["label"] == 0]["mean_error"]
    rare = df[df["label"] == 1]["mean_error"]

    axes[i].boxplot([controls, rare], labels=["Control", "Rare"])
    axes[i].set_title(f"{ch}")
    axes[i].set_ylabel("Reconstruction Error")

plt.tight_layout()
plt.savefig(FIG_DIR / "Figure1_Boxplots.png", dpi=300)
plt.close()


# ==========================================================
# FUSION SCORE (Mean Across Channels)
# ==========================================================

fusion_df = None

for ch in channels:
    df = dfs[ch][["subject_id", "mean_error", "label"]].copy()
    df = df.rename(columns={"mean_error": f"{ch}_error"})

    if fusion_df is None:
        fusion_df = df
    else:
        fusion_df = fusion_df.merge(
            df[["subject_id", f"{ch}_error"]],
            on="subject_id",
            how="inner"
        )

# Compute fusion score
error_cols = [f"{ch}_error" for ch in channels]
fusion_df["fusion_score"] = fusion_df[error_cols].mean(axis=1)

# Remove NaNs safely
fusion_df = fusion_df.dropna(subset=["label", "fusion_score"])

fusion_df["label"] = fusion_df["label"].astype(int)


# ==========================================================
# FIGURE 2 — ROC CURVE (Fusion)
# ==========================================================

fpr, tpr, _ = roc_curve(
    fusion_df["label"],
    fusion_df["fusion_score"]
)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Fusion ROC Curve")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(FIG_DIR / "Figure2_Fusion_ROC.png", dpi=300)
plt.close()


# ==========================================================
# FIGURE 3 — FUSION DISTRIBUTION
# ==========================================================

controls = fusion_df[fusion_df["label"] == 0]["fusion_score"]
rare = fusion_df[fusion_df["label"] == 1]["fusion_score"]

plt.figure(figsize=(8, 5))
plt.hist(controls, bins=30, alpha=0.6, label="Control")
plt.hist(rare, bins=30, alpha=0.6, label="Rare")
plt.xlabel("Fusion Score")
plt.ylabel("Frequency")
plt.title("Fusion Score Distribution")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "Figure3_Fusion_Distribution.png", dpi=300)
plt.close()


# ==========================================================
# DONE
# ==========================================================

print("\n=====================================")
print(" Journal-Grade Visualizations Created ")
print("=====================================")
print("Saved to:", FIG_DIR)
print(f"Fusion AUC: {roc_auc:.4f}")