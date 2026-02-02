"""
PHASE 7.5 — INFERENCE & UNCERTAINTY EXTRACTION
CAP-Gated Temporal Normality Model

✔ Loads trained checkpoints (raw state_dict supported)
✔ Runs inference on all subjects
✔ Computes reconstruction-based uncertainty
✔ Saves subject-level CSV
✔ Saves epoch-level arrays for PhD-grade visualization
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config
from .dataset import TemporalEEGDataset, collate_fn
from .model import CAPGatedTemporalModel


# =========================================================
# DEVICE
# =========================================================

def get_device():
    if torch.cuda.is_available():
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        print("⚠️ CUDA not available — using CPU for inference")
        return torch.device("cpu")


# =========================================================
# SUBJECT-LEVEL + EPOCH-LEVEL METRICS
# =========================================================

@torch.no_grad()
def compute_subject_metrics(
    model,
    loader,
    device,
    epoch_out_dir: Path
):
    """
    Computes next-step reconstruction error based uncertainty
    ALSO saves epoch-level error + CAP flags for visualization
    """
    model.eval()
    rows = []

    epoch_out_dir.mkdir(parents=True, exist_ok=True)

    for batch in tqdm(loader, desc="Inference"):
        features = batch["features"].to(device)     # [B, T, F]
        cap = batch["cap_flags"].to(device)         # [B, T]
        mask = batch["mask"].to(device)             # [B, T]

        labels = batch["labels"].cpu().numpy()
        subjects = batch["subject_ids"]
        lengths = batch["lengths"].cpu().numpy()

        # Forward
        preds, _ = model(features, cap, mask)       # [B, T, F]

        # Next-step targets
        targets = torch.zeros_like(features)
        targets[:, :-1] = features[:, 1:]
        targets[:, -1] = features[:, -1]

        # Epoch-wise MSE
        epoch_mse = ((preds - targets) ** 2).mean(dim=2).cpu().numpy()

        for i, sid in enumerate(subjects):
            T = int(lengths[i])
            e = epoch_mse[i, :T]
            c = cap[i, :T].cpu().numpy()

            # --------------------------------------------------
            # SAVE EPOCH-LEVEL SIGNALS (KEY FOR PhD FIGURES)
            # --------------------------------------------------
            np.save(epoch_out_dir / f"{sid}_epoch_errors.npy", e)
            np.save(epoch_out_dir / f"{sid}_cap_flags.npy", c)
            np.save(epoch_out_dir / f"{sid}_length.npy", np.array([T]))

            # --------------------------------------------------
            # SUBJECT-LEVEL SUMMARY
            # --------------------------------------------------
            rows.append({
                "subject_id": sid,
                "label": int(labels[i]),
                "label_name": "Control" if labels[i] == 0 else "Rare",
                "mean_error": float(np.mean(e)),
                "std_error": float(np.std(e)),
                "p90_error": float(np.percentile(e, 90)),
                "p95_error": float(np.percentile(e, 95)),
                "max_error": float(np.max(e)),
                "cap_ratio": float(np.mean(c)),
                "n_epochs": int(T),
            })

    return pd.DataFrame(rows)


# =========================================================
# INFERENCE FOR ONE CHANNEL
# =========================================================

def infer_channel(channel: str, config: Config, device):
    print("\n" + "=" * 80)
    print(f"🔍 INFERENCE — CHANNEL: {channel}")
    print("=" * 80)

    # -----------------------------------------------------
    # Load selected features
    # -----------------------------------------------------
    feat_path = config.PHASE5_FEATURE_DIR / f"{channel}_selected_features.csv"
    feature_names = list(
        np.loadtxt(feat_path, dtype=str, delimiter=",", skiprows=1)
    )

    # -----------------------------------------------------
    # Dataset (SAME CLEAN DATA AS TRAINING)
    # -----------------------------------------------------
    csv_path = config.PHASE7_DATA_DIR / f"{channel}_clean.csv"
    dataset = TemporalEEGDataset(csv_path, feature_names, verbose=False)

    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # -----------------------------------------------------
    # Load model + checkpoint
    # -----------------------------------------------------
    ckpt_path = config.CHECKPOINT_DIR / f"{channel}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = CAPGatedTemporalModel(
        input_dim=config.INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        output_dim=config.OUTPUT_DIM,
        alpha_A_init=config.ALPHA_A_INIT,
        alpha_B_init=config.ALPHA_B_INIT,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)

    # ✔ RAW state_dict SUPPORT (THIS FIXED YOUR ERROR)
    if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state_dict = ckpt
        print("✔ Loaded raw state_dict checkpoint")
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        print("✔ Loaded wrapped model_state_dict checkpoint")
    else:
        raise KeyError(f"Unsupported checkpoint format: {ckpt.keys()}")

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # -----------------------------------------------------
    # Output directories
    # -----------------------------------------------------
    base_out = config.OUTPUT_DIR / "inference"
    epoch_out = base_out / "epoch_level" / channel

    base_out.mkdir(parents=True, exist_ok=True)
    epoch_out.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # Inference
    # -----------------------------------------------------
    df = compute_subject_metrics(
        model=model,
        loader=loader,
        device=device,
        epoch_out_dir=epoch_out
    )

    out_path = base_out / f"{channel}_subject_metrics.csv"
    df.to_csv(out_path, index=False)

    print(f"📁 Saved subject metrics: {out_path}")
    print(f"📁 Saved epoch-level signals: {epoch_out}")

    return df


# =========================================================
# MAIN
# =========================================================

def main():
    print("\n" + "=" * 80)
    print("PHASE 7.5 — INFERENCE & UNCERTAINTY EXTRACTION")
    print("=" * 80)

    config = Config()
    device = get_device()

    all_results = []

    for ch in config.CHANNELS:
        df = infer_channel(ch, config, device)
        df["channel"] = ch
        all_results.append(df)

    summary = pd.concat(all_results, ignore_index=True)

    summary_path = (
        config.OUTPUT_DIR
        / "inference"
        / "summary_all_channels.csv"
    )
    summary.to_csv(summary_path, index=False)

    print("\n📊 SUMMARY SAVED")
    print(summary.groupby(["channel", "label_name"])["p90_error"].mean())

    print("\n🎉 PHASE 7.5 INFERENCE COMPLETED")


if __name__ == "__main__":
    main()
