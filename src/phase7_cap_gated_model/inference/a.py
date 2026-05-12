import sys
from pathlib import Path

# ==========================================================
# Ensure project root is in PYTHONPATH (safe mode)
# ==========================================================
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# ==========================================================
# Imports
# ==========================================================
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from captum.attr import IntegratedGradients

from src.phase7_cap_gated_model.dataset import Phase7Dataset
from src.phase7_cap_gated_model.model import CapGatedModel
from src.phase7_cap_gated_model.config import Config


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================================
# Utility Functions
# ==========================================================

def compute_reconstruction_error(x, recon):
    return torch.mean((x - recon) ** 2, dim=(1, 2))


def compute_cap_ratio(cap_flags):
    return cap_flags.float().mean(dim=1)


def validate_config():
    required_attrs = [
        "DATA_PATH",
        "CHANNEL",
        "BATCH_SIZE",
        "MODEL_PATH",
        "RESULTS_DIR"
    ]

    for attr in required_attrs:
        if not hasattr(Config, attr):
            raise AttributeError(f"Config missing required attribute: {attr}")


# ==========================================================
# Main Function
# ==========================================================

def run_epoch_metrics():

    print("\n========== SAVING EPOCH METRICS ==========")
    print("Device:", DEVICE)

    validate_config()

    # ------------------------------------------------------
    # Dataset
    # ------------------------------------------------------
    print("\nLoading dataset...")
    dataset = Phase7Dataset(
        csv_path=Config.DATA_PATH,
        channel=Config.CHANNEL
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False
    )

    print("Dataset size:", len(dataset))

    # ------------------------------------------------------
    # Model
    # ------------------------------------------------------
    print("\nLoading model...")
    model = CapGatedModel(Config)

    if not Path(Config.MODEL_PATH).exists():
        raise FileNotFoundError(f"Model not found: {Config.MODEL_PATH}")

    ckpt = torch.load(Config.MODEL_PATH, map_location=DEVICE)

    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)

    model.to(DEVICE)
    model.eval()

    print("Model loaded successfully.")

    # ------------------------------------------------------
    # Integrated Gradients
    # ------------------------------------------------------
    ig = IntegratedGradients(model)

    # ------------------------------------------------------
    # Storage
    # ------------------------------------------------------
    all_rows = []

    # ------------------------------------------------------
    # Inference Loop
    # ------------------------------------------------------
    print("\nRunning inference...")

    with torch.no_grad():

        for batch in tqdm(loader):

            required_keys = ["features", "cap_flags", "label", "subject_id", "epoch_index"]
            for key in required_keys:
                if key not in batch:
                    raise KeyError(f"Batch missing required key: {key}")

            x = batch["features"].to(DEVICE)
            cap_flags = batch["cap_flags"].to(DEVICE)

            labels = batch["label"].cpu().numpy()
            subjects = batch["subject_id"]
            epochs = batch["epoch_index"]

            # --------------------------------------------------
            # Forward pass
            # --------------------------------------------------
            try:
                recon, tau, z = model(
                    x,
                    cap_flags,
                    return_tau=True,
                    return_latent=True
                )
            except TypeError:
                raise RuntimeError(
                    "Model forward must support return_tau=True and return_latent=True"
                )

            # --------------------------------------------------
            # Compute metrics
            # --------------------------------------------------
            error = compute_reconstruction_error(x, recon)
            cap_ratio = compute_cap_ratio(cap_flags)

            error = error.cpu().numpy()
            cap_ratio = cap_ratio.cpu().numpy()

            tau = tau.detach().cpu().numpy() if isinstance(tau, torch.Tensor) else tau
            z = z.detach().cpu().numpy() if isinstance(z, torch.Tensor) else z

            # --------------------------------------------------
            # Integrated Gradients
            # --------------------------------------------------
            attr = ig.attribute(
                inputs=x,
                additional_forward_args=(cap_flags,),
                target=None
            )

            ig_sum = attr.sum(dim=(1, 2)).detach().cpu().numpy()

            # --------------------------------------------------
            # Save
            # --------------------------------------------------
            batch_size = x.size(0)

            for i in range(batch_size):

                row = {
                    "subject_id": subjects[i],
                    "epoch_index": int(epochs[i]),
                    "label": int(labels[i]),
                    "reconstruction_error": float(error[i]),
                    "cap_ratio": float(cap_ratio[i]),
                    "tau": float(tau[i]),
                    "ig_sum": float(ig_sum[i]),
                }

                # Add latent features
                if z is not None:
                    for j in range(z.shape[1]):
                        row[f"z_{j}"] = float(z[i, j])

                all_rows.append(row)

    # ------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------
    df = pd.DataFrame(all_rows)

    results_dir = Path(Config.RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / "epoch_metrics.csv"
    df.to_csv(output_path, index=False)

    print("\nSaved epoch metrics to:", output_path)
    print("Total epochs saved:", len(df))
    print("Columns:", list(df.columns))


# ==========================================================
# Entry
# ==========================================================

if __name__ == "__main__":
    run_epoch_metrics()