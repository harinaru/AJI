import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from captum.attr import IntegratedGradients

from src.phase7_cap_gated_model.dataset import TemporalEEGDataset, collate_fn
from src.phase7_cap_gated_model.model import CapGatedModel
from src.phase7_cap_gated_model.config import Config


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_reconstruction_error(x, recon, mask):
    mse = (x - recon) ** 2
    mse = mse.mean(dim=2)  # mean over features
    mse = mse * mask
    return mse.sum(dim=1) / mask.sum(dim=1)


def compute_cap_ratio(cap_flags, mask):
    cap_flags = cap_flags * mask
    return cap_flags.sum(dim=1) / mask.sum(dim=1)


def run_epoch_metrics():

    print("\n========== SAVING EPOCH METRICS ==========")

    dataset = TemporalEEGDataset(
        csv_path=Config.DATA_PATH,
        feature_names=Config.FEATURE_COLUMNS,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = CapGatedModel(Config)
    ckpt = torch.load(Config.MODEL_PATH, map_location=DEVICE)

    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)

    model.to(DEVICE)
    model.eval()

    ig = IntegratedGradients(model)

    all_rows = []

    with torch.no_grad():

        for batch in tqdm(loader):

            x = batch["features"].to(DEVICE)
            cap_flags = batch["cap_flags"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)

            labels = batch["labels"].cpu().numpy()
            subject_ids = batch["subject_ids"]

            recon, tau, z = model(
                x,
                cap_flags,
                return_tau=True,
                return_latent=True
            )

            error = compute_reconstruction_error(x, recon, mask)
            cap_ratio = compute_cap_ratio(cap_flags, mask)

            error = error.cpu().numpy()
            cap_ratio = cap_ratio.cpu().numpy()
            tau = tau.detach().cpu().numpy()
            z = z.detach().cpu().numpy()

            attr = ig.attribute(
                inputs=x,
                additional_forward_args=(cap_flags,),
                target=None
            )

            ig_sum = attr.sum(dim=(1, 2)).detach().cpu().numpy()

            for i in range(x.size(0)):

                row = {
                    "subject_id": subject_ids[i],
                    "label": int(labels[i]),
                    "reconstruction_error": float(error[i]),
                    "cap_ratio": float(cap_ratio[i]),
                    "tau": float(tau[i]),
                    "ig_sum": float(ig_sum[i]),
                }

                for j in range(z.shape[1]):
                    row[f"z_{j}"] = float(z[i, j])

                all_rows.append(row)

    df = pd.DataFrame(all_rows)

    output_path = Path(Config.RESULTS_DIR) / "epoch_metrics.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("Saved to:", output_path)
    print("Rows:", len(df))


if __name__ == "__main__":
    run_epoch_metrics()