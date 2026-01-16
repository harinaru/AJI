"""
ABLATION — CAP OFF
Removes CAP influence from temporal memory.

Question answered:
❓ Does CAP-gating actually matter?
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from ..config import Config
from ..dataset import TemporalEEGDataset, collate_fn
from ..model import CAPGatedTemporalModel
from torch.utils.data import DataLoader


def main():
    print("\n" + "=" * 80)
    print(" ABLATION: CAP-GATING REMOVED ")
    print("=" * 80)

    config = Config()
    device = torch.device("cuda")

    results = []

    for channel in config.CHANNELS:
        print(f"\n🔬 Channel: {channel}")

        # Load features
        feat_path = config.PHASE5_FEATURE_DIR / f"{channel}_selected_features.csv"
        feature_names = list(np.loadtxt(feat_path, dtype=str, delimiter=",", skiprows=1))

        # Dataset
        csv_path = config.PHASE4_DATA_DIR / f"{channel}.csv"
        dataset = TemporalEEGDataset(csv_path, feature_names)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

        # Model (CAP ignored by forcing constant alpha)
        model = CAPGatedTemporalModel(
            input_dim=config.INPUT_DIM,
            hidden_dim=config.HIDDEN_DIM,
            output_dim=config.OUTPUT_DIM,
            alpha_A_init=0.7,
            alpha_B_init=0.7
        ).to(device)

        model.eval()

        # Load trained weights
        ckpt = torch.load(config.CHECKPOINT_DIR / f"{channel}_best.pt")
        model.load_state_dict(ckpt["model_state"])

        with torch.no_grad():
            for batch in tqdm(loader, desc="Inference"):
                X = batch["features"].to(device)
                cap = torch.zeros_like(batch["cap_flags"]).to(device)  # 🚨 CAP OFF
                mask = batch["mask"].to(device)
                label = batch["labels"].item()

                loss = model.compute_loss(X, cap, mask).item()
                results.append({
                    "channel": channel,
                    "label": label,
                    "recon_error": loss
                })

    df = pd.DataFrame(results)
    out_path = config.OUTPUT_DIR / "logs" / "ablation_cap_off.csv"
    df.to_csv(out_path, index=False)

    print(f"\n📁 CAP-OFF results saved to: {out_path}")
    print("❗ EXPECTED: weaker separation vs CAP-ON")


if __name__ == "__main__":
    main()
