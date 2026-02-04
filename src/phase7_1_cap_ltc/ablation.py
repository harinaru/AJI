"""
Phase 7.3 — Ablation Studies (FINAL, SAFE)
------------------------------------------
✔ Full CAP-LTC
✔ LTC without CAP modulation
✔ LSTM baseline
✔ Window → scalar → dataset aggregation
✔ Overlap-agnostic
"""

import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config
from .dataset import TemporalEEGDataset, collate_fn
from .model.cap_ltc_model import CAP_LTC_Model
from .metrics import reconstruction_error


# =============================================================================
# BASELINE — SIMPLE LSTM AUTOENCODER
# =============================================================================
class LSTM_AE(nn.Module):
    """
    Baseline LSTM Autoencoder (variable-channel safe)
    """

    def __init__(self):
        super().__init__()
        F = config.FEATURE_DIM
        H = config.LTC_HIDDEN

        self.encoder = nn.LSTM(
            input_size=F,
            hidden_size=H,
            batch_first=True,
        )
        self.decoder = nn.Linear(H, F)

    def forward(self, x: torch.Tensor):
        """
        x: [B, T, C, F]
        """
        B, T, C, F = x.shape
        x = x.view(B * C, T, F)

        z, _ = self.encoder(x)
        recon = self.decoder(z)

        recon = recon.view(B, C, T, F).permute(0, 2, 1, 3)
        return recon


# =============================================================================
# DATA LOADER
# =============================================================================
def get_loader():
    dataset = TemporalEEGDataset(overlap=config.OVERLAP_PERCENT)

    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.NUM_WORKERS,
    )


# =============================================================================
# EVALUATION (SAFE SCALAR AGGREGATION)
# =============================================================================
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    window_errors = []

    for batch in loader:
        x = batch["features"].to(device)
        cap = batch["cap"].to(device)
        mask = batch["mask"].to(device)

        # Forward
        if isinstance(model, CAP_LTC_Model):
            recon, _ = model(x, cap, mask)
        else:
            recon = model(x)

        # [B, T] error
        err_ts = reconstruction_error(recon, x, mask)

        # reduce per window → scalar
        per_window = err_ts.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        window_errors.extend(per_window.cpu().tolist())

    return float(sum(window_errors) / len(window_errors))


# =============================================================================
# MAIN ABLATION
# =============================================================================
def run_ablation() -> None:
    logging.basicConfig(level=logging.INFO)
    device = torch.device(config.DEVICE)

    loader = get_loader()

    # --------------------------------------------------
    # 1. FULL CAP-LTC
    # --------------------------------------------------
    cap_ltc = CAP_LTC_Model().to(device)
    ckpt_name = f"cap_ltc_best_overlap_{int(config.OVERLAP_PERCENT * 100)}.pt"
    ckpt = torch.load(
        config.MODEL_DIR / ckpt_name,
        map_location=device,
    )
    cap_ltc.load_state_dict(ckpt["model_state"])

    err_cap_ltc = evaluate(cap_ltc, loader, device)
    logging.info(f"[Ablation] CAP-LTC error = {err_cap_ltc:.6f}")

    # --------------------------------------------------
    # 2. LTC WITHOUT CAP MODULATION
    # --------------------------------------------------
    ltc_no_cap = CAP_LTC_Model().to(device)
    ltc_no_cap.load_state_dict(cap_ltc.state_dict(), strict=False)

    with torch.no_grad():
        ltc_no_cap.ltc_cell.tau_cap.zero_()

    err_ltc = evaluate(ltc_no_cap, loader, device)
    logging.info(f"[Ablation] LTC (no CAP) error = {err_ltc:.6f}")

    # --------------------------------------------------
    # 3. LSTM BASELINE
    # --------------------------------------------------
    lstm = LSTM_AE().to(device)
    err_lstm = evaluate(lstm, loader, device)
    logging.info(f"[Ablation] LSTM error = {err_lstm:.6f}")

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------
    print("\n" + "=" * 80)
    print("ABLATION SUMMARY")
    print("=" * 80)
    print(f"CAP-LTC        : {err_cap_ltc:.6f}")
    print(f"LTC (no CAP)   : {err_ltc:.6f}")
    print(f"LSTM baseline  : {err_lstm:.6f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_ablation()
