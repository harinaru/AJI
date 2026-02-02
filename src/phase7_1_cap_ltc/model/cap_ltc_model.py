# src/phase7_1_cap_ltc/model/cap_ltc_model.py

"""
CAP-LTC Model
-------------
Encoder → CAP-LTC → Decoder

Design principles:
• Variable-channel safe
• Per-channel temporal dynamics
• Global CAP modulation
• Mask-aware recurrence
"""

import torch
import torch.nn as nn
from typing import Tuple

from .. import config
from .cap_ltc_cell import CAPLTCCell


class CAP_LTC_Model(nn.Module):
    """
    CAP-LTC autoencoder with per-channel continuous-time dynamics.
    """

    def __init__(self):
        super().__init__()

        F = config.FEATURE_DIM
        H_enc = config.ENCODER_HIDDEN
        H_ltc = config.LTC_HIDDEN

        # -------------------------------------------------
        # Channel-agnostic encoder (shared across channels)
        # -------------------------------------------------
        self.encoder_fc1 = nn.Linear(F, H_enc)
        self.encoder_fc2 = nn.Linear(H_enc, H_ltc)

        # CAP-modulated recurrent cell
        self.ltc_cell = CAPLTCCell(hidden_size=H_ltc)

        # -------------------------------------------------
        # Decoder (shared across channels)
        # -------------------------------------------------
        self.decoder_fc1 = nn.Linear(H_ltc, H_enc)
        self.decoder_fc2 = nn.Linear(H_enc, F)

    def forward(
        self,
        x: torch.Tensor,     # (B, T, C, F)
        cap: torch.Tensor,   # (B, T)
        mask: torch.Tensor,  # (B, T)
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, T, C, F = x.shape

        # -------------------------------------------------
        # Encode per-channel (shared weights)
        # -------------------------------------------------
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, C, T, F)
        x = x.view(B * C, T, F)                 # (B*C, T, F)

        z = torch.relu(self.encoder_fc1(x))
        z = self.encoder_fc2(z)                 # (B*C, T, H)

        # -------------------------------------------------
        # CAP-LTC recurrence (per-channel state)
        # CAP is global → expanded across channels
        # -------------------------------------------------
        h = torch.zeros(
            B * C,
            z.size(-1),
            device=z.device,
            dtype=z.dtype,
        )

        outputs = []

        for t in range(T):
            cap_t = cap[:, t].unsqueeze(1).expand(B, C).reshape(-1)
            mask_t = mask[:, t].unsqueeze(1).expand(B, C).reshape(-1)

            h_new = self.ltc_cell(z[:, t], h, cap_t)

            # Masked update: keep previous state if invalid
            h = torch.where(
                mask_t.unsqueeze(-1),
                h_new,
                h,
            )

            outputs.append(h.unsqueeze(1))

        h_seq = torch.cat(outputs, dim=1)  # (B*C, T, H)

        # -------------------------------------------------
        # Decode
        # -------------------------------------------------
        y = torch.relu(self.decoder_fc1(h_seq))
        y = self.decoder_fc2(y)             # (B*C, T, F)

        recon = y.view(B, C, T, F).permute(0, 2, 1, 3)

        return recon, h_seq
