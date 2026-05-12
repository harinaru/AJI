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
• XAI-ready
• Latent-manifold ready
• Reviewer-grade τ exposure
"""

import torch
import torch.nn as nn

from typing import Tuple, Dict, Optional

from .. import config
from .cap_ltc_cell import CAPLTCCell


class CAP_LTC_Model(nn.Module):
    """
    CAP-LTC autoencoder with
    CAP-conditioned continuous-time dynamics.
    """

    # =========================================================================
    # INIT
    # =========================================================================
    def __init__(self):

        super().__init__()

        F = config.FEATURE_DIM
        H_enc = config.ENCODER_HIDDEN
        H_ltc = config.LTC_HIDDEN

        # ---------------------------------------------------------------------
        # Encoder
        # ---------------------------------------------------------------------
        self.encoder_fc1 = nn.Linear(
            F,
            H_enc,
        )

        self.encoder_fc2 = nn.Linear(
            H_enc,
            H_ltc,
        )

        # ---------------------------------------------------------------------
        # CAP-LTC CELL
        # ---------------------------------------------------------------------
        self.ltc_cell = CAPLTCCell(
            hidden_size=H_ltc
        )

        # ---------------------------------------------------------------------
        # Decoder
        # ---------------------------------------------------------------------
        self.decoder_fc1 = nn.Linear(
            H_ltc,
            H_enc,
        )

        self.decoder_fc2 = nn.Linear(
            H_enc,
            F,
        )

    # =========================================================================
    # τ PARAMETERS
    # =========================================================================
    def get_tau_parameters(
        self
    ) -> Dict[str, torch.Tensor]:
        """
        Returns positive τ parameters.

        Returns
        -------
        {
            tau_base:
                scalar

            tau_cap:
                [N_CAP]

            tau_total:
                [N_CAP]
        }
        """

        tau_base = torch.nn.functional.softplus(
            self.ltc_cell.tau_base
        )

        tau_cap = torch.nn.functional.softplus(
            self.ltc_cell.tau_cap
        )

        return {

            "tau_base":
                tau_base,

            "tau_cap":
                tau_cap,

            "tau_total":
                tau_base + tau_cap,
        }

    # =========================================================================
    # FORWARD
    # =========================================================================
    def forward(
        self,
        x: torch.Tensor,      # (B,T,C,F)
        cap: torch.Tensor,    # (B,T)
        mask: torch.Tensor,   # (B,T)
        return_latents: bool = False,
        return_tau: bool = False,
    ) -> Tuple[
        torch.Tensor,
        Optional[Dict[str, torch.Tensor]]
    ]:
        """
        Forward pass.

        Parameters
        ----------
        return_latents:
            Returns:
                encoder_latent
                hidden_states
                subject_embedding

        return_tau:
            Returns τ values

        Returns
        -------
        recon:
            (B,T,C,F)

        extras:
            Optional dictionary
        """

        B, T, C, F = x.shape

        # =====================================================================
        # ENCODER
        # =====================================================================
        # Convert:
        #   (B,T,C,F)
        # →
        #   (B,C,T,F)
        # →
        #   (B*C,T,F)
        # =====================================================================
        x_perm = (
            x.permute(0, 2, 1, 3)
            .contiguous()
        )

        x_flat = x_perm.view(
            B * C,
            T,
            F,
        )

        # ---------------------------------------------------------------------
        # Shared encoder
        # ---------------------------------------------------------------------
        z1 = torch.relu(
            self.encoder_fc1(x_flat)
        )

        z = self.encoder_fc2(z1)

        # Shape:
        #   (B*C,T,H)
        # ---------------------------------------------------------------------

        # =====================================================================
        # CAP-LTC RECURRENCE
        # =====================================================================
        h = torch.zeros(
            B * C,
            z.size(-1),
            device=z.device,
            dtype=z.dtype,
        )

        hidden_seq = []

        for t in range(T):

            # -------------------------------------------------------------
            # CAP expansion across channels
            # -------------------------------------------------------------
            cap_t = (
                cap[:, t]
                .unsqueeze(1)
                .expand(B, C)
                .reshape(-1)
            )

            # -------------------------------------------------------------
            # Temporal mask
            # -------------------------------------------------------------
            mask_t = (
                mask[:, t]
                .unsqueeze(1)
                .expand(B, C)
                .reshape(-1)
            )

            # -------------------------------------------------------------
            # LTC dynamics
            # -------------------------------------------------------------
            h_new = self.ltc_cell(
                z[:, t],
                h,
                cap_t,
            )

            # -------------------------------------------------------------
            # Mask-aware update
            # -------------------------------------------------------------
            h = torch.where(
                mask_t.unsqueeze(-1),
                h_new,
                h,
            )

            hidden_seq.append(
                h.unsqueeze(1)
            )

        # ---------------------------------------------------------------------
        # Temporal hidden states
        # Shape:
        #   (B*C,T,H)
        # ---------------------------------------------------------------------
        hidden_seq = torch.cat(
            hidden_seq,
            dim=1,
        )

        # =====================================================================
        # DECODER
        # =====================================================================
        y = torch.relu(
            self.decoder_fc1(hidden_seq)
        )

        y = self.decoder_fc2(y)

        # ---------------------------------------------------------------------
        # Reconstruct:
        #   (B*C,T,F)
        # →
        #   (B,C,T,F)
        # →
        #   (B,T,C,F)
        # ---------------------------------------------------------------------
        recon = (
            y.view(B, C, T, F)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        # =====================================================================
        # OPTIONAL OUTPUTS
        # =====================================================================
        extras = {}

        # ---------------------------------------------------------------------
        # LATENT REPRESENTATIONS
        # ---------------------------------------------------------------------
        if return_latents:

            # -------------------------------------------------------------
            # Encoder latent
            # Shape:
            #   (B*C,T,H)
            # -------------------------------------------------------------
            extras["encoder_latent"] = z

            # -------------------------------------------------------------
            # Hidden temporal states
            # Shape:
            #   (B*C,T,H)
            # -------------------------------------------------------------
            extras["hidden_states"] = hidden_seq

            # -------------------------------------------------------------
            # Subject-level embedding
            #
            # Shape:
            #   (B,H)
            #
            # Mean pooling:
            #   channels + time
            # -------------------------------------------------------------
            subject_embedding = (
                hidden_seq
                .view(B, C, T, -1)
                .mean(dim=1)
                .mean(dim=1)
            )

            extras["subject_embedding"] = (
                subject_embedding
            )

        # ---------------------------------------------------------------------
        # τ PARAMETERS
        # ---------------------------------------------------------------------
        if return_tau:

            extras["tau"] = (
                self.get_tau_parameters()
            )

        # =====================================================================
        # RETURN
        # =====================================================================
        if len(extras) > 0:

            return recon, extras

        return recon, None