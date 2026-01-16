"""
Phase 7 — CAP-Gated Temporal Model (FINAL, STABLE)

Core novelty:
- CAP-controlled temporal memory
- Delta (Δx) prediction instead of raw reconstruction
- Bounded outputs + proper normalization
"""

import torch
import torch.nn as nn


# =========================================================
# CAP-GATED TEMPORAL CELL
# =========================================================

class CAPGatedCell(nn.Module):
    """
    CAP-Gated Temporal Memory Cell
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        alpha_A_init: float = 0.9,
        alpha_B_init: float = 0.5,
    ):
        super().__init__()

        self.alpha_A = nn.Parameter(torch.tensor(alpha_A_init))
        self.alpha_B = nn.Parameter(torch.tensor(alpha_B_init))

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.Tanh()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x_t: torch.Tensor,      # [B, input_dim]
        h_prev: torch.Tensor,  # [B, hidden_dim]
        cap_t: torch.Tensor,   # [B]
    ) -> torch.Tensor:

        if cap_t.dim() == 1:
            cap_t = cap_t.unsqueeze(1)

        alpha = cap_t * self.alpha_A + (1.0 - cap_t) * self.alpha_B
        alpha = torch.clamp(alpha, 0.0, 1.0)

        x_proj = self.activation(self.input_proj(x_t))
        h_t = alpha * h_prev + (1.0 - alpha) * x_proj

        return self.norm(h_t)


# =========================================================
# COMPLETE TEMPORAL MODEL (Δx PREDICTION)
# =========================================================

class CAPGatedTemporalModel(nn.Module):
    """
    CAP-Gated Temporal Normality Model
    Predicts Δx_t = x_{t+1} − x_t
    """

    def __init__(
        self,
        input_dim: int = 45,
        hidden_dim: int = 64,
        output_dim: int = 45,
        alpha_A_init: float = 0.9,
        alpha_B_init: float = 0.5,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # CAP-gated memory
        self.cell = CAPGatedCell(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            alpha_A_init=alpha_A_init,
            alpha_B_init=alpha_B_init,
        )

        # 🔑 Delta predictor (bounded)
        self.delta_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )

        # Learnable scale (kept small)
        self.delta_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        features: torch.Tensor,   # [B, T, F]
        cap_flags: torch.Tensor,  # [B, T]
        mask: torch.Tensor,       # [B, T]
    ):
        B, T, _ = features.shape
        device = features.device

        h = torch.zeros(B, self.hidden_dim, device=device)

        delta_preds = []
        hidden_states = []

        for t in range(T):
            x_t = features[:, t]
            cap_t = cap_flags[:, t]
            valid_t = mask[:, t].unsqueeze(1)

            x_enc = self.encoder(x_t)
            h = self.cell(x_enc, h, cap_t)

            delta = self.delta_predictor(h) * self.delta_scale

            h = h * valid_t
            delta = delta * valid_t

            delta_preds.append(delta)
            hidden_states.append(h)

        return (
            torch.stack(delta_preds, dim=1),   # [B, T, F]
            torch.stack(hidden_states, dim=1), # [B, T, H]
        )

    def compute_loss(
        self,
        features: torch.Tensor,
        cap_flags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Stable delta-prediction loss
        """

        delta_preds, _ = self.forward(features, cap_flags, mask)

        # True deltas
        delta_targets = torch.zeros_like(features)
        delta_targets[:, :-1] = features[:, 1:] - features[:, :-1]

        valid = mask.clone().float()
        valid[:, -1] = 0
        valid = valid.unsqueeze(-1)

        loss = ((delta_preds - delta_targets) ** 2) * valid

        denom = valid.sum().clamp(min=1.0)
        return loss.sum() / denom
