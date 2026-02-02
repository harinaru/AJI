# src/phase7_1_cap_ltc/model/cap_ltc_cell.py

"""
CAP-LTC Cell
------------
CAP-modulated continuous-time recurrent dynamics
(uncertainty-aware, 5-state safe)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import config


class CAPLTCCell(nn.Module):
    """
    Single CAP-modulated Liquid Time-Constant (LTC) cell.

    τ(t) = softplus(τ_base + τ_cap[state_t])
    """

    def __init__(self, hidden_size: int):
        super().__init__()

        self.hidden_size = hidden_size

        # -------------------------------------------------
        # Learnable time constants
        # -------------------------------------------------
        self.tau_base = nn.Parameter(
            torch.tensor(
                config.TAU_BASE_INIT,
                dtype=torch.float32,
            )
        )

        # One τ offset per CAP state (NC, A1, A2, A3, A?)
        self.tau_cap = nn.Parameter(
            torch.tensor(
                config.TAU_CAP_INIT,
                dtype=torch.float32,
            )
        )  # shape: [N_CAP_STATES]

        self.n_cap_states = len(config.TAU_CAP_INIT)

        # -------------------------------------------------
        # State update projections
        # -------------------------------------------------
        self.input_proj = nn.Linear(hidden_size, hidden_size)
        self.state_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        x_t: torch.Tensor,     # (B, H)
        h_prev: torch.Tensor, # (B, H)
        cap_t: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        """
        One CAP-LTC time-step update.
        """

        # -------------------------------------------------
        # SAFETY: clamp CAP indices
        # -------------------------------------------------
        cap_t = cap_t.long().clamp(0, self.n_cap_states - 1)

        # -------------------------------------------------
        # Time constant τ(t)
        # -------------------------------------------------
        tau = F.softplus(
            self.tau_base + self.tau_cap[cap_t]
        ).unsqueeze(-1)  # (B, 1)

        # -------------------------------------------------
        # Continuous-time inspired update
        # -------------------------------------------------
        dh = torch.tanh(
            self.input_proj(x_t) +
            self.state_proj(h_prev)
        )

        h_new = h_prev + (config.DT / tau) * (-h_prev + dh)

        return h_new
