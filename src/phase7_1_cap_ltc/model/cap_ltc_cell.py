# src/phase7_1_cap_ltc/model/cap_ltc_cell.py

"""
CAP-LTC Cell
------------
CAP-modulated continuous-time recurrent dynamics.

Mathematical Properties
-----------------------
✔ CAP-conditioned temporal adaptation
✔ Continuous-time neural dynamics
✔ Stable positive τ via softplus
✔ Numerical stability constraints
✔ Hidden-state stabilization
✔ Reviewer-grade explainability support
✔ τ trajectory extraction
✔ XAI-ready implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Union

from .. import config



class CAPLTCCell(nn.Module):
    """
    CAP-modulated Liquid Time-Constant recurrent cell.

    Temporal dynamics:

        τ(t) = softplus(
            τ_base + τ_cap[s_t]
        )

    Continuous-time update:

        h_t = h_(t-1)
              +
              (Δt / τ_t)
              *
              (
                  -h_(t-1)
                  +
                  φ(Wx + Uh)
              )

    where:
        s_t = CAP state
    """

    # =========================================================================
    # INIT
    # =========================================================================
    def __init__(
        self,
        hidden_size: int,
    ):

        super().__init__()

        self.hidden_size = hidden_size

        # ---------------------------------------------------------------------
        # Learnable base time constant
        # ---------------------------------------------------------------------
        self.tau_base = nn.Parameter(
            torch.tensor(
                config.TAU_BASE_INIT,
                dtype=torch.float32,
            )
        )

        # ---------------------------------------------------------------------
        # CAP-specific τ offsets
        #
        # One parameter per CAP state:
        #   NC, A1, A2, A3, A?
        # ---------------------------------------------------------------------
        self.tau_cap = nn.Parameter(
            torch.tensor(
                config.TAU_CAP_INIT,
                dtype=torch.float32,
            )
        )

        self.n_cap_states = len(
            config.TAU_CAP_INIT
        )

        # ---------------------------------------------------------------------
        # Continuous-time projections
        # ---------------------------------------------------------------------
        self.input_proj = nn.Linear(
            hidden_size,
            hidden_size,
        )

        self.state_proj = nn.Linear(
            hidden_size,
            hidden_size,
        )

        # ---------------------------------------------------------------------
        # τ cache for explainability
        # ---------------------------------------------------------------------
        self.last_tau = None

    # =========================================================================
    # FORWARD
    # =========================================================================
    def forward(
        self,
        x_t: torch.Tensor,        # (B,H)
        h_prev: torch.Tensor,     # (B,H)
        cap_t: torch.Tensor,      # (B,)
        return_tau: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor]
    ]:
        """
        One CAP-LTC recurrent update step.

        Parameters
        ----------
        x_t:
            Current input state

        h_prev:
            Previous hidden state

        cap_t:
            CAP state index

        return_tau:
            If True:
                returns (h_new, tau)

        Returns
        -------
        h_new:
            Updated hidden state

        Optional:
            tau
        """

        # =====================================================================
        # SAFETY: CAP INDEX VALIDATION
        # =====================================================================
        cap_t = (
            cap_t.long()
            .clamp(
                0,
                self.n_cap_states - 1,
            )
        )

        # =====================================================================
        # CAP-CONDITIONED τ
        # =====================================================================
        tau = F.softplus(
            self.tau_base
            +
            self.tau_cap[cap_t]
        ).unsqueeze(-1)

        # =====================================================================
        # NUMERICAL STABILITY
        # =====================================================================
        tau = torch.clamp(
            tau,
            min=config.TAU_MIN,
            max=config.TAU_MAX,
        )

        # =====================================================================
        # SAVE τ FOR XAI
        # =====================================================================
        self.last_tau = tau.detach()

        # =====================================================================
        # CONTINUOUS-TIME DYNAMICS
        # =====================================================================
        dh = torch.tanh(
            self.input_proj(x_t)
            +
            self.state_proj(h_prev)
        )

        # =====================================================================
        # LIQUID TIME-CONSTANT UPDATE
        # =====================================================================
        h_new = h_prev + (
            config.DT / tau
        ) * (
            -h_prev + dh
        )

        # =====================================================================
        # HIDDEN STATE STABILIZATION
        # =====================================================================
        h_new = torch.clamp(
            h_new,
            min=-5.0,
            max=5.0,
        )

        # =====================================================================
        # OPTIONAL τ RETURN
        # =====================================================================
        if return_tau:

            return h_new, tau

        return h_new