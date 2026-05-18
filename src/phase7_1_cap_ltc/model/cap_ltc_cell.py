# src/phase7_1_cap_ltc/model/cap_ltc_cell.py

"""
CAP-LTC Cell
------------
CAP-modulated continuous-time recurrent dynamics.

FIXED VERSION — AUROC IMPROVEMENT
==================================

CHANGES FROM PREVIOUS VERSION
------------------------------
1. Removed double tanh saturation.
   The old code applied tanh to dh AND then tanh(h_new)*5 immediately after.
   This killed gradients and made all hidden states near-constant regardless
   of CAP state. Now only ONE activation is applied to dh. h_new is stabilized
   with a simple clamp, preserving gradient flow.

2. Removed broken h_new = tanh(h_new)*5.0 block.
   That line destroyed the continuous-time ODE semantics entirely. The leaky
   integration formula (1-γ)*h_prev + γ*dh now works as intended.

3. τ initialization moved to config.py (TAU_BASE_INIT=3.0).
   With the old init (0.5), softplus(0.5)≈0.97, DT/τ≈1.0 — meaning γ≈1,
   h_prev was forgotten every step. Now τ≈3.1, γ≈0.32, the cell integrates
   ~3 timesteps of context per step.

Mathematical Formulation
------------------------

CAP-conditioned time constants:

    τ(t) = τ_min + softplus(τ_base + τ_cap[s_t])

Stable continuous-time integration:

    γ(t) = clamp(Δt / τ(t), 0, 1)

Hidden-state update:

    h_t = (1 - γ_t) * h_(t-1)  +  γ_t * tanh(α(Wx + Uh))

Stabilization (no saturation):

    h_t = clamp(h_t, -5, 5)     ← simple clamp, no tanh, no gradient kill

✔ Stable bounded dynamics
✔ Continuous-time temporal adaptation
✔ Physiologically interpretable τ modulation
✔ Gradient-friendly — no double tanh
✔ NaN-safe dynamics
✔ XAI-ready τ extraction
"""

# =============================================================================
# IMPORTS
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import (
    Tuple,
    Union,
)

from .. import config


# =============================================================================
# CAP-LTC CELL
# =============================================================================

class CAPLTCCell(nn.Module):

    """
    CAP-modulated Liquid Time-Constant recurrent cell.
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
        # Learnable base τ
        # NOTE: initialized from config.TAU_BASE_INIT
        # Old value 0.5 gave γ≈1 (no memory). New value 3.0 gives γ≈0.32
        # (integrates ~3 timesteps). This is the key fix for temporal learning.
        # ---------------------------------------------------------------------

        self.tau_base = nn.Parameter(

            torch.tensor(

                config.TAU_BASE_INIT,

                dtype=torch.float32,
            )
        )

        # ---------------------------------------------------------------------
        # CAP-specific τ modulation
        # Larger offsets let A1/A2/A3 produce genuinely different τ values,
        # giving the cell distinct temporal dynamics per CAP state.
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
        # Input/state projections
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
        # Learnable dynamical gain
        # ---------------------------------------------------------------------

        self.alpha = nn.Parameter(
            torch.tensor(
                1.0,
                dtype=torch.float32,
            )
        )

        # ---------------------------------------------------------------------
        # τ cache for XAI
        # ---------------------------------------------------------------------

        self.last_tau = None

        self.last_gamma = None

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
            If True: returns (h_new, tau)

        Returns
        -------
        h_new:
            Updated hidden state (clamped, no double-tanh)

        Optional:
            tau
        """

        # =====================================================================
        # CAP SAFETY
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

        tau_raw = (

            self.tau_base

            +

            self.tau_cap[cap_t]
        )

        # ---------------------------------------------------------------------
        # Stable positive τ via softplus
        # softplus(3.0) ≈ 3.05, so τ ≈ TAU_MIN + 3.05 ≈ 3.1
        # γ = DT/τ ≈ 1/3.1 ≈ 0.32  →  cell integrates ~3 timesteps
        # ---------------------------------------------------------------------

        tau = (

            config.TAU_MIN

            +

            F.softplus(
                tau_raw
            )
        ).unsqueeze(-1)

        # ---------------------------------------------------------------------
        # Numerical stability bounds
        # ---------------------------------------------------------------------

        tau = torch.clamp(

            tau,

            min=config.TAU_MIN,

            max=config.TAU_MAX,
        )

        # =====================================================================
        # LEAK COEFFICIENT  γ = Δt / τ
        # Clamped to [0,1] — valid continuous-time integration coefficient
        # =====================================================================

        gamma = torch.clamp(

            config.DT / tau,

            min=0.0,

            max=1.0,
        )

        # =====================================================================
        # SAVE FOR XAI
        # =====================================================================

        self.last_tau = (
            tau.detach().cpu()
        )

        self.last_gamma = (
            gamma.detach().cpu()
        )

        # =====================================================================
        # CONTINUOUS-TIME DYNAMICS
        # dh = tanh(α*(W*x + U*h))
        # SINGLE activation — NOT applied again after integration
        # =====================================================================

        projected = (

            self.input_proj(x_t)

            +

            self.state_proj(h_prev)
        )

        projected = (
            self.alpha
            *
            projected
        )

        dh = torch.tanh(
            projected
        )

        # =====================================================================
        # LEAKY INTEGRATION  (the true LTC update)
        # h_new = (1-γ)*h_prev + γ*dh
        # =====================================================================

        h_new = (

            (1.0 - gamma)
            *
            h_prev

            +

            gamma
            *
            dh
        )

        # =====================================================================
        # STABILIZATION — simple clamp only, NO second tanh
        #
        # OLD (broken):
        #     h_new = torch.tanh(h_new) * 5.0
        #
        # WHY IT WAS BROKEN:
        #   tanh saturates at ±1 → gradient near-zero for most values
        #   multiplying by 5.0 only scales the output, does NOT fix gradients
        #   result: all hidden states converge regardless of input
        #
        # FIX:
        #     h_new = h_new.clamp(-5.0, 5.0)
        #
        # Clamp preserves gradient (gradient=1 inside range, 0 only at walls)
        # Bounds hidden state to prevent blow-up without killing learning
        # =====================================================================

        h_new = h_new.clamp(
            min=-5.0,
            max=5.0,
        )

        # =====================================================================
        # NaN / INF SAFETY
        # =====================================================================

        h_new = torch.nan_to_num(

            h_new,

            nan=0.0,

            posinf=5.0,

            neginf=-5.0,
        )

        # =====================================================================
        # OPTIONAL τ RETURN
        # =====================================================================

        if return_tau:

            return (
                h_new,
                tau,
            )

        return h_new