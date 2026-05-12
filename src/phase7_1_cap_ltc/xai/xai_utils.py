"""
XAI Utilities for CAP-LTC (TIME-SERIES SAFE, FINAL)
--------------------------------------------------
✔ Gradient × Input
✔ Integrated Gradients
✔ CAP-conditioned attribution
✔ Channel × CAP sensitivity
✔ No gradient leakage
✔ Shape-safe
"""

import torch
import numpy as np


# ============================================================================
# Gradient × Input
# ============================================================================
def gradient_x_input(model, x, cap, mask):
    """
    Gradient × Input attribution (time-series safe).
    """
    model.eval()
    model.zero_grad()

    x = x.clone().detach().requires_grad_(True)

    recon, _ = model(x, cap, mask)
    loss = ((recon - x) ** 2)[mask].mean()

    loss.backward()

    attr = x.grad * x
    return attr.detach()


# ============================================================================
# Integrated Gradients
# ============================================================================
def integrated_gradients(
    model,
    x,
    cap,
    mask,
    steps: int = 32,
):
    """
    Integrated Gradients with zero baseline.
    """
    model.eval()

    baseline = torch.zeros_like(x).detach()
    total_grad = torch.zeros_like(x)

    for alpha in torch.linspace(0, 1, steps, device=x.device):
        model.zero_grad()

        x_step = (baseline + alpha * (x - baseline)).detach()
        x_step.requires_grad_(True)

        recon, _ = model(x_step, cap, mask)
        loss = ((recon - x_step) ** 2)[mask].mean()
        loss.backward()

        total_grad += x_step.grad.detach()

    ig = (x - baseline) * total_grad / steps
    return ig.detach()


# ============================================================================
# CAP-conditioned Δ-error attribution (NOVEL)
# ============================================================================
def cap_conditioned_delta(err, cap):
    """
    CAP-conditioned delta-error attribution.
    """
    delta = err[:, 1:] - err[:, :-1]
    cap_mid = cap[:, :-1]

    out = {}
    for k in torch.unique(cap_mid):
        sel = cap_mid == k
        if sel.sum() == 0:
            continue
        out[int(k.item())] = delta[sel].mean().item()

    return out


# ============================================================================
# Channel × CAP sensitivity heatmap
# ============================================================================
def channel_cap_heatmap(xai_map, cap):
    """
    Channel × CAP sensitivity aggregation.

    xai_map: (B, T, C, F)
    cap:     (B, T)
    """
    xai_map = xai_map.abs()

    heat = {}
    for k in torch.unique(cap):
        mask = cap == k
        if mask.sum() == 0:
            continue

        # Expand mask safely
        sel = mask.unsqueeze(-1).unsqueeze(-1).expand_as(xai_map)
        heat[int(k.item())] = xai_map[sel].mean().item()

    return heat
