"""Loss builders for LunarLander regression experiments."""

from __future__ import annotations

from torch import nn

LOSS_TYPES = ("mse", "smooth_l1", "l1")


def build_loss(loss_type: str = "mse") -> nn.Module:
    """Build a regression loss by config name."""

    if loss_type == "mse":
        return nn.MSELoss()
    if loss_type == "smooth_l1":
        return nn.SmoothL1Loss()
    if loss_type == "l1":
        return nn.L1Loss()
    raise ValueError(f"Unsupported loss type: {loss_type}. Expected one of {LOSS_TYPES}.")


__all__ = ["LOSS_TYPES", "build_loss"]
