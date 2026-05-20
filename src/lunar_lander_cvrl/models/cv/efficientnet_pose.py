"""EfficientNet regressors for LunarLander CV experiments."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import efficientnet_b0


class StateRegressorEfficientNetB0(nn.Module):
    """EfficientNet-B0 backbone with a small state-regression head."""

    def __init__(self, out_dim: int = 4) -> None:
        super().__init__()
        self.backbone = efficientnet_b0(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, 128),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


__all__ = ["StateRegressorEfficientNetB0"]
