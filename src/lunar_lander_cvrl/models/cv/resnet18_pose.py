import torch
from torch import nn
from torchvision.models import resnet18

class StateRegressorResNet18(nn.Module):
    def __init__(self, out_dim: int = 4) -> None:
        super().__init__()
        self.backbone = resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.backbone(x)

__all__ = [
    "StateRegressorResNet18",
]

