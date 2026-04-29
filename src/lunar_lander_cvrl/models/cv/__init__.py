"""Computer-vision model implementations."""

from .custom_cnn import SimpleCNNRegressor
from .resnet18_pose import StateRegressorResNet18

__all__ = ["SimpleCNNRegressor",
           "StateRegressorResNet18",
           ]
