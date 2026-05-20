"""Computer-vision model implementations."""

from .custom_cnn import SimpleCNNRegressor
from .efficientnet_pose import StateRegressorEfficientNetB0
from .resnet18_pose import StateRegressorResNet18

CV_MODEL_TYPES = ("resnet18", "efficientnet-b0", "simple-cnn")


def build_cv_model(model_type: str, out_dim: int = 4):
    """Build a CV regressor by registry name."""

    if model_type == "resnet18":
        return StateRegressorResNet18(out_dim=out_dim)
    if model_type == "efficientnet-b0":
        return StateRegressorEfficientNetB0(out_dim=out_dim)
    if model_type == "simple-cnn":
        return SimpleCNNRegressor(out_dim=out_dim)
    raise ValueError(f"Unsupported CV model type: {model_type}")


__all__ = [
    "CV_MODEL_TYPES",
    "SimpleCNNRegressor",
    "StateRegressorEfficientNetB0",
    "StateRegressorResNet18",
    "build_cv_model",
]
