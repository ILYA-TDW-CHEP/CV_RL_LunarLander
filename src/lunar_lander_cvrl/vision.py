"""Computer-vision inference utilities for LunarLander frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from .models.cv.resnet18_pose import StateRegressorResNet18
from torch import nn

import numpy as np
import torch


@dataclass(frozen=True)
class PredictedPose:
    """Pose estimated from a rendered LunarLander RGB frame."""

    x: float
    y: float
    theta: float
    sin_theta: float
    cos_theta: float

    def as_pose_array(self) -> np.ndarray:
        """Return ``[x, y, theta]`` as a float32 array."""

        return np.array([self.x, self.y, self.theta], dtype=np.float32)

    def as_model_array(self) -> np.ndarray:
        """Return ``[x, y, sin(theta), cos(theta)]`` as a float32 array."""

        return np.array(
            [self.x, self.y, self.sin_theta, self.cos_theta],
            dtype=np.float32,
        )


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but torch.cuda.is_available() is False.")

    return resolved


def _load_state_dict(weights_path: str | Path, device: torch.device) -> dict[str, Any]:
    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(f"CV weights not found: {path}")

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict"):
            nested = checkpoint.get(key)
            if isinstance(nested, dict):
                return nested
        return checkpoint

    raise TypeError(
        f"Unsupported checkpoint type {type(checkpoint)!r}; expected a state_dict.",
    )


class StatePredictor:
    """Load a trained CV model and predict LunarLander pose from RGB frames."""

    def __init__(
        self,
        weights_path: str | Path,
        device: str | torch.device = "auto",
        model: nn.Module | None = None,
        strict: bool = True,
    ) -> None:
        self.device = _resolve_device(device)
        self.model = model if model is not None else StateRegressorResNet18()

        state_dict = _load_state_dict(weights_path, self.device)
        self.model.load_state_dict(state_dict, strict=strict)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def preprocess(frame: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Convert an ``HWC`` RGB frame to a batched ``NCHW`` float tensor."""

        if isinstance(frame, torch.Tensor):
            tensor = frame.detach().clone()
            if tensor.ndim != 3:
                raise ValueError(f"Expected a 3D image tensor, got shape {tuple(tensor.shape)}")
            if tensor.shape[-1] in (1, 3, 4):
                tensor = tensor.permute(2, 0, 1)
            elif tensor.shape[0] not in (1, 3, 4):
                raise ValueError(
                    "Expected a CHW or HWC image tensor with 1, 3, or 4 channels, "
                    f"got shape {tuple(tensor.shape)}",
                )
            if tensor.shape[0] == 4:
                tensor = tensor[:3]
            if tensor.shape[0] == 1:
                tensor = tensor.repeat(3, 1, 1)
            tensor = tensor.float()
        else:
            array = np.asarray(frame)
            if array.ndim != 3:
                raise ValueError(f"Expected an HWC RGB image, got shape {array.shape}")
            if array.shape[2] not in (1, 3, 4):
                raise ValueError(f"Expected 1, 3, or 4 image channels, got {array.shape[2]}")
            if array.shape[2] == 4:
                array = array[:, :, :3]
            if array.shape[2] == 1:
                array = np.repeat(array, 3, axis=2)
            tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).float()

        if tensor.max().item() > 1.0:
            tensor = tensor / 255.0

        return tensor.unsqueeze(0)

    def predict_raw(self, frame: np.ndarray | torch.Tensor) -> np.ndarray:
        """Return raw model output ``[x, y, sin(theta), cos(theta)]``."""

        inputs = self.preprocess(frame).to(self.device)
        with torch.inference_mode():
            prediction = self.model(inputs).squeeze(0)
        return prediction.detach().cpu().numpy().astype(np.float32)

    def predict_pose(self, frame: np.ndarray | torch.Tensor) -> PredictedPose:
        """Return normalized ``x``, ``y`` and angle from a rendered frame."""

        raw = self.predict_raw(frame)
        sin_theta = float(raw[2])
        cos_theta = float(raw[3])
        norm = float(np.hypot(sin_theta, cos_theta))
        if norm > 1e-8:
            sin_theta /= norm
            cos_theta /= norm
        else:
            sin_theta = 0.0
            cos_theta = 1.0

        theta = float(np.arctan2(sin_theta, cos_theta))
        return PredictedPose(
            x=float(raw[0]),
            y=float(raw[1]),
            theta=theta,
            sin_theta=sin_theta,
            cos_theta=cos_theta,
        )

    def predict_state(self, frame: np.ndarray | torch.Tensor) -> np.ndarray:
        """Return the CV state vector ``[x, y, sin(theta), cos(theta)]``."""

        return self.predict_pose(frame).as_model_array()
