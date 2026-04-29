"""Computer-vision inference utilities for LunarLander frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from .models.cv.resnet18_pose import StateRegressorResNet18
from torch import nn

import numpy as np
import torch


@dataclass(frozen=True)
class PredictedState:
    """State estimated from a rendered LunarLander RGB frame."""

    x: float
    y: float
    theta: float | None = None
    sin_theta: float | None = None
    cos_theta: float | None = None

    def as_pose_array(self) -> np.ndarray:
        """Return ``[x, y, theta]`` as a float32 array.

        If theta is not predicted by the model, NaN is used as a sentinel value.
        """

        theta = np.nan if self.theta is None else self.theta
        return np.array([self.x, self.y, theta], dtype=np.float32)

    def as_model_array(self) -> np.ndarray:
        """Return ``[x, y, sin(theta), cos(theta)]`` as a float32 array."""

        return np.array(
            [
                self.x,
                self.y,
                np.nan if self.sin_theta is None else self.sin_theta,
                np.nan if self.cos_theta is None else self.cos_theta,
            ],
            dtype=np.float32,
        )


PredictedPose = PredictedState


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
        output_columns: Sequence[str] | None = None,
        strict: bool = True,
    ) -> None:
        self.device = _resolve_device(device)
        self.model = model if model is not None else StateRegressorResNet18()
        self.output_columns = tuple(output_columns or ("x", "y", "sin_theta", "cos_theta"))

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
        """Return raw model output in ``self.output_columns`` order."""

        inputs = self.preprocess(frame).to(self.device)
        with torch.inference_mode():
            prediction = self.model(inputs).squeeze(0)
        raw = prediction.detach().cpu().numpy().astype(np.float32)
        if raw.shape[0] != len(self.output_columns):
            raise ValueError(
                "CV model output size does not match output_columns: "
                f"got {raw.shape[0]}, expected {len(self.output_columns)} "
                f"for {self.output_columns}.",
            )
        return raw

    def predict_state(self, frame: np.ndarray | torch.Tensor) -> PredictedState:
        """Return the state predicted by the configured CV model."""

        raw = self.predict_raw(frame)
        values = {column: float(value) for column, value in zip(self.output_columns, raw)}
        if "x" not in values or "y" not in values:
            raise ValueError(f"CV output_columns must include x and y, got {self.output_columns}.")

        theta = values.get("theta")
        sin_theta = values.get("sin_theta")
        cos_theta = values.get("cos_theta")

        if theta is None and sin_theta is not None and cos_theta is not None:
            norm = float(np.hypot(sin_theta, cos_theta))
            if norm > 1e-8:
                sin_theta /= norm
                cos_theta /= norm
            else:
                sin_theta = 0.0
                cos_theta = 1.0
            theta = float(np.arctan2(sin_theta, cos_theta))
        elif theta is not None and (sin_theta is None or cos_theta is None):
            sin_theta = float(np.sin(theta))
            cos_theta = float(np.cos(theta))

        return PredictedState(
            x=values["x"],
            y=values["y"],
            theta=theta,
            sin_theta=sin_theta,
            cos_theta=cos_theta,
        )

    def predict_pose(self, frame: np.ndarray | torch.Tensor) -> PredictedState:
        """Backward-compatible alias for ``predict_state``."""

        return self.predict_state(frame)
