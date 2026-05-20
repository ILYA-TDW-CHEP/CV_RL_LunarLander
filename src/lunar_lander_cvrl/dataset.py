"""Dataset utilities for LunarLander CV frame-to-state regression."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

LabelSample = dict[str, float | str]


@dataclass(frozen=True)
class CVIntegrationConfig:
    name: str
    metadata_path: Path
    images_dir: Path
    labels_file: Path
    target_columns: list[str]
    raw_metadata: dict[str, Any]


class LunarLanderCVDataset(Dataset):
    """Dataset driven by a CV integration metadata file."""

    def __init__(
        self,
        config: CVIntegrationConfig,
        angle_target: str = "sincos",
        augment: bool = True,
        particle_prob: float = 0.35,
        seed: int = 42,
        samples: list[LabelSample] | None = None,
    ) -> None:
        self.config = config
        self.images_dir = config.images_dir
        self.labels_file = config.labels_file
        self.target_columns = config.target_columns
        self.angle_target = angle_target
        self.augment = augment
        self.particle_prob = float(particle_prob)
        self.rng = np.random.default_rng(seed)
        self.output_columns = make_output_columns(self.target_columns, angle_target)
        self.samples = (
            list(samples)
            if samples is not None
            else _read_label_rows(self.labels_file, self.target_columns)
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        image = np.load(self.images_dir / sample["image_name"]).astype(np.float32)
        if image.max(initial=0.0) > 1.0:
            image = image / 255.0

        if self.augment and all(key in sample for key in ("x", "y", "theta")):
            image = self._add_engine_particles(image, sample["x"], sample["y"], sample["theta"])

        image = np.transpose(image[:, :, :3], (2, 0, 1))
        target = self._make_target(sample)
        return torch.from_numpy(np.ascontiguousarray(image)), torch.from_numpy(target)

    def _make_target(self, sample: LabelSample) -> np.ndarray:
        values: list[float] = []
        for column in self.target_columns:
            value = float(sample[column])
            if column == "theta" and self.angle_target == "sincos":
                values.extend([math.sin(value), math.cos(value)])
            else:
                values.append(value)
        return np.asarray(values, dtype=np.float32)

    def _obs_to_pixel(self, x_obs: float, y_obs: float, h: int, w: int) -> tuple[int, int]:
        px = int(np.clip((x_obs + 1.0) * 0.5 * w, 0, w - 1))
        py = int(np.clip(h * (0.705 - 0.5 * y_obs), 0, h - 1))
        return px, py

    def _draw_disk(
        self,
        image: np.ndarray,
        cx: int,
        cy: int,
        radius: int,
        color: np.ndarray,
        alpha: float = 0.6,
    ) -> np.ndarray:
        h, w, _ = image.shape
        x_min = max(0, cx - radius)
        x_max = min(w, cx + radius + 1)
        y_min = max(0, cy - radius)
        y_max = min(h, cy + radius + 1)
        if x_min >= x_max or y_min >= y_max:
            return image

        yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
        patch = image[y_min:y_max, x_min:x_max].copy()
        patch[mask] = (1.0 - alpha) * patch[mask] + alpha * color
        image[y_min:y_max, x_min:x_max] = patch
        return image

    def _add_engine_particles(
        self,
        image: np.ndarray,
        x_obs: float,
        y_obs: float,
        theta: float,
    ) -> np.ndarray:
        if self.rng.random() > self.particle_prob:
            return image

        h, w, _ = image.shape
        cx, cy = self._obs_to_pixel(float(x_obs), float(y_obs), h, w)
        body_size = max(10.0, 0.03 * w)
        down = np.array([math.sin(theta), math.cos(theta)], dtype=np.float32)
        right = np.array([math.cos(theta), -math.sin(theta)], dtype=np.float32)
        center = np.array([cx, cy], dtype=np.float32)
        main_nozzle = center + 0.85 * body_size * down
        left_nozzle = center + 0.65 * body_size * down - 0.50 * body_size * right
        right_nozzle = center + 0.65 * body_size * down + 0.50 * body_size * right

        engine_specs = []
        if self.rng.random() < 0.75:
            engine_specs.append((main_nozzle, 4, 8, 0.25, 1.30))
        if self.rng.random() < 0.30:
            engine_specs.append((left_nozzle, 2, 4, 0.15, 0.75))
        if self.rng.random() < 0.30:
            engine_specs.append((right_nozzle, 2, 4, 0.15, 0.75))

        for nozzle, n_min, n_max, spread_scale, length_scale in engine_specs:
            n_particles = int(self.rng.integers(n_min, n_max + 1))
            for _ in range(n_particles):
                dist = float(self.rng.uniform(0.25 * body_size, length_scale * body_size))
                lateral = float(self.rng.normal(0.0, spread_scale * body_size))
                pos = nozzle + dist * down + lateral * right
                radius = int(self.rng.integers(2, 6))
                alpha = float(self.rng.uniform(0.35, 0.75))
                color = np.array(
                    [
                        self.rng.uniform(0.90, 1.00),
                        self.rng.uniform(0.10, 0.35),
                        self.rng.uniform(0.00, 0.08),
                    ],
                    dtype=np.float32,
                )
                image = self._draw_disk(
                    image,
                    int(round(pos[0])),
                    int(round(pos[1])),
                    radius,
                    color,
                    alpha=alpha,
                )
        return image


def load_integration_config(integration: str, metadata_path: str | None) -> CVIntegrationConfig:
    path = (
        Path(metadata_path)
        if metadata_path is not None
        else Path("data") / "cv_integrations" / integration / "metadata.json"
    )
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Integration metadata not found: {path}")

    metadata = json.loads(path.read_text(encoding="utf-8"))
    target_columns = metadata.get("target_columns")
    if not isinstance(target_columns, list) or not all(isinstance(c, str) for c in target_columns):
        raise ValueError(f"metadata target_columns must be a list of strings: {path}")

    root = path.parent
    images_dir = (root / metadata.get("images_dir", "../../images")).resolve()
    labels_file = (root / metadata.get("labels_file", "../../labels.csv")).resolve()
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_file.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_file}")

    return CVIntegrationConfig(
        name=str(metadata.get("name", integration)),
        metadata_path=path,
        images_dir=images_dir,
        labels_file=labels_file,
        target_columns=target_columns,
        raw_metadata=metadata,
    )


def make_output_columns(target_columns: list[str], angle_target: str) -> list[str]:
    output_columns: list[str] = []
    for column in target_columns:
        if column == "theta" and angle_target == "sincos":
            output_columns.extend(["sin_theta", "cos_theta"])
        else:
            output_columns.append(column)
    return output_columns


def _read_label_rows(labels_file: Path, target_columns: list[str]) -> list[LabelSample]:
    with labels_file.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        required = {"image_name", *target_columns}
        missing = sorted(required - fieldnames)
        if missing:
            raise ValueError(f"Labels file {labels_file} is missing columns: {missing}")

        rows: list[LabelSample] = []
        for row in reader:
            parsed: LabelSample = {"image_name": row["image_name"]}
            for key, value in row.items():
                if key == "image_name" or value is None or value == "":
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
    if not rows:
        raise ValueError(f"Labels file is empty: {labels_file}")
    return rows


__all__ = [
    "CVIntegrationConfig",
    "LabelSample",
    "LunarLanderCVDataset",
    "load_integration_config",
    "make_output_columns",
]
