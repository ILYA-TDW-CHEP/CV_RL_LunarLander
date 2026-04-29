"""Train CV regressors for LunarLander frame-to-state prediction."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lunar_lander_cvrl.models.cv import CV_MODEL_TYPES, build_cv_model


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
    ) -> None:
        self.config = config
        self.images_dir = config.images_dir
        self.labels_file = config.labels_file
        self.target_columns = config.target_columns
        self.angle_target = angle_target
        self.augment = augment
        self.particle_prob = float(particle_prob)
        self.rng = np.random.default_rng(seed)
        self.output_columns = _make_output_columns(self.target_columns, angle_target)
        self.samples = _read_label_rows(self.labels_file, self.target_columns)

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

    def _make_target(self, sample: dict[str, float | str]) -> np.ndarray:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--integration",
        default="x_y_theta",
        help="CV integration variant under data/cv_integrations.",
    )
    parser.add_argument(
        "--metadata-path",
        default=None,
        help="Optional explicit path to an integration metadata.json.",
    )
    parser.add_argument(
        "--model-type",
        choices=CV_MODEL_TYPES,
        default="resnet18",
        help="CV regressor architecture to train.",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Checkpoint version folder name. Defaults to <model-type>_<integration>.",
    )
    parser.add_argument(
        "--output-dir",
        default="checkpoints/cv",
        help="Root directory for CV checkpoint versions.",
    )
    parser.add_argument(
        "--angle-target",
        choices=("sincos", "raw"),
        default="sincos",
        help="How to encode theta targets when target_columns contains theta.",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Adam learning rate.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", default="auto", help="Torch device: auto, cpu, or cuda.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--limit-samples", type=int, default=0, help="Optional smoke-test sample limit.")
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable engine-particle augmentation.",
    )
    parser.add_argument(
        "--particle-prob",
        type=float,
        default=0.35,
        help="Probability of adding synthetic engine particles when augmentation is enabled.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_args(args)
    _seed_everything(args.seed)

    device = _resolve_device(args.device)
    config = load_integration_config(args.integration, args.metadata_path)
    dataset = LunarLanderCVDataset(
        config,
        angle_target=args.angle_target,
        augment=not args.no_augment,
        particle_prob=args.particle_prob,
        seed=args.seed,
    )
    if args.limit_samples > 0:
        dataset.samples = dataset.samples[: args.limit_samples]

    train_loader, val_loader = make_loaders(
        dataset=dataset,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = build_model(args.model_type, out_dim=len(dataset.output_columns)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print(f"Integration: {config.name}")
    print(f"Images: {config.images_dir}")
    print(f"Labels: {config.labels_file}")
    print(f"Target columns: {config.target_columns}")
    print(f"Model outputs: {dataset.output_columns}")
    print(f"Model type: {args.model_type}")
    print(f"Device: {device}")

    best_val_loss = float("inf")
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        best_val_loss = min(best_val_loss, val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

    version = args.version or f"{args.model_type.replace('-', '_')}_{config.name}"
    version_dir = Path(args.output_dir) / version
    version_dir.mkdir(parents=True, exist_ok=True)
    weights_path = version_dir / "model.pth"
    torch.save(model.state_dict(), weights_path)

    run_metadata = {
        "version": version,
        "model_type": args.model_type,
        "integration": config.name,
        "integration_metadata_path": str(config.metadata_path),
        "images_dir": str(config.images_dir),
        "labels_file": str(config.labels_file),
        "target_columns": config.target_columns,
        "model_output_columns": dataset.output_columns,
        "angle_target": args.angle_target,
        "out_dim": len(dataset.output_columns),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "augment": not args.no_augment,
        "particle_prob": args.particle_prob,
        "final_train_loss": history[-1]["train_loss"] if history else None,
        "final_val_loss": history[-1]["val_loss"] if history else None,
        "best_val_loss": best_val_loss,
        "weights_file": weights_path.name,
    }
    (version_dir / "training_config.json").write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (version_dir / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved weights to {weights_path}")
    print(f"Saved training metadata to {version_dir / 'training_config.json'}")


def _validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.lr <= 0:
        raise ValueError("--lr must be positive.")
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1.")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")
    if args.limit_samples < 0:
        raise ValueError("--limit-samples cannot be negative.")
    if not 0.0 <= args.particle_prob <= 1.0:
        raise ValueError("--particle-prob must be between 0 and 1.")


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but torch.cuda.is_available() is False.")
    return resolved


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


def _read_label_rows(labels_file: Path, target_columns: list[str]) -> list[dict[str, float | str]]:
    with labels_file.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        required = {"image_name", *target_columns}
        missing = sorted(required - fieldnames)
        if missing:
            raise ValueError(f"Labels file {labels_file} is missing columns: {missing}")

        rows: list[dict[str, float | str]] = []
        for row in reader:
            parsed: dict[str, float | str] = {"image_name": row["image_name"]}
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


def _make_output_columns(target_columns: list[str], angle_target: str) -> list[str]:
    output_columns: list[str] = []
    for column in target_columns:
        if column == "theta" and angle_target == "sincos":
            output_columns.extend(["sin_theta", "cos_theta"])
        else:
            output_columns.append(column)
    return output_columns


def make_loaders(
    dataset: LunarLanderCVDataset,
    val_ratio: float,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    n_total = len(dataset)
    n_val = max(1, int(round(n_total * val_ratio)))
    n_train = n_total - n_val
    if n_train <= 0:
        raise ValueError("Dataset is too small for the requested validation split.")

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader


def build_model(model_type: str, out_dim: int) -> nn.Module:
    return build_cv_model(model_type, out_dim=out_dim)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        preds = model(images)
        loss = criterion(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
    return total_loss / max(1, total_samples)


def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            preds = model(images)
            loss = criterion(preds, targets)
            batch_size = images.size(0)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
    return total_loss / max(1, total_samples)


if __name__ == "__main__":
    main()
