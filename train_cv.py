"""Train CV regressors for LunarLander frame-to-state prediction."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

try:
    import hydra
except ImportError:  # pragma: no cover - exercised only without optional deps.
    hydra = None

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lunar_lander_cvrl.dataset import (
    CVIntegrationConfig,
    LabelSample,
    LunarLanderCVDataset,
    load_integration_config,
)
from lunar_lander_cvrl.dataloader import make_loaders
from lunar_lander_cvrl.losses import LOSS_TYPES, build_loss
from lunar_lander_cvrl.models.cv import build_cv_model
from lunar_lander_cvrl.training import evaluate_loss, run_epoch


def run_training(args: SimpleNamespace) -> None:
    _validate_args(args)
    _seed_everything(args.seed)

    device = _resolve_device(args.device)
    config = load_integration_config(args.integration, args.metadata_path)
    dataset = LunarLanderCVDataset(
        config,
        angle_target=args.angle_target,
        augment=args.augment,
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
    criterion = build_loss(args.loss_type)

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
        "loss_type": args.loss_type,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "augment": args.augment,
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


def _config_to_args(cfg) -> SimpleNamespace:
    return SimpleNamespace(
        integration=cfg.data.integration,
        metadata_path=cfg.data.metadata_path,
        model_type=cfg.model.type,
        version=cfg.output.version,
        output_dir=cfg.output.dir,
        angle_target=cfg.data.angle_target,
        epochs=cfg.train.epochs,
        batch_size=cfg.train.batch_size,
        lr=cfg.optimizer.lr,
        loss_type=cfg.loss.type,
        val_ratio=cfg.data.val_ratio,
        seed=cfg.seed,
        device=cfg.device,
        num_workers=cfg.data.num_workers,
        limit_samples=cfg.data.limit_samples,
        augment=bool(cfg.augmentation.enabled),
        particle_prob=cfg.augmentation.particle_prob,
    )


def _run_hydra_main() -> None:
    if hydra is None:
        raise SystemExit(
            "Hydra is required for config-driven CV training. "
            "Install dependencies with: pip install -r requirements.txt",
        )

    @hydra.main(version_base=None, config_path="configs/cv", config_name="train")
    def _main(cfg) -> None:
        run_training(_config_to_args(cfg))

    _main()


def main() -> None:
    _run_hydra_main()


def _validate_args(args: SimpleNamespace) -> None:
    if args.epochs <= 0:
        raise ValueError("train.epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("train.batch_size must be positive.")
    if args.lr <= 0:
        raise ValueError("optimizer.lr must be positive.")
    if args.loss_type not in LOSS_TYPES:
        raise ValueError(f"loss.type must be one of {LOSS_TYPES}, got {args.loss_type!r}.")
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("data.val_ratio must be between 0 and 1.")
    if args.num_workers < 0:
        raise ValueError("data.num_workers cannot be negative.")
    if args.limit_samples < 0:
        raise ValueError("data.limit_samples cannot be negative.")
    if not 0.0 <= args.particle_prob <= 1.0:
        raise ValueError("augmentation.particle_prob must be between 0 and 1.")


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


def build_model(model_type: str, out_dim: int) -> nn.Module:
    return build_cv_model(model_type, out_dim=out_dim)


if __name__ == "__main__":
    main()
