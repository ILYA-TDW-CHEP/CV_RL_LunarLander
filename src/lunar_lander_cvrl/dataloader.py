"""DataLoader builders for LunarLander CV datasets."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from .dataset import LunarLanderCVDataset


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

    split_generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_total, generator=split_generator).tolist()
    train_samples = [dataset.samples[i] for i in indices[:n_train]]
    val_samples = [dataset.samples[i] for i in indices[n_train:]]

    train_ds = LunarLanderCVDataset(
        dataset.config,
        angle_target=dataset.angle_target,
        augment=dataset.augment,
        particle_prob=dataset.particle_prob,
        seed=seed,
        samples=train_samples,
    )
    val_ds = LunarLanderCVDataset(
        dataset.config,
        angle_target=dataset.angle_target,
        augment=False,
        particle_prob=dataset.particle_prob,
        seed=seed + 1,
        samples=val_samples,
    )

    loader_generator = torch.Generator().manual_seed(seed + 2)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader


__all__ = ["make_loaders"]
