"""Dataloader construction for training."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from foundry.data.dataset import CurriculumSampler, MixtureDataset, TokenDataset

if TYPE_CHECKING:
    import torch

    from foundry.config import RunConfig

Sampler = DistributedSampler[TokenDataset | MixtureDataset] | CurriculumSampler | RandomSampler


def set_sampler_epoch(sampler: Sampler, epoch: int) -> None:
    if isinstance(sampler, (DistributedSampler, CurriculumSampler)):
        sampler.set_epoch(epoch)


def set_sampler_stage(sampler: Sampler, stage: int) -> None:
    if isinstance(sampler, CurriculumSampler):
        sampler.set_stage(stage)


def build_datasets(
    config: RunConfig,
    seed: int,
) -> tuple[TokenDataset | MixtureDataset, TokenDataset | MixtureDataset]:
    """Build train and val datasets from config."""
    data_dir = Path("data") / config.data.dataset

    if config.data.sources:

        def get_train_path(src_path: str) -> str:
            p = Path(src_path)
            if p.stem.endswith("_train"):
                return src_path
            if p.stem.endswith("_val"):
                return str(p.parent / f"{p.stem.replace('_val', '_train')}{p.suffix}")
            return str(p.parent / f"{p.stem}_train{p.suffix}")

        def get_val_path(src_path: str) -> str:
            p = Path(src_path)
            if p.stem.endswith("_val"):
                return src_path
            if p.stem.endswith("_train"):
                return str(p.parent / f"{p.stem.replace('_train', '_val')}{p.suffix}")
            return str(p.parent / f"{p.stem}_val{p.suffix}")

        train_datasets = [
            TokenDataset(get_train_path(src.path), block_size=config.data.block_size)
            for src in config.data.sources
        ]
        val_datasets = [
            TokenDataset(get_val_path(src.path), block_size=config.data.block_size)
            for src in config.data.sources
        ]
        weights = [src.weight for src in config.data.sources]

        return (
            MixtureDataset(train_datasets, weights, seed=seed),
            MixtureDataset(val_datasets, weights, seed=seed),
        )

    return (
        TokenDataset(data_dir / "train.bin", block_size=config.data.block_size),
        TokenDataset(data_dir / "val.bin", block_size=config.data.block_size),
    )


def build_dataloaders(
    config: RunConfig,
    train_dataset: TokenDataset | MixtureDataset,
    val_dataset: TokenDataset | MixtureDataset,
    seed: int,
    world_size: int,
    device_type: str,
) -> tuple[
    DataLoader[tuple[torch.Tensor, torch.Tensor]],
    DataLoader[tuple[torch.Tensor, torch.Tensor]],
    Sampler,
]:
    """Build train/val dataloaders and return (train_loader, val_loader, train_sampler)."""
    use_curriculum = config.data.curriculum.enabled and world_size == 1
    train_sampler: Sampler
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
    elif use_curriculum:
        train_sampler = CurriculumSampler(
            train_dataset,
            num_stages=config.data.curriculum.num_stages,
            schedule=config.data.curriculum.schedule,
            seed=seed,
        )
    else:
        train_sampler = RandomSampler(
            train_dataset, replacement=True, num_samples=len(train_dataset)
        )
    val_sampler: DistributedSampler[TokenDataset | MixtureDataset] | None = (
        DistributedSampler(val_dataset, shuffle=False) if world_size > 1 else None
    )

    num_workers = 0 if device_type == "mps" else 4
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=(device_type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device_type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader, train_sampler
