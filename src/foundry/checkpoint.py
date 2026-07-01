"""Checkpoint save/load for foundry training runs."""

from dataclasses import dataclass
from typing import Any

import torch

from foundry.types import ModelProtocol, OptimizerProtocol


def save_checkpoint(
    model: ModelProtocol, optimizer: OptimizerProtocol, config: dict[str, Any], path: str
) -> None:
    """Save foundry checkpoint to disk."""
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
    }
    torch.save(checkpoint, path)


def validate_checkpoint(checkpoint: dict[str, Any]) -> None:
    """Validate checkpoint integrity.

    Raises:
        ValueError: If checkpoint is corrupted
    """
    if len(checkpoint.get("model", {})) == 0:
        raise ValueError("Checkpoint has empty model state dict")


@dataclass
class ResumeState:
    iter_num: int
    best_val_loss: float
    config: dict[str, Any]


def load_checkpoint(
    model: ModelProtocol, optimizer: OptimizerProtocol | None, path: str, device: str = "cpu"
) -> ResumeState:
    """Load foundry checkpoint from disk.

    Returns:
        ResumeState with iter_num, best_val_loss, and config

    Raises:
        ValueError: If checkpoint is corrupted or missing required fields
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    validate_checkpoint(checkpoint)

    if "iter_num" not in checkpoint:
        raise ValueError("Checkpoint missing iter_num - cannot resume")
    if "best_val_loss" not in checkpoint:
        raise ValueError("Checkpoint missing best_val_loss - cannot resume")

    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    model.load_state_dict(state_dict)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return ResumeState(
        iter_num=checkpoint["iter_num"],
        best_val_loss=checkpoint["best_val_loss"],
        config=checkpoint.get("config", {}),
    )
