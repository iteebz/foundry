"""LR Finder: Leslie Smith's range test, reference implementation."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from torch.optim import Optimizer


@dataclass(frozen=True)
class LRFinderResult:
    lrs: list[float]
    losses: list[float]
    suggested_lr: float
    steepest_lr: float
    min_loss_lr: float

    def plot(self, out_path: str | Path = "out/lr_find.png") -> None:
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

        _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(self.lrs, self.losses)
        ax1.set_xscale("log")
        ax1.set_xlabel("Learning Rate")
        ax1.set_ylabel("Smoothed Loss")
        ax1.set_title("LR Finder")
        ax1.grid(True, alpha=0.3)
        ax1.axvline(
            self.suggested_lr,
            color="g",
            linestyle="--",
            label=f"Suggested: {self.suggested_lr:.2e}",
        )
        ax1.axvline(
            self.steepest_lr, color="r", linestyle=":", label=f"Steepest: {self.steepest_lr:.2e}"
        )
        ax1.axvline(
            self.min_loss_lr,
            color="b",
            linestyle=":",
            alpha=0.5,
            label=f"Min loss: {self.min_loss_lr:.2e}",
        )
        ax1.legend()

        if len(self.lrs) > 1:
            log_lrs = np.log10(self.lrs)
            gradients = np.gradient(self.losses, log_lrs)
            ax2.plot(self.lrs, gradients)
            ax2.set_xscale("log")
            ax2.set_xlabel("Learning Rate")
            ax2.set_ylabel("d(loss)/d(log_lr)")
            ax2.set_title("Loss Gradient (steepest descent = most negative)")
            ax2.grid(True, alpha=0.3)
            ax2.axhline(0, color="k", linestyle="-", alpha=0.3)
            ax2.axvline(
                self.steepest_lr,
                color="r",
                linestyle=":",
                label=f"Steepest: {self.steepest_lr:.2e}",
            )
            ax2.legend()

        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()


def lr_find(
    model: nn.Module,
    optimizer: Optimizer,
    data_iter: Iterator[tuple[Tensor, Tensor]],
    loss_fn: callable,
    *,
    min_lr: float = 1e-8,
    max_lr: float = 10.0,
    num_steps: int = 100,
    smooth_beta: float = 0.98,
    diverge_thresh: float = 4.0,
    grad_accum_steps: int = 1,
) -> LRFinderResult:
    """
    Find optimal learning rate using exponential range test.

    Uses steepest descent heuristic: the best LR is where loss decreases fastest,
    not where loss is minimum (which is often too high and about to diverge).

    Args:
        model: Model to train (must be in train mode)
        optimizer: Optimizer with all param groups configured
        data_iter: Iterator yielding (x, y) batches
        loss_fn: Callable (model, x, y) -> loss tensor
        min_lr: Starting learning rate
        max_lr: Ending learning rate
        num_steps: Number of LR steps to test
        smooth_beta: EMA smoothing factor for loss
        diverge_thresh: Stop if smoothed_loss > diverge_thresh * best_loss
        grad_accum_steps: Accumulate gradients over N batches (matches real training)

    Returns:
        LRFinderResult with suggested learning rate and diagnostics
    """
    import numpy as np

    model.train()

    initial_state = {k: v.clone() for k, v in model.state_dict().items()}
    initial_opt_state = copy.deepcopy(optimizer.state_dict())

    lr_mult = (max_lr / min_lr) ** (1.0 / num_steps)

    lrs: list[float] = []
    losses: list[float] = []
    raw_losses: list[float] = []
    avg_loss = 0.0
    best_loss = float("inf")

    for step in range(num_steps):
        lr = min_lr * (lr_mult**step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        accumulated_loss = 0.0
        for _micro_step in range(grad_accum_steps):
            try:
                x, y = next(data_iter)
            except StopIteration:
                break

            loss = loss_fn(model, x, y)
            (loss / grad_accum_steps).backward()
            accumulated_loss += loss.item()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        batch_loss = accumulated_loss / grad_accum_steps
        raw_losses.append(batch_loss)

        avg_loss = smooth_beta * avg_loss + (1 - smooth_beta) * batch_loss
        smoothed_loss = avg_loss / (1 - smooth_beta ** (step + 1))

        lrs.append(lr)
        losses.append(smoothed_loss)

        if smoothed_loss < best_loss:
            best_loss = smoothed_loss

        if math.isnan(smoothed_loss) or math.isinf(smoothed_loss):
            break
        if smoothed_loss > diverge_thresh * best_loss and step > 10:
            break

    model.load_state_dict(initial_state)
    optimizer.load_state_dict(initial_opt_state)

    if len(lrs) < 3:
        return LRFinderResult(
            lrs=lrs,
            losses=losses,
            suggested_lr=min_lr,
            steepest_lr=min_lr,
            min_loss_lr=min_lr,
        )

    log_lrs = np.log10(lrs)
    gradients = np.gradient(losses, log_lrs)

    steepest_idx = int(np.argmin(gradients))
    steepest_lr = lrs[steepest_idx]

    min_loss_idx = int(np.argmin(losses))
    min_loss_lr = lrs[min_loss_idx]

    suggested_lr = steepest_lr / 10.0

    return LRFinderResult(
        lrs=lrs,
        losses=losses,
        suggested_lr=suggested_lr,
        steepest_lr=steepest_lr,
        min_loss_lr=min_loss_lr,
    )


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(dtype: str) -> str:
    if dtype != "auto":
        return dtype
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return "bfloat16"
    return "float16"


def _build_dataset(config):
    from pathlib import Path

    from foundry.data.dataset import MixtureDataset, TokenDataset

    data_dir = Path("data") / config.data.dataset

    if not config.data.sources:
        return TokenDataset(data_dir / "train.bin", block_size=config.data.block_size)

    def get_train_path(src_path: str) -> str:
        p = Path(src_path)
        if p.stem.endswith("_train"):
            return src_path
        if p.stem.endswith("_val"):
            return str(p.parent / f"{p.stem.replace('_val', '_train')}{p.suffix}")
        return str(p.parent / f"{p.stem}_train{p.suffix}")

    train_datasets = [
        TokenDataset(get_train_path(src.path), block_size=config.data.block_size)
        for src in config.data.sources
    ]
    weights = [src.weight for src in config.data.sources]
    return MixtureDataset(train_datasets, weights, seed=config.training.seed or 42)


def lr_find_from_config(config_path: str | Path) -> LRFinderResult:
    from contextlib import nullcontext
    from pathlib import Path

    from torch.utils.data import DataLoader

    from foundry.config import RunConfig
    from foundry.model import GPT

    config = RunConfig.from_yaml(Path(config_path))

    device = _resolve_device(config.training.device)
    dtype = _resolve_dtype(config.training.dtype)
    device_type = "cuda" if "cuda" in device else "mps" if "mps" in device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[
        dtype
    ]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )

    train_dataset = _build_dataset(config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    model = GPT(config.model)
    model.to(device)

    optimizer = model.configure_optimizers(
        config.training.weight_decay,
        1e-7,
        (config.training.beta1, config.training.beta2),
        device_type,
    )

    def data_iter():
        while True:
            for x, y in train_loader:
                yield x.to(device), y.to(device)

    def loss_fn(m, x, y):
        with ctx:
            _, loss = m(x, y)
        return loss

    return lr_find(
        model=model,
        optimizer=optimizer,
        data_iter=data_iter(),
        loss_fn=loss_fn,
        grad_accum_steps=config.training.gradient_accumulation_steps,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        sys.exit(1)

    result = lr_find_from_config(sys.argv[1])
    result.plot()
