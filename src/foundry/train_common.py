"""Shared training boilerplate for SFT and DPO loops.

Extracted from train.py and train_dpo.py which duplicated ~60% of their
setup code (distributed init, device/dtype resolution, AMP context,
LoRA, optimizer, scaler step, checkpoint save, LR schedule, signal handler).
"""

from __future__ import annotations

import math
import signal
import sys
import traceback
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from types import FrameType

    from foundry.config import RunConfig

from foundry.distributed import init_distributed
from foundry.metrics import MetricLogger
from foundry.model import GPT


def _sigint_handler(sig: int, frame: FrameType | None) -> None:
    traceback.print_stack(frame)
    sys.exit(1)


signal.signal(signal.SIGINT, _sigint_handler)


def get_lr(it: int, config: RunConfig) -> float:
    if it < config.training.warmup_iters:
        return config.training.learning_rate * (it + 1) / (config.training.warmup_iters + 1)
    if it > config.training.lr_decay_iters:
        return config.training.min_lr
    decay_ratio = (it - config.training.warmup_iters) / (
        config.training.lr_decay_iters - config.training.warmup_iters
    )
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.training.min_lr + coeff * (config.training.learning_rate - config.training.min_lr)


def resolve_device(config: RunConfig) -> str:
    device = config.training.device
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def resolve_dtype(config: RunConfig) -> str:
    dtype = config.training.dtype
    if dtype == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return "bfloat16"
        return "float16"
    return dtype


def save_checkpoint(
    raw_model: Any,
    optimizer: Any,
    config: RunConfig,
    iter_num: int,
    best_val_loss: float,
    out_dir: Path,
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    checkpoint = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config.to_dict(),
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
    }
    if extra:
        checkpoint.update(extra)
    ckpt_path = out_dir / "ckpt.pt"
    tmp_path = ckpt_path.with_suffix(".tmp")
    torch.save(checkpoint, tmp_path)
    tmp_path.replace(ckpt_path)


_PTDTYPE = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


@dataclass
class TrainContext:
    """Everything both training loops need from the shared setup preamble."""

    config: RunConfig
    model: GPT
    optimizer: torch.optim.Optimizer
    scaler: torch.amp.GradScaler
    amp_ctx: Any  # autocast context or nullcontext
    device: str
    device_type: str
    dtype: str
    out_dir: Path
    master_process: bool
    rank: int
    world_size: int
    effective_grad_accum: int
    metric_logger: MetricLogger | None
    is_ddp: bool
    is_fsdp: bool
    raw_model: GPT


def setup_training(config: RunConfig) -> TrainContext:
    """Shared training setup: distributed, device, model, optimizer, LoRA, scaler."""
    from foundry.distributed import print_distributed_info, wrap_model_distributed

    master_process, rank, world_size = init_distributed(backend="nccl")

    effective_grad_accum = config.training.gradient_accumulation_steps
    if world_size > 1:
        assert effective_grad_accum % world_size == 0
        effective_grad_accum //= world_size

    out_dir = Path(config.training.out_dir) / config.name
    if master_process:
        out_dir.mkdir(parents=True, exist_ok=True)

    metric_logger = MetricLogger(str(out_dir)) if master_process else None

    seed = config.training.seed
    torch.manual_seed((seed or 0) + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = resolve_device(config)
    dtype = resolve_dtype(config)
    device_type = "cuda" if "cuda" in device else "mps" if "mps" in device else "cpu"
    ptdtype = _PTDTYPE[dtype]
    use_amp = device_type == "cuda" and dtype in ("float16", "bfloat16")
    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype) if use_amp else nullcontext()

    model: GPT | torch.nn.Module = GPT(config.model)
    model.to(device)

    if config.lora.enabled:
        from foundry.lora import apply_lora_to_model

        model = apply_lora_to_model(
            model,
            r=config.lora.r,
            lora_alpha=config.lora.lora_alpha,
            lora_dropout=config.lora.lora_dropout,
        )

    assert isinstance(model, GPT)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and dtype == "float16"))
    optimizer = model.configure_optimizers(
        config.training.weight_decay,
        config.training.learning_rate,
        (config.training.beta1, config.training.beta2),
        device_type,
    )

    if config.training.compile:
        model = torch.compile(model, mode=config.training.compile_mode)

    model, is_ddp, is_fsdp = wrap_model_distributed(
        model,
        strategy=config.training.distributed,
        fsdp_min_params=config.training.fsdp_min_params,
    )

    if master_process:
        print_distributed_info(model, is_ddp, is_fsdp)

    raw_model = model.module if (is_ddp or is_fsdp) else model

    return TrainContext(
        config=config,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        amp_ctx=amp_ctx,
        device=device,
        device_type=device_type,
        dtype=dtype,
        out_dir=out_dir,
        master_process=master_process,
        rank=rank,
        world_size=world_size,
        effective_grad_accum=effective_grad_accum,
        metric_logger=metric_logger,
        is_ddp=is_ddp,
        is_fsdp=is_fsdp,
        raw_model=raw_model,
    )


def step_optimizer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    grad_clip: float,
) -> None:
    """Grad clip + scaler step + zero_grad (was duplicated verbatim)."""
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
