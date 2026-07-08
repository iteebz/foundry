"""Training script with RunConfig (v2 - clean refactor)."""

from __future__ import annotations

import math
import pickle
import signal
import sys
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import yaml

from foundry.config import RunConfig
from foundry.data.curriculum import get_curriculum_stage
from foundry.data.loaders import (
    build_dataloaders,
    build_datasets,
    set_sampler_epoch,
    set_sampler_stage,
)
from foundry.distributed import (
    cleanup_distributed,
    init_distributed,
    print_distributed_info,
    wrap_model_distributed,
)
from foundry.metrics import MetricLogger
from foundry.model import GPT


def check_m4_safety() -> None:
    import platform
    import subprocess

    if platform.system() == "Darwin":
        try:
            # Enforce human constraint d/8b11c2c4: No training on M4
            # Using absolute path for sysctl to satisfy S607
            brand_bytes = subprocess.check_output(
                ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
                stderr=subprocess.DEVNULL,
            )
            brand = brand_bytes.decode().strip()
            if "M4" in brand:
                sys.stderr.write(f"FATAL: Training blocked on {brand} (d/8b11c2c4)\n")
                sys.exit(1)
        except (subprocess.SubprocessError, OSError):
            # If sysctl fails, we cannot verify, so we allow it to proceed.
            pass


if TYPE_CHECKING:
    from types import FrameType


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


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {
            k: v.clone().detach() for k, v in model.state_dict().items()
        }

    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)


def train(config_path: str | Path) -> None:
    check_m4_safety()
    config = RunConfig.from_yaml(Path(config_path))

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
    assert seed is not None
    torch.manual_seed(seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = config.training.device
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    dtype = config.training.dtype
    if dtype == "auto":
        dtype = (
            "bfloat16"
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else "float16"
        )

    device_type = "cuda" if "cuda" in device else "mps" if "mps" in device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[
        dtype
    ]
    use_amp = device_type == "cuda" and dtype in ("float16", "bfloat16")
    ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype) if use_amp else nullcontext()

    meta_path = Path("data") / config.data.dataset / "meta.pkl"
    if meta_path.exists():
        with meta_path.open("rb") as f:
            meta = pickle.load(f)  # noqa: S301
        config.model.vocab_size = meta["vocab_size"]

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

    iter_num = 0
    best_val_loss = 1e9

    if config.training.init_from == "resume":
        from foundry.checkpoint import load_checkpoint

        ckpt_path = out_dir / "ckpt.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Cannot resume: {ckpt_path} not found")
        resume_state = load_checkpoint(model, optimizer, str(ckpt_path), device=device)
        iter_num = resume_state.iter_num
        best_val_loss = resume_state.best_val_loss
    elif config.training.init_from != "scratch":
        from foundry.checkpoint import load_checkpoint

        ckpt_path = Path(config.training.init_from)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        load_checkpoint(model, None, str(ckpt_path), device=device)

    if config.training.compile:
        model = torch.compile(model, mode=config.training.compile_mode)

    model, is_ddp, is_fsdp = wrap_model_distributed(
        model,
        strategy=config.training.distributed,
        fsdp_min_params=config.training.fsdp_min_params,
    )

    if master_process:
        print_distributed_info(model, is_ddp, is_fsdp)

    ema_model = EMA(model, decay=config.training.ema_decay) if config.training.use_ema else None

    raw_model = model.module if (is_ddp or is_fsdp) else model

    train_dataset, val_dataset = build_datasets(config, seed)
    use_curriculum = config.data.curriculum.enabled and world_size == 1
    train_loader, val_loader, train_sampler = build_dataloaders(
        config,
        train_dataset,
        val_dataset,
        seed,
        world_size,
        device_type,
    )

    @torch.no_grad()
    def estimate_loss() -> dict[str, float]:
        out: dict[str, float] = {}
        model.eval()
        for split, loader in [("train", train_loader), ("val", val_loader)]:
            losses: list[float] = []
            for k, (x, y) in enumerate(loader):
                if k >= config.training.eval_iters:
                    break
                x, y = x.to(device), y.to(device)
                with ctx:
                    _logits, loss = model(x, y)
                losses.append(loss.item())
            out[split] = float(np.mean(losses))
        model.train()
        return out

    current_epoch = 0
    last_gns: float | None = None
    train_iter = iter(train_loader)
    set_sampler_epoch(train_sampler, current_epoch)

    total_epochs = max(1, config.training.max_iters // len(train_loader))

    while True:
        try:
            batch_x, batch_y = next(train_iter)
        except StopIteration:
            current_epoch += 1
            set_sampler_epoch(train_sampler, current_epoch)
            if use_curriculum:
                new_stage = get_curriculum_stage(
                    current_epoch, total_epochs, config.data.curriculum.num_stages
                )
                set_sampler_stage(train_sampler, new_stage)
            train_iter = iter(train_loader)
            batch_x, batch_y = next(train_iter)

        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        lr = get_lr(iter_num, config) if config.training.decay_lr else config.training.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if iter_num > 0 and iter_num % config.training.eval_interval == 0 and master_process:
            losses = estimate_loss()

            metrics: dict[str, Any] = {
                "iter": iter_num,
                "train_loss": float(losses["train"]),
                "val_loss": float(losses["val"]),
                "lr": lr,
                "batch_size": config.data.batch_size * effective_grad_accum * world_size,
            }
            if last_gns is not None:
                metrics["gns"] = last_gns

            if metric_logger:
                metric_logger.log(metrics)

            if config.wandb.enabled:
                import wandb

                wandb.log(
                    {
                        "iter": iter_num,
                        "train/loss": losses["train"],
                        "val/loss": losses["val"],
                        "lr": lr,
                        "batch_size": config.data.batch_size * effective_grad_accum * world_size,
                        **(({"train/gns": last_gns}) if last_gns is not None else {}),
                    }
                )

            if losses["val"] < best_val_loss or config.training.always_save_checkpoint:
                best_val_loss = losses["val"]
                if iter_num > 0:
                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "config": config.to_dict(),
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                    }
                    if ema_model:
                        checkpoint["ema"] = ema_model.shadow
                    ckpt_path = out_dir / "ckpt.pt"
                    tmp_path = ckpt_path.with_suffix(".tmp")
                    torch.save(checkpoint, tmp_path)
                    tmp_path.replace(ckpt_path)

        if iter_num == 0 and config.training.eval_only:
            break

        sum_sq_grad_norms = 0.0
        prev_grads: list[torch.Tensor | None] | None = None
        for micro_step in range(effective_grad_accum):
            if is_ddp and hasattr(model, "require_backward_grad_sync"):
                model.require_backward_grad_sync = micro_step == effective_grad_accum - 1
            with ctx:
                _logits, loss = model(batch_x, batch_y)
                loss = loss / effective_grad_accum

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at iter {iter_num}: {loss.item()}")

            scaler.scale(loss).backward()

            if config.training.log_gns and effective_grad_accum > 1:
                with torch.no_grad():
                    scale = scaler.get_scale() if dtype == "float16" else 1.0
                    if prev_grads is None:
                        prev_grads = []
                        for p in model.parameters():
                            if p.grad is None:
                                prev_grads.append(None)
                                continue
                            grad = p.grad.detach()
                            prev_grads.append(grad.clone())
                            sum_sq_grad_norms += (grad.float() / scale).pow(2).sum().item()
                    else:
                        for idx, p in enumerate(model.parameters()):
                            if p.grad is None:
                                prev_grads[idx] = None
                                continue
                            grad = p.grad.detach()
                            prev = prev_grads[idx]
                            delta = grad if prev is None else grad - prev
                            prev_grads[idx] = grad.clone()
                            sum_sq_grad_norms += (delta.float() / scale).pow(2).sum().item()

            if micro_step < effective_grad_accum - 1:
                try:
                    batch_x, batch_y = next(train_iter)
                except StopIteration:
                    current_epoch += 1
                    set_sampler_epoch(train_sampler, current_epoch)
                    train_iter = iter(train_loader)
                    batch_x, batch_y = next(train_iter)
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        if config.training.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)

        if config.training.log_gns and effective_grad_accum > 1:
            with torch.no_grad():
                scale = scaler.get_scale() if dtype == "float16" else 1.0
                acc_scale = 1.0 if config.training.grad_clip != 0.0 else scale
                acc_norm_sq = float(
                    sum(
                        (p.grad.float() / acc_scale).pow(2).sum()
                        for p in model.parameters()
                        if p.grad is not None
                    )
                )
                n = effective_grad_accum
                last_gns = (sum_sq_grad_norms * n) / max(acc_norm_sq, 1e-8) - 1

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if ema_model:
            ema_model.update(raw_model)

        if master_process and iter_num % config.training.log_interval == 0:
            pass

        iter_num += 1

        if iter_num > config.training.max_iters:
            break

    cleanup_distributed()


def _is_dpo_config(config_path: str | Path) -> bool:
    with Path(config_path).open() as f:
        raw = yaml.safe_load(f)
    return raw.get("dpo", {}).get("enabled", False)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)

    config_path = sys.argv[1]
    if _is_dpo_config(config_path):
        from foundry.train_dpo import train_dpo

        train_dpo(config_path)
    else:
        train(config_path)
