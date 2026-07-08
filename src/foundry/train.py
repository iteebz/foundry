"""SFT training script."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any

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
from foundry.distributed import cleanup_distributed
from foundry.train_common import (
    get_lr,
    save_checkpoint,
    setup_training,
    step_optimizer,
)


def check_m4_safety() -> None:
    import platform
    import subprocess

    if platform.system() == "Darwin":
        try:
            # Enforce human constraint d/8b11c2c4: No training on M4
            brand_bytes = subprocess.check_output(
                ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
                stderr=subprocess.DEVNULL,
            )
            brand = brand_bytes.decode().strip()
            if "M4" in brand:
                sys.stderr.write(f"FATAL: Training blocked on {brand} (d/8b11c2c4)\n")
                sys.exit(1)
        except (subprocess.SubprocessError, OSError):
            pass


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

    # Vocab size override from dataset meta
    meta_path = Path("data") / config.data.dataset / "meta.pkl"
    if meta_path.exists():
        with meta_path.open("rb") as f:
            meta = pickle.load(f)  # noqa: S301
        config.model.vocab_size = meta["vocab_size"]

    tc = setup_training(config)

    iter_num = 0
    best_val_loss = 1e9

    if config.training.init_from == "resume":
        from foundry.checkpoint import load_checkpoint

        ckpt_path = tc.out_dir / "ckpt.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Cannot resume: {ckpt_path} not found")
        resume_state = load_checkpoint(tc.raw_model, tc.optimizer, str(ckpt_path), device=tc.device)
        iter_num = resume_state.iter_num
        best_val_loss = resume_state.best_val_loss
    elif config.training.init_from != "scratch":
        from foundry.checkpoint import load_checkpoint

        ckpt_path = Path(config.training.init_from)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        load_checkpoint(tc.raw_model, None, str(ckpt_path), device=tc.device)

    ema_model = EMA(tc.model, decay=config.training.ema_decay) if config.training.use_ema else None

    seed = config.training.seed
    assert seed is not None
    train_dataset, val_dataset = build_datasets(config, seed)
    use_curriculum = config.data.curriculum.enabled and tc.world_size == 1
    train_loader, val_loader, train_sampler = build_dataloaders(
        config,
        train_dataset,
        val_dataset,
        seed,
        tc.world_size,
        tc.device_type,
    )

    @torch.no_grad()
    def estimate_loss() -> dict[str, float]:
        out: dict[str, float] = {}
        tc.model.eval()
        for split, loader in [("train", train_loader), ("val", val_loader)]:
            losses: list[float] = []
            for k, (x, y) in enumerate(loader):
                if k >= config.training.eval_iters:
                    break
                x, y = x.to(tc.device), y.to(tc.device)
                with tc.amp_ctx:
                    _logits, loss = tc.model(x, y)
                losses.append(loss.item())
            out[split] = float(np.mean(losses))
        tc.model.train()
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

        batch_x, batch_y = batch_x.to(tc.device), batch_y.to(tc.device)
        lr = get_lr(iter_num, config) if config.training.decay_lr else config.training.learning_rate
        for param_group in tc.optimizer.param_groups:
            param_group["lr"] = lr

        if iter_num > 0 and iter_num % config.training.eval_interval == 0 and tc.master_process:
            losses = estimate_loss()

            metrics: dict[str, Any] = {
                "iter": iter_num,
                "train_loss": float(losses["train"]),
                "val_loss": float(losses["val"]),
                "lr": lr,
                "batch_size": config.data.batch_size * tc.effective_grad_accum * tc.world_size,
            }
            if last_gns is not None:
                metrics["gns"] = last_gns
            if tc.metric_logger:
                tc.metric_logger.log(metrics)

            if config.wandb.enabled:
                import wandb

                wandb.log(
                    {
                        "iter": iter_num,
                        "train/loss": losses["train"],
                        "val/loss": losses["val"],
                        "lr": lr,
                        "batch_size": config.data.batch_size
                        * tc.effective_grad_accum
                        * tc.world_size,
                        **(({"train/gns": last_gns}) if last_gns is not None else {}),
                    }
                )

            if losses["val"] < best_val_loss or config.training.always_save_checkpoint:
                best_val_loss = losses["val"]
                if iter_num > 0:
                    extra = {"ema": ema_model.shadow} if ema_model else None
                    save_checkpoint(
                        tc.raw_model,
                        tc.optimizer,
                        config,
                        iter_num,
                        best_val_loss,
                        tc.out_dir,
                        extra=extra,
                    )

        if iter_num == 0 and config.training.eval_only:
            break

        sum_sq_grad_norms = 0.0
        prev_grads: list[torch.Tensor | None] | None = None
        for micro_step in range(tc.effective_grad_accum):
            if tc.is_ddp and hasattr(tc.model, "require_backward_grad_sync"):
                tc.model.require_backward_grad_sync = micro_step == tc.effective_grad_accum - 1
            with tc.amp_ctx:
                _logits, loss = tc.model(batch_x, batch_y)
                loss = loss / tc.effective_grad_accum

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at iter {iter_num}: {loss.item()}")

            tc.scaler.scale(loss).backward()

            if config.training.log_gns and tc.effective_grad_accum > 1:
                with torch.no_grad():
                    scale = tc.scaler.get_scale() if tc.dtype == "float16" else 1.0
                    if prev_grads is None:
                        prev_grads = []
                        for p in tc.model.parameters():
                            if p.grad is None:
                                prev_grads.append(None)
                                continue
                            grad = p.grad.detach()
                            prev_grads.append(grad.clone())
                            sum_sq_grad_norms += (grad.float() / scale).pow(2).sum().item()
                    else:
                        for idx, p in enumerate(tc.model.parameters()):
                            if p.grad is None:
                                prev_grads[idx] = None
                                continue
                            grad = p.grad.detach()
                            prev = prev_grads[idx]
                            delta = grad if prev is None else grad - prev
                            prev_grads[idx] = grad.clone()
                            sum_sq_grad_norms += (delta.float() / scale).pow(2).sum().item()

            if micro_step < tc.effective_grad_accum - 1:
                try:
                    batch_x, batch_y = next(train_iter)
                except StopIteration:
                    current_epoch += 1
                    set_sampler_epoch(train_sampler, current_epoch)
                    train_iter = iter(train_loader)
                    batch_x, batch_y = next(train_iter)
                batch_x, batch_y = batch_x.to(tc.device), batch_y.to(tc.device)

        if config.training.log_gns and tc.effective_grad_accum > 1:
            with torch.no_grad():
                scale = tc.scaler.get_scale() if tc.dtype == "float16" else 1.0
                acc_scale = 1.0 if config.training.grad_clip != 0.0 else scale
                acc_norm_sq = float(
                    sum(
                        (p.grad.float() / acc_scale).pow(2).sum()
                        for p in tc.model.parameters()
                        if p.grad is not None
                    )
                )
                n = tc.effective_grad_accum
                last_gns = (sum_sq_grad_norms * n) / max(acc_norm_sq, 1e-8) - 1

        step_optimizer(tc.model, tc.optimizer, tc.scaler, config.training.grad_clip)

        if ema_model:
            ema_model.update(tc.raw_model)

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
