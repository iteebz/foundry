"""Training script with RunConfig (v2 - clean refactor)."""

import math
import pickle
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from foundry.config import RunConfig
from foundry.data.dataset import MixtureDataset, TokenDataset
from foundry.distributed import (
    cleanup_distributed,
    init_distributed,
    print_distributed_info,
    wrap_model_distributed,
)
from foundry.eval import evaluate
from foundry.metrics import MetricLogger
from foundry.model import GPT


def get_lr(it, config):
    """Learning rate schedule with warmup and cosine decay."""
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
    """Exponential moving average of model parameters."""

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)

    def apply_shadow(self, model):
        for k, v in model.named_parameters():
            if k in self.shadow:
                v.data.copy_(self.shadow[k])


def train(config_path: str | Path):
    """Main training loop."""
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
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )

    data_dir = Path("data") / config.data.dataset

    meta_path = data_dir / "meta.pkl"
    if meta_path.exists():
        with meta_path.open("rb") as f:
            meta = pickle.load(f)  # noqa: S301 - trusted internal checkpoint
        config.model.vocab_size = meta["vocab_size"]
        if master_process:
            pass
    elif config.model.vocab_size is None:
        config.model.vocab_size = 50304

    model = GPT(config.model)
    model.to(device)

    if config.lora.enabled:
        from foundry.lora import apply_lora_to_model, get_lora_params

        model = apply_lora_to_model(
            model,
            r=config.lora.r,
            lora_alpha=config.lora.lora_alpha,
            lora_dropout=config.lora.lora_dropout,
        )
        if master_process:
            get_lora_params(model)

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
    optimizer = model.configure_optimizers(
        config.training.weight_decay,
        config.training.learning_rate,
        (config.training.beta1, config.training.beta2),
        device_type,
    )

    if config.training.compile:
        if master_process:
            pass
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

    if config.data.sources:

        def get_train_path(src_path: str) -> str:
            """Infer train split path. Handles: foo.bin -> foo_train.bin, foo_val.bin -> foo_train.bin, foo_train.bin -> foo_train.bin"""
            p = Path(src_path)
            if p.stem.endswith("_train"):
                return src_path
            if p.stem.endswith("_val"):
                return str(p.parent / f"{p.stem.replace('_val', '_train')}{p.suffix}")
            return str(p.parent / f"{p.stem}_train{p.suffix}")

        def get_val_path(src_path: str) -> str:
            """Infer val split path. Handles: foo.bin -> foo_val.bin, foo_train.bin -> foo_val.bin, foo_val.bin -> foo_val.bin"""
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

        train_dataset = MixtureDataset(train_datasets, weights, seed=seed)
        val_dataset = MixtureDataset(val_datasets, weights, seed=seed)

        if master_process:
            for _i, _src in enumerate(config.data.sources):
                pass
    else:
        train_dataset = TokenDataset(data_dir / "train.bin", block_size=config.data.block_size)
        val_dataset = TokenDataset(data_dir / "val.bin", block_size=config.data.block_size)

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split, loader in [("train", train_loader), ("val", val_loader)]:
            losses = []
            for k, (X, Y) in enumerate(loader):
                if k >= config.training.eval_iters:
                    break
                X, Y = X.to(device), Y.to(device)
                with ctx:
                    _logits, loss = model(X, Y)
                losses.append(loss.item())
            out[split] = np.mean(losses)
        model.train()
        return out

    iter_num = 0
    best_val_loss = 1e9
    current_epoch = 0
    len(train_loader) * config.data.batch_size
    t0 = time.time()

    if master_process:
        pass

    train_iter = iter(train_loader)
    if train_sampler is not None:
        train_sampler.set_epoch(current_epoch)

    while True:
        try:
            X, Y = next(train_iter)
        except StopIteration:
            current_epoch += 1
            if train_sampler is not None:
                train_sampler.set_epoch(current_epoch)
            train_iter = iter(train_loader)
            X, Y = next(train_iter)

        X, Y = X.to(device), Y.to(device)
        lr = get_lr(iter_num, config) if config.training.decay_lr else config.training.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if iter_num % config.training.eval_interval == 0 and master_process:
            losses = estimate_loss()

            if metric_logger:
                metric_logger.log(
                    {
                        "iter": iter_num,
                        "train_loss": float(losses["train"]),
                        "val_loss": float(losses["val"]),
                        "lr": lr,
                    }
                )

            if config.wandb.enabled:
                import wandb

                wandb.log(
                    {
                        "iter": iter_num,
                        "train/loss": losses["train"],
                        "val/loss": losses["val"],
                        "lr": lr,
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
                    if master_process:
                        pass

        if iter_num == 0 and config.training.eval_only:
            break

        for micro_step in range(effective_grad_accum):
            if is_ddp:
                model.require_backward_grad_sync = micro_step == effective_grad_accum - 1
            with ctx:
                _logits, loss = model(X, Y)
                loss = loss / effective_grad_accum

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at iter {iter_num}: {loss.item()}")

            scaler.scale(loss).backward()

            if micro_step < effective_grad_accum - 1:
                try:
                    X, Y = next(train_iter)
                except StopIteration:
                    current_epoch += 1
                    if train_sampler is not None:
                        train_sampler.set_epoch(current_epoch)
                    train_iter = iter(train_loader)
                    X, Y = next(train_iter)
                X, Y = X.to(device), Y.to(device)

        if config.training.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if ema_model:
            ema_model.update(raw_model)

        t1 = time.time()
        t1 - t0
        t0 = t1

        if iter_num % config.training.log_interval == 0 and master_process:
            loss.item() * config.training.gradient_accumulation_steps

        iter_num += 1

        if iter_num > config.training.max_iters:
            break

    if master_process:

        def batch_fn(split):
            loader = train_loader if split == "train" else val_loader
            for X, Y in loader:
                yield X.to(device), Y.to(device)

        evaluate(
            raw_model,
            batch_fn,
            max_iters=config.training.eval_iters,
            device=device,
            ctx=ctx,
        )

    cleanup_distributed()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)

    train(sys.argv[1])
