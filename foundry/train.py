"""Training script with RunConfig (v2 - clean refactor)."""

import math
import os
import pickle
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from foundry.config import RunConfig
from foundry.distributed import (
    cleanup_distributed,
    init_distributed,
    print_distributed_info,
    wrap_model_distributed,
)
from foundry.eval import evaluate
from foundry.metrics import MetricLogger
from foundry.model import GPT


def get_batch(split, data_dir, block_size, batch_size, device):
    """Load a batch of data."""
    if split == "train":
        data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    else:
        data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack(
        [torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix]
    )

    if "cuda" in device:
        x, y = (
            x.pin_memory().to(device, non_blocking=True),
            y.pin_memory().to(device, non_blocking=True),
        )
    else:
        x, y = x.to(device), y.to(device)

    return x, y


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

    if world_size > 1:
        assert config.training.gradient_accumulation_steps % world_size == 0
        config.training.gradient_accumulation_steps //= world_size

    if master_process:
        os.makedirs(config.training.out_dir, exist_ok=True)
        print(f"Training config: {config.name}")

    metric_logger = MetricLogger(config.training.out_dir) if master_process else None

    torch.manual_seed(1337 + rank)
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

    data_dir = os.path.join("data", config.data.dataset)

    meta_path = os.path.join(data_dir, "meta.pkl")
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        config.model.vocab_size = meta["vocab_size"]
        if master_process:
            print(f"vocab_size = {config.model.vocab_size}")
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
            stats = get_lora_params(model)
            print(f"LoRA: r={config.lora.r}, trainable={stats['trainable_pct']:.2f}%")

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
    optimizer = model.configure_optimizers(
        config.training.weight_decay,
        config.training.learning_rate,
        (config.training.beta1, config.training.beta2),
        device_type,
    )

    if config.training.compile:
        if master_process:
            print(f"Compiling model (mode={config.training.compile_mode})...")
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

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(config.training.eval_iters)
            for k in range(config.training.eval_iters):
                X, Y = get_batch(
                    split, data_dir, config.data.block_size, config.data.batch_size, device
                )
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    iter_num = 0
    best_val_loss = 1e9
    t0 = time.time()

    tokens_per_iter = (
        config.training.gradient_accumulation_steps
        * world_size
        * config.data.batch_size
        * config.data.block_size
    )
    if master_process:
        print(f"Tokens per iteration: {tokens_per_iter:,}")

    X, Y = get_batch("train", data_dir, config.data.block_size, config.data.batch_size, device)

    while True:
        lr = get_lr(iter_num, config) if config.training.decay_lr else config.training.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if iter_num % config.training.eval_interval == 0 and master_process:
            losses = estimate_loss()
            print(f"step {iter_num}: train={losses['train']:.4f}, val={losses['val']:.4f}")

            if metric_logger:
                metric_logger.log(
                    {
                        "iter": iter_num,
                        "train_loss": losses["train"].item(),
                        "val_loss": losses["val"].item(),
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
                    torch.save(checkpoint, os.path.join(config.training.out_dir, "ckpt.pt"))
                    if master_process:
                        print(f"Saved checkpoint (val_loss={best_val_loss:.4f})")

        if iter_num == 0 and config.training.eval_only:
            break

        for micro_step in range(config.training.gradient_accumulation_steps):
            if is_ddp:
                model.require_backward_grad_sync = (
                    micro_step == config.training.gradient_accumulation_steps - 1
                )
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / config.training.gradient_accumulation_steps
            X, Y = get_batch(
                "train", data_dir, config.data.block_size, config.data.batch_size, device
            )
            scaler.scale(loss).backward()

        if config.training.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if ema_model:
            ema_model.update(raw_model)

        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        if iter_num % config.training.log_interval == 0 and master_process:
            lossf = loss.item() * config.training.gradient_accumulation_steps
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms")

        iter_num += 1

        if iter_num > config.training.max_iters:
            break

    if master_process:
        results = evaluate(
            raw_model,
            lambda split: get_batch(
                split, data_dir, config.data.block_size, config.data.batch_size, device
            ),
            max_iters=config.training.eval_iters,
            device=device,
            ctx=ctx,
        )
        print(f"\nFinal: loss={results['loss']:.4f}, perplexity={results['perplexity']:.2f}")

    cleanup_distributed()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m foundry.train_v2 <config.yaml>")
        sys.exit(1)

    train(sys.argv[1])
