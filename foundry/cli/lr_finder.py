#!/usr/bin/env python3
"""LR Finder: Find optimal learning rate using Leslie Smith's method.

Usage:
    python lr_finder.py --model=v2 --dataset=shakespeare_char
"""

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))


def lr_find(model, get_batch, device, ctx, min_lr=1e-7, max_lr=10, num_steps=100, beta=0.98):
    """Find optimal learning rate using exponential range test."""
    model.train()
    optimizer = model.configure_optimizers(
        weight_decay=1e-1,
        learning_rate=min_lr,
        betas=(0.9, 0.95),
        device_type="cuda" if "cuda" in str(device) else "cpu",
    )

    lr_mult = (max_lr / min_lr) ** (1 / num_steps)
    lrs = []
    losses = []
    avg_loss = 0
    best_loss = float("inf")

    for step in range(num_steps):
        lr = min_lr * (lr_mult**step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        X, Y = get_batch("train")
        with ctx:
            logits, loss = model(X, Y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta ** (step + 1))

        lrs.append(lr)
        losses.append(smoothed_loss)

        if smoothed_loss < best_loss:
            best_loss = smoothed_loss

        if smoothed_loss > 4 * best_loss or math.isnan(smoothed_loss):
            print(f"Stopping early at step {step} (loss diverged)")
            break

    return lrs, losses


def plot_lr_find(lrs, losses, out_path="out/lr_find.png"):
    """Plot LR finder results."""
    Path(out_path).parent.mkdir(exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(lrs, losses)
    plt.xscale("log")
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.title("LR Finder")
    plt.grid(True, alpha=0.3)

    min_loss_idx = np.argmin(losses)
    suggested_lr = lrs[max(0, min_loss_idx - 5)]
    plt.axvline(
        suggested_lr,
        color="r",
        linestyle="--",
        label=f"Suggested LR: {suggested_lr:.2e}",
    )
    plt.legend()

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    print(f"Suggested learning rate: {suggested_lr:.2e}")

    return suggested_lr


def main():
    import pickle
    from contextlib import nullcontext

    dataset = "shakespeare_char"
    model_type = "v2"
    batch_size = 12
    block_size = 1024

    for arg in sys.argv[1:]:
        if arg.startswith("--dataset="):
            dataset = arg.split("=")[1]
        elif arg.startswith("--model="):
            model_type = arg.split("=")[1]

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    device_type = "cuda" if "cuda" in device else "cpu"
    dtype = (
        "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
    )
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )

    data_dir = Path("data") / dataset
    meta_path = data_dir / "meta.pkl"

    if meta_path.exists():
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        vocab_size = meta["vocab_size"]
    else:
        vocab_size = 50304

    print(f"Initializing {model_type} model with vocab_size={vocab_size}")

    if model_type == "v1":
        from model import GPT, GPTConfig

        config = GPTConfig(
            n_layer=6,
            n_head=6,
            n_embd=384,
            block_size=block_size,
            vocab_size=vocab_size,
            bias=False,
            dropout=0.0,
        )
    elif model_type == "v2":
        from model_v2 import GPT, GPTConfig

        config = GPTConfig(
            n_layer=6,
            n_head=6,
            n_kv_head=2,
            n_embd=384,
            block_size=block_size,
            vocab_size=vocab_size,
            bias=False,
            dropout=0.0,
        )
    else:
        raise ValueError(f"Unknown model: {model_type}")

    model = GPT(config).to(device)

    def get_batch(split):
        data = np.memmap(str(data_dir / f"{split}.bin"), dtype=np.uint16, mode="r")
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
        y = torch.stack(
            [torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix]
        )
        if device_type == "cuda":
            x, y = (
                x.pin_memory().to(device, non_blocking=True),
                y.pin_memory().to(device, non_blocking=True),
            )
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    print("Running LR finder...")
    lrs, losses = lr_find(model, get_batch, device, ctx)
    suggested_lr = plot_lr_find(lrs, losses)

    print("\nRecommended config:")
    print(f"learning_rate = {suggested_lr:.2e}")


if __name__ == "__main__":
    main()
