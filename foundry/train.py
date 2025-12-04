"""Training script with single GPU and DDP support."""

import math
import os
import pickle
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel

from foundry.eval import evaluate
from foundry.model import GPT, GPTConfig

out_dir = "out"
eval_interval = 500
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = "scratch"
wandb_log = False
wandb_project = "foundry"
wandb_run_name = "run"
dataset = "shakespeare_char"
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256
experiment = ""
n_layer = 6
n_head = 6
n_kv_head = 2
n_embd = 384
dropout = 0.0
bias = False
learning_rate = 6e-4
max_iters = 5000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
use_ema = True
ema_decay = 0.9999
decay_lr = True
warmup_iters = 100
lr_decay_iters = 5000
min_lr = 6e-5
backend = "nccl"
device = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
compile = True
compile_mode = "default"
lora_enabled = False
lora_r = 8
lora_alpha = 16
lora_dropout = 0.0

if len(sys.argv) > 1:
    experiment = sys.argv[1]

if experiment:
    from foundry.model_factory import get_training_overrides

    overrides = get_training_overrides(experiment)
    for key, val in overrides.items():
        if key in globals():
            globals()[key] = val

    if "lora" in overrides and isinstance(overrides["lora"], dict):
        lora_config = overrides["lora"]
        if lora_config.get("enabled", False):
            globals()["lora_enabled"] = True
            globals()["lora_r"] = lora_config.get("r", 8)
            globals()["lora_alpha"] = lora_config.get("lora_alpha", 16)
            globals()["lora_dropout"] = lora_config.get("lora_dropout", 0.0)

    print(f"Loaded config from {experiment}")

config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, int | float | bool | str)
]
config = {k: globals()[k] for k in config_keys}

ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = "cuda" if "cuda" in device else "mps" if "mps" in device else "cpu"
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

data_dir = os.path.join("data", dataset)


def get_batch(split):
    if split == "train":
        data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    else:
        data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
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


iter_num = 0
best_val_loss = 1e9


class EMA:
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


ema_model = None

meta_path = os.path.join(data_dir, "meta.pkl")
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    meta_vocab_size = meta["vocab_size"]
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

model_args = {
    "n_layer": n_layer,
    "n_head": n_head,
    "n_kv_head": n_kv_head,
    "n_embd": n_embd,
    "block_size": block_size,
    "bias": bias,
    "vocab_size": None,
    "dropout": dropout,
}
if init_from == "scratch":
    print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size of 50304")
    model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == "resume":
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, _v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
model.to(device)

lora_enabled = globals().get("lora_enabled", False)
lora_r = globals().get("lora_r", 8)
lora_alpha = globals().get("lora_alpha", 16)
lora_dropout = globals().get("lora_dropout", 0.0)

if lora_enabled:
    from foundry.lora import apply_lora_to_model, get_lora_params

    model = apply_lora_to_model(model, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    stats = get_lora_params(model)
    print(f"LoRA enabled: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    print(f"Trainable params: {stats['trainable_params']:,} ({stats['trainable_pct']:.2f}%)")

if use_ema:
    ema_model = EMA(model, decay=ema_decay)
    print(f"initialized EMA with decay={ema_decay}")

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None

compile_mode = globals().get("compile_mode", "default")
if compile:
    print(f"compiling the model with mode={compile_mode}... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model, mode=compile_mode)

if ddp:
    model = DistributedDataParallel(model, device_ids=[ddp_local_rank])


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


if wandb_log and master_process:
    import wandb

    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

X, Y = get_batch("train")
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log(
                {
                    "iter": iter_num,
                    "train/loss": losses["train"],
                    "val/loss": losses["val"],
                    "lr": lr,
                }
            )
        if losses["val"] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses["val"]
            if iter_num > 0:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                if use_ema and ema_model is not None:
                    checkpoint["ema"] = ema_model.shadow
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
    if iter_num == 0 and eval_only:
        break

    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = micro_step == gradient_accumulation_steps - 1
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch("train")
        scaler.scale(loss).backward()
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    if use_ema and ema_model is not None:
        ema_model.update(raw_model)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms")
    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

if master_process:
    results = evaluate(raw_model, get_batch, max_iters=eval_iters, device=device, ctx=ctx)
    print(f"\nFinal evaluation: loss={results['loss']:.4f}, perplexity={results['perplexity']:.2f}")
    eval_path = os.path.join(out_dir, "final_eval.txt")
    with open(eval_path, "w") as f:
        f.write(f"loss: {results['loss']}\n")
        f.write(f"perplexity: {results['perplexity']}\n")

if ddp:
    destroy_process_group()
