"""DPO training script."""

from __future__ import annotations

import copy
import math
import signal
import sys
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from types import FrameType

    from foundry.types import ModelProtocol
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from foundry.config import RunConfig
from foundry.data.dataset import PreferenceDataset, collate_preference_batch
from foundry.distributed import (
    cleanup_distributed,
    init_distributed,
    print_distributed_info,
    wrap_model_distributed,
)
from foundry.metrics import MetricLogger
from foundry.model import GPT
from foundry.modules.dpo_loss import DPOLoss, compute_log_probs


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


def compute_sequence_logprobs(
    model: ModelProtocol,
    input_ids: torch.Tensor,
    prompt_lens: torch.Tensor,
    ctx: Any,
    device: str,
) -> torch.Tensor:
    with ctx:
        logits, _ = model(input_ids)

    batch_size = input_ids.size(0)
    log_probs = torch.zeros(batch_size, device=device)

    for i in range(batch_size):
        prompt_len = prompt_lens[i].item()
        seq_logits = logits[i, prompt_len - 1 : -1, :]
        seq_labels = input_ids[i, prompt_len:]
        if seq_labels.numel() > 0:
            log_probs[i] = compute_log_probs(
                seq_logits.unsqueeze(0), seq_labels.unsqueeze(0)
            ).squeeze(0)

    return log_probs


def _resolve_device(config: RunConfig) -> str:
    device = config.training.device
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def _resolve_dtype(config: RunConfig) -> str:
    dtype = config.training.dtype
    if dtype == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return "bfloat16"
        return "float16"
    return dtype


def _load_model_checkpoint(model: ModelProtocol, path: str, device: str) -> None:
    from foundry.checkpoint import load_checkpoint

    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    load_checkpoint(model, None, str(ckpt_path), device=device)


def _setup_reference_model(config: RunConfig, model: ModelProtocol, device: str) -> ModelProtocol:
    if config.dpo.reference_model:
        ref_model = GPT(config.model)
        ref_model.to(device)
        _load_model_checkpoint(ref_model, config.dpo.reference_model, device)
    else:
        ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    return ref_model


def _get_tokenizer():
    try:
        import tiktoken

        return tiktoken.get_encoding("gpt2")
    except ImportError as e:
        raise ImportError("DPO training requires tiktoken: pip install tiktoken") from e


def _save_checkpoint(
    raw_model: Any,
    optimizer: Any,
    config: RunConfig,
    iter_num: int,
    best_val_loss: float,
    out_dir: Path,
) -> None:
    checkpoint = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config.to_dict(),
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
    }
    ckpt_path = out_dir / "ckpt.pt"
    tmp_path = ckpt_path.with_suffix(".tmp")
    torch.save(checkpoint, tmp_path)
    tmp_path.replace(ckpt_path)


def train_dpo(config_path: str | Path):  # noqa: C901
    config = RunConfig.from_yaml(Path(config_path))

    if not config.dpo.enabled:
        raise ValueError("DPO training requires dpo.enabled=true in config")
    if not config.dpo.preference_data:
        raise ValueError("DPO training requires dpo.preference_data path")

    master_process, rank, world_size = init_distributed(backend="nccl")

    effective_grad_accum = config.training.gradient_accumulation_steps
    if world_size > 1:
        assert effective_grad_accum % world_size == 0
        effective_grad_accum //= world_size

    out_dir = Path(config.training.out_dir) / config.name
    if master_process:
        out_dir.mkdir(parents=True, exist_ok=True)

    metric_logger = MetricLogger(str(out_dir)) if master_process else None

    torch.manual_seed(config.training.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = _resolve_device(config)
    dtype = _resolve_dtype(config)
    device_type = "cuda" if "cuda" in device else "mps" if "mps" in device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[
        dtype
    ]
    use_amp = device_type == "cuda" and dtype in ("float16", "bfloat16")
    ctx = (
        torch.amp.autocast(device_type="cuda", dtype=ptdtype)  # type: ignore[attr-defined]
        if use_amp
        else nullcontext()
    )

    model = GPT(config.model)
    model.to(device)

    if config.training.init_from not in ("scratch", "resume"):
        _load_model_checkpoint(model, config.training.init_from, device)

    ref_model = _setup_reference_model(config, model, device)

    if config.lora.enabled:
        from foundry.lora import apply_lora_to_model

        model = apply_lora_to_model(
            model,
            r=config.lora.r,
            lora_alpha=config.lora.lora_alpha,
            lora_dropout=config.lora.lora_dropout,
        )

    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and dtype == "float16"))  # type: ignore[attr-defined]
    optimizer = model.configure_optimizers(
        config.training.weight_decay,
        config.training.learning_rate,
        (config.training.beta1, config.training.beta2),
        device_type,
    )

    iter_num = 0
    best_val_loss = 1e9

    if config.training.compile:
        model = torch.compile(model, mode=config.training.compile_mode)
        ref_model = torch.compile(ref_model, mode=config.training.compile_mode)

    model, is_ddp, is_fsdp = wrap_model_distributed(
        model,  # type: ignore[arg-type]
        strategy=config.training.distributed,
        fsdp_min_params=config.training.fsdp_min_params,
    )

    if master_process:
        print_distributed_info(model, is_ddp, is_fsdp)

    raw_model = model.module if (is_ddp or is_fsdp) else model

    tokenizer = _get_tokenizer()
    train_dataset = PreferenceDataset(
        config.dpo.preference_data,
        tokenizer,  # type: ignore[arg-type]
        max_length=config.data.block_size,
    )

    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = RandomSampler(
            train_dataset, replacement=True, num_samples=len(train_dataset)
        )

    num_workers = 0 if device_type == "mps" else 4
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=(device_type == "cuda"),
        collate_fn=collate_preference_batch,
    )

    dpo_loss_fn = DPOLoss(beta=config.dpo.beta, label_smoothing=config.dpo.label_smoothing)

    current_epoch = 0
    train_iter = iter(train_loader)
    if hasattr(train_sampler, "set_epoch"):
        train_sampler.set_epoch(current_epoch)  # type: ignore[attr-defined]

    while iter_num <= config.training.max_iters:
        try:
            batch = next(train_iter)
        except StopIteration:
            current_epoch += 1
            if hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(current_epoch)  # type: ignore[attr-defined]
            train_iter = iter(train_loader)
            batch = next(train_iter)

        chosen_ids = batch["chosen_ids"].to(device)
        rejected_ids = batch["rejected_ids"].to(device)
        prompt_lens = batch["prompt_lens"].to(device)

        lr = get_lr(iter_num, config) if config.training.decay_lr else config.training.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        for micro_step in range(effective_grad_accum):
            if is_ddp:
                model.require_backward_grad_sync = micro_step == effective_grad_accum - 1  # type: ignore[attr-defined]

            with torch.no_grad():
                ref_chosen_logps = compute_sequence_logprobs(
                    ref_model,
                    chosen_ids,
                    prompt_lens,
                    ctx,
                    device,  # type: ignore[arg-type]
                )
                ref_rejected_logps = compute_sequence_logprobs(
                    ref_model,
                    rejected_ids,
                    prompt_lens,
                    ctx,
                    device,  # type: ignore[arg-type]
                )

            policy_chosen_logps = compute_sequence_logprobs(
                model,
                chosen_ids,
                prompt_lens,
                ctx,
                device,  # type: ignore[arg-type]
            )
            policy_rejected_logps = compute_sequence_logprobs(
                model,
                rejected_ids,
                prompt_lens,
                ctx,
                device,  # type: ignore[arg-type]
            )

            loss = dpo_loss_fn(
                policy_chosen_logps,
                policy_rejected_logps,
                ref_chosen_logps,
                ref_rejected_logps,
            )
            loss = loss / effective_grad_accum

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at iter {iter_num}: {loss.item()}")

            scaler.scale(loss).backward()

            if micro_step < effective_grad_accum - 1:
                try:
                    batch = next(train_iter)
                except StopIteration:
                    current_epoch += 1
                    if hasattr(train_sampler, "set_epoch"):
                        train_sampler.set_epoch(current_epoch)  # type: ignore[attr-defined]
                    train_iter = iter(train_loader)
                    batch = next(train_iter)
                chosen_ids = batch["chosen_ids"].to(device)
                rejected_ids = batch["rejected_ids"].to(device)
                prompt_lens = batch["prompt_lens"].to(device)

        if config.training.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if iter_num > 0 and iter_num % config.training.eval_interval == 0 and master_process:
            metrics = {
                "iter": iter_num,
                "dpo_loss": float(loss.item() * effective_grad_accum),
                "lr": lr,
                "batch_size": config.data.batch_size * effective_grad_accum * world_size,
            }

            if metric_logger:
                metric_logger.log(metrics)

            if config.wandb.enabled:
                import wandb

                wandb.log(metrics)

            current_loss = loss.item() * effective_grad_accum
            if current_loss < best_val_loss or config.training.always_save_checkpoint:
                best_val_loss = current_loss
                _save_checkpoint(raw_model, optimizer, config, iter_num, best_val_loss, out_dir)

        iter_num += 1

    cleanup_distributed()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)

    train_dpo(sys.argv[1])
