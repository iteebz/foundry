"""DPO training script."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from foundry.types import ModelProtocol
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from foundry.config import RunConfig
from foundry.data.dataset import PreferenceDataset, collate_preference_batch
from foundry.distributed import cleanup_distributed
from foundry.model import GPT
from foundry.modules.dpo_loss import DPOLoss, compute_log_probs
from foundry.train_common import (
    get_lr,
    save_checkpoint,
    setup_training,
    step_optimizer,
)


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


def _setup_reference_model(config: RunConfig, model: ModelProtocol, device: str) -> ModelProtocol:
    if config.dpo.reference_model:
        from foundry.checkpoint import load_checkpoint

        ref_model = GPT(config.model)
        ref_model.to(device)
        ckpt_path = Path(config.dpo.reference_model)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        load_checkpoint(ref_model, None, str(ckpt_path), device=device)
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


def train_dpo(config_path: str | Path):  # noqa: C901
    config = RunConfig.from_yaml(Path(config_path))

    if not config.dpo.enabled:
        raise ValueError("DPO training requires dpo.enabled=true in config")
    if not config.dpo.preference_data:
        raise ValueError("DPO training requires dpo.preference_data path")

    # Load base checkpoint before setup_training builds the model,
    # since DPO needs the ref model cloned from the loaded weights.
    # setup_training handles: distributed, device, model init, LoRA, optimizer, scaler.
    tc = setup_training(config)

    # Load pretrained weights if specified
    if config.training.init_from not in ("scratch", "resume"):
        from foundry.checkpoint import load_checkpoint

        ckpt_path = Path(config.training.init_from)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        load_checkpoint(tc.raw_model, None, str(ckpt_path), device=tc.device)

    # Reference model: clone from policy (or load separate checkpoint)
    ref_model = _setup_reference_model(config, tc.raw_model, tc.device)
    if config.training.compile:
        ref_model = torch.compile(ref_model, mode=config.training.compile_mode)

    iter_num = 0
    best_val_loss = 1e9

    tokenizer = _get_tokenizer()
    train_dataset = PreferenceDataset(
        config.dpo.preference_data,
        tokenizer,
        max_length=config.data.block_size,
    )

    if tc.world_size > 1:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = RandomSampler(
            train_dataset, replacement=True, num_samples=len(train_dataset)
        )

    num_workers = 0 if tc.device_type == "mps" else 4
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=(tc.device_type == "cuda"),
        collate_fn=collate_preference_batch,
    )

    dpo_loss_fn = DPOLoss(beta=config.dpo.beta, label_smoothing=config.dpo.label_smoothing)

    current_epoch = 0
    train_iter = iter(train_loader)
    if hasattr(train_sampler, "set_epoch"):
        train_sampler.set_epoch(current_epoch)

    while iter_num <= config.training.max_iters:
        try:
            batch = next(train_iter)
        except StopIteration:
            current_epoch += 1
            if hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(current_epoch)
            train_iter = iter(train_loader)
            batch = next(train_iter)

        chosen_ids = batch["chosen_ids"].to(tc.device)
        rejected_ids = batch["rejected_ids"].to(tc.device)
        prompt_lens = batch["prompt_lens"].to(tc.device)

        lr = get_lr(iter_num, config) if config.training.decay_lr else config.training.learning_rate
        for param_group in tc.optimizer.param_groups:
            param_group["lr"] = lr

        for micro_step in range(tc.effective_grad_accum):
            if tc.is_ddp:
                tc.model.require_backward_grad_sync = micro_step == tc.effective_grad_accum - 1

            with torch.no_grad():
                ref_chosen_logps = compute_sequence_logprobs(
                    ref_model,
                    chosen_ids,
                    prompt_lens,
                    tc.amp_ctx,
                    tc.device,
                )
                ref_rejected_logps = compute_sequence_logprobs(
                    ref_model,
                    rejected_ids,
                    prompt_lens,
                    tc.amp_ctx,
                    tc.device,
                )

            policy_chosen_logps = compute_sequence_logprobs(
                tc.model,
                chosen_ids,
                prompt_lens,
                tc.amp_ctx,
                tc.device,
            )
            policy_rejected_logps = compute_sequence_logprobs(
                tc.model,
                rejected_ids,
                prompt_lens,
                tc.amp_ctx,
                tc.device,
            )

            loss = dpo_loss_fn(
                policy_chosen_logps,
                policy_rejected_logps,
                ref_chosen_logps,
                ref_rejected_logps,
            )
            loss = loss / tc.effective_grad_accum

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at iter {iter_num}: {loss.item()}")

            tc.scaler.scale(loss).backward()

            if micro_step < tc.effective_grad_accum - 1:
                try:
                    batch = next(train_iter)
                except StopIteration:
                    current_epoch += 1
                    if hasattr(train_sampler, "set_epoch"):
                        train_sampler.set_epoch(current_epoch)
                    train_iter = iter(train_loader)
                    batch = next(train_iter)
                chosen_ids = batch["chosen_ids"].to(tc.device)
                rejected_ids = batch["rejected_ids"].to(tc.device)
                prompt_lens = batch["prompt_lens"].to(tc.device)

        step_optimizer(tc.model, tc.optimizer, tc.scaler, config.training.grad_clip)

        if iter_num > 0 and iter_num % config.training.eval_interval == 0 and tc.master_process:
            metrics = {
                "iter": iter_num,
                "dpo_loss": float(loss.item() * tc.effective_grad_accum),
                "lr": lr,
                "batch_size": config.data.batch_size * tc.effective_grad_accum * tc.world_size,
            }

            if tc.metric_logger:
                tc.metric_logger.log(metrics)

            if config.wandb.enabled:
                import wandb

                wandb.log(metrics)

            current_loss = loss.item() * tc.effective_grad_accum
            if current_loss < best_val_loss or config.training.always_save_checkpoint:
                best_val_loss = current_loss
                save_checkpoint(
                    tc.raw_model,
                    tc.optimizer,
                    config,
                    iter_num,
                    best_val_loss,
                    tc.out_dir,
                )

        iter_num += 1

    cleanup_distributed()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        sys.exit(1)

    train_dpo(sys.argv[1])
