"""Mutation framework for architecture experiments."""

from .architecture import (
    mutate_activation,
    mutate_attention,
    mutate_depth,
    mutate_dpo,
    mutate_loss,
    mutate_mla,
    mutate_moe,
    mutate_norm,
    mutate_position_encoding,
    mutate_sliding_window,
    mutate_sparse_attention,
    mutate_width,
)
from .core import (
    generate_sweep,
    load_baseline,
    save_mutation,
)
from .data import (
    mutate_conversation_format,
    mutate_curriculum,
    mutate_data_filter,
)
from .training import (
    mutate_adam_betas,
    mutate_batch_size,
    mutate_grad_clip,
    mutate_lora_alpha,
    mutate_lora_dropout,
    mutate_lora_rank,
    mutate_lr,
    mutate_warmup,
    mutate_weight_decay,
)

__all__ = [
    "load_baseline",
    "save_mutation",
    "generate_sweep",
    "mutate_attention",
    "mutate_depth",
    "mutate_width",
    "mutate_norm",
    "mutate_activation",
    "mutate_position_encoding",
    "mutate_loss",
    "mutate_dpo",
    "mutate_mla",
    "mutate_moe",
    "mutate_sliding_window",
    "mutate_sparse_attention",
    "mutate_lr",
    "mutate_batch_size",
    "mutate_warmup",
    "mutate_grad_clip",
    "mutate_weight_decay",
    "mutate_adam_betas",
    "mutate_lora_rank",
    "mutate_lora_alpha",
    "mutate_lora_dropout",
    "mutate_data_filter",
    "mutate_conversation_format",
    "mutate_curriculum",
]
