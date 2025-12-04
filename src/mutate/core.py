"""Core mutation utilities."""

from pathlib import Path
from typing import Any

import yaml


def load_baseline() -> dict[str, Any]:
    """Load baseline experiment config."""
    baseline_path = Path("experiments/baseline.yaml")
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline not found: {baseline_path}")

    with open(baseline_path) as f:
        return yaml.safe_load(f)


def save_mutation(config: dict[str, Any], output_dir: str = "experiments") -> Path:
    """Save mutation config to YAML."""
    output_path = Path(output_dir) / f"{config['name']}.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return output_path


def generate_sweep(
    mutation_type: str, variants: list, output_dir: str = "experiments"
) -> list[Path]:
    """Generate a sweep of mutations."""
    from .architecture import (
        mutate_activation,
        mutate_attention,
        mutate_depth,
        mutate_loss,
        mutate_mla,
        mutate_moe,
        mutate_norm,
        mutate_position_encoding,
        mutate_sliding_window,
        mutate_sparse_attention,
        mutate_width,
    )
    from .data import mutate_conversation_format, mutate_data_filter
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

    mutation_funcs = {
        "attention": mutate_attention,
        "depth": mutate_depth,
        "width": mutate_width,
        "lr": mutate_lr,
        "norm": mutate_norm,
        "activation": mutate_activation,
        "position": mutate_position_encoding,
        "loss": mutate_loss,
        "batch_size": mutate_batch_size,
        "warmup": mutate_warmup,
        "grad_clip": mutate_grad_clip,
        "data_filter": mutate_data_filter,
        "weight_decay": mutate_weight_decay,
        "adam_betas": mutate_adam_betas,
        "lora_rank": mutate_lora_rank,
        "lora_alpha": mutate_lora_alpha,
        "lora_dropout": mutate_lora_dropout,
        "conversation_format": mutate_conversation_format,
        "mla": mutate_mla,
        "moe": mutate_moe,
        "sliding_window": mutate_sliding_window,
        "sparse_attention": mutate_sparse_attention,
    }

    if mutation_type not in mutation_funcs:
        raise ValueError(f"Unknown mutation type: {mutation_type}")

    func = mutation_funcs[mutation_type]
    paths = []

    for variant in variants:
        config = func(variant)
        path = save_mutation(config, output_dir)
        paths.append(path)
        print(f"Generated: {path}")

    return paths
