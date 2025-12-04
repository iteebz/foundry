"""Training hyperparameter mutation functions."""

from typing import Any

from .core import load_baseline


def mutate_lr(learning_rate: float, base_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Generate learning rate mutations."""
    config = base_config or load_baseline()
    config["name"] = f"lr_{learning_rate:.0e}"
    config["training"]["learning_rate"] = learning_rate
    return config


def mutate_batch_size(batch_size: int, base_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Generate batch size mutations."""
    config = base_config or load_baseline()
    config["name"] = f"batch_{batch_size}"
    if "training" not in config:
        config["training"] = {}
    config["training"]["batch_size"] = batch_size
    return config


def mutate_warmup(warmup_iters: int, base_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Generate warmup schedule mutations."""
    config = base_config or load_baseline()
    config["name"] = f"warmup_{warmup_iters}"
    if "training" not in config:
        config["training"] = {}
    config["training"]["warmup_iters"] = warmup_iters
    return config


def mutate_grad_clip(grad_clip: float, base_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Generate gradient clipping mutations."""
    config = base_config or load_baseline()
    config["name"] = f"gradclip_{grad_clip}"
    if "training" not in config:
        config["training"] = {}
    config["training"]["grad_clip"] = grad_clip
    return config


def mutate_weight_decay(
    weight_decay: float, base_config: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Generate weight decay mutations."""
    config = base_config or load_baseline()
    config["name"] = f"wd_{weight_decay:.0e}"
    if "training" not in config:
        config["training"] = {}
    config["training"]["weight_decay"] = weight_decay
    return config


def mutate_adam_betas(
    beta1: float, beta2: float, base_config: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Generate Adam beta mutations."""
    config = base_config or load_baseline()
    config["name"] = f"adam_b1_{beta1}_b2_{beta2}"
    if "training" not in config:
        config["training"] = {}
    config["training"]["beta1"] = beta1
    config["training"]["beta2"] = beta2
    return config


def mutate_lora_rank(r: int, base_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Generate LoRA rank mutations."""
    config = base_config or load_baseline()
    config["name"] = f"lora_r{r}"
    if "lora" not in config:
        config["lora"] = {}
    config["lora"]["r"] = r
    config["lora"]["enabled"] = True
    return config


def mutate_lora_alpha(lora_alpha: int, base_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Generate LoRA alpha mutations."""
    config = base_config or load_baseline()
    config["name"] = f"lora_alpha{lora_alpha}"
    if "lora" not in config:
        config["lora"] = {}
    config["lora"]["lora_alpha"] = lora_alpha
    config["lora"]["enabled"] = True
    return config


def mutate_lora_dropout(
    lora_dropout: float, base_config: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Generate LoRA dropout mutations."""
    config = base_config or load_baseline()
    config["name"] = f"lora_dropout{lora_dropout}"
    if "lora" not in config:
        config["lora"] = {}
    config["lora"]["lora_dropout"] = lora_dropout
    config["lora"]["enabled"] = True
    return config
