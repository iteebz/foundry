"""Programmatic mutation generator for architecture experiments."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_baseline() -> Dict[str, Any]:
    """Load baseline experiment config."""
    baseline_path = Path("experiments/baseline.yaml")
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline not found: {baseline_path}")
    
    with open(baseline_path) as f:
        return yaml.safe_load(f)


def mutate_attention(
    variant: str = "gqa_2kv",
    base_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Generate attention mechanism mutations.
    
    Variants:
        gqa_2kv: 2 KV heads (standard GQA)
        gqa_1kv: 1 KV head (extreme GQA, approaching MQA)
        mha: Multi-head attention (n_kv_head = n_head)
    """
    config = base_config or load_baseline()
    n_head = config["model_args"]["n_head"]
    
    variants = {
        "gqa_2kv": 2,
        "gqa_1kv": 1,
        "mha": n_head,
    }
    
    if variant not in variants:
        raise ValueError(f"Unknown attention variant: {variant}. Choose from {list(variants.keys())}")
    
    config["name"] = f"attn_{variant}"
    config["model_args"]["n_kv_head"] = variants[variant]
    
    return config


def mutate_depth(
    n_layers: int,
    base_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Generate depth mutations."""
    config = base_config or load_baseline()
    config["name"] = f"depth_{n_layers}L"
    config["model_args"]["n_layer"] = n_layers
    return config


def mutate_width(
    n_embd: int,
    base_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Generate width mutations."""
    config = base_config or load_baseline()
    config["name"] = f"width_{n_embd}d"
    config["model_args"]["n_embd"] = n_embd
    
    # Adjust n_head to maintain head_dim divisibility
    head_dim = 64
    n_head = n_embd // head_dim
    if n_head < 1:
        n_head = 1
    config["model_args"]["n_head"] = n_head
    
    # Adjust n_kv_head proportionally
    if "n_kv_head" in config["model_args"]:
        config["model_args"]["n_kv_head"] = max(1, n_head // 3)
    
    return config


def mutate_lr(
    learning_rate: float,
    base_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Generate learning rate mutations."""
    config = base_config or load_baseline()
    config["name"] = f"lr_{learning_rate:.0e}"
    config["training"]["learning_rate"] = learning_rate
    return config


def mutate_norm(
    norm_type: str = "layernorm",
    base_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Generate normalization mutations.
    
    Variants:
        rmsnorm: RMSNorm (default, no bias, more efficient)
        layernorm: Standard LayerNorm (with bias option)
    """
    config = base_config or load_baseline()
    
    variants = ["rmsnorm", "layernorm"]
    if norm_type not in variants:
        raise ValueError(f"Unknown norm type: {norm_type}. Choose from {variants}")
    
    config["name"] = f"norm_{norm_type}"
    if "model_args" not in config:
        config["model_args"] = {}
    config["model_args"]["norm_type"] = norm_type
    
    return config


def mutate_activation(
    activation: str = "gelu",
    base_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Generate activation function mutations.
    
    Variants:
        swiglu: SwiGLU (default, gated variant)
        gelu: Standard GELU
        glu: Gated Linear Unit
    """
    config = base_config or load_baseline()
    
    variants = ["swiglu", "gelu", "glu"]
    if activation not in variants:
        raise ValueError(f"Unknown activation: {activation}. Choose from {variants}")
    
    config["name"] = f"act_{activation}"
    if "model_args" not in config:
        config["model_args"] = {}
    config["model_args"]["activation"] = activation
    
    return config


def mutate_position_encoding(
    position_encoding: str = "alibi",
    base_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Generate position encoding mutations.
    
    Variants:
        rope: Rotary Position Embedding (default)
        alibi: Attention with Linear Biases
    """
    config = base_config or load_baseline()
    
    variants = ["rope", "alibi"]
    if position_encoding not in variants:
        raise ValueError(f"Unknown position encoding: {position_encoding}. Choose from {variants}")
    
    config["name"] = f"pos_{position_encoding}"
    if "model_args" not in config:
        config["model_args"] = {}
    config["model_args"]["position_encoding"] = position_encoding
    
    return config


def mutate_loss(
    loss_type: str = "focal",
    base_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Generate loss function mutations.
    
    Variants:
        cross_entropy: Standard cross entropy (default)
        focal: Focal loss (focuses on hard examples)
        label_smoothing: Cross entropy with label smoothing
    """
    config = base_config or load_baseline()
    
    variants = ["cross_entropy", "focal", "label_smoothing"]
    if loss_type not in variants:
        raise ValueError(f"Unknown loss type: {loss_type}. Choose from {variants}")
    
    config["name"] = f"loss_{loss_type}"
    if "model_args" not in config:
        config["model_args"] = {}
    config["model_args"]["loss_type"] = loss_type
    
    return config


def mutate_batch_size(
    batch_size: int,
    base_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Generate batch size mutations."""
    config = base_config or load_baseline()
    config["name"] = f"batch_{batch_size}"
    if "training" not in config:
        config["training"] = {}
    config["training"]["batch_size"] = batch_size
    return config


def mutate_warmup(
    warmup_iters: int,
    base_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Generate warmup schedule mutations."""
    config = base_config or load_baseline()
    config["name"] = f"warmup_{warmup_iters}"
    if "training" not in config:
        config["training"] = {}
    config["training"]["warmup_iters"] = warmup_iters
    return config


def mutate_grad_clip(
    grad_clip: float,
    base_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Generate gradient clipping mutations."""
    config = base_config or load_baseline()
    config["name"] = f"gradclip_{grad_clip}"
    if "training" not in config:
        config["training"] = {}
    config["training"]["grad_clip"] = grad_clip
    return config


def mutate_data_filter(
    min_len: int,
    max_len: Optional[int] = None,
    dedupe: bool = True,
    base_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Generate data filtering mutations."""
    config = base_config or load_baseline()
    config["name"] = f"filter_min{min_len}"
    if max_len:
        config["name"] += f"_max{max_len}"
    if dedupe:
        config["name"] += "_dedupe"
    
    if "data" not in config:
        config["data"] = {}
    config["data"]["min_len"] = min_len
    if max_len:
        config["data"]["max_len"] = max_len
    config["data"]["dedupe"] = dedupe
    
    return config


def mutate_weight_decay(
    weight_decay: float,
    base_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Generate weight decay mutations."""
    config = base_config or load_baseline()
    config["name"] = f"wd_{weight_decay:.0e}"
    if "training" not in config:
        config["training"] = {}
    config["training"]["weight_decay"] = weight_decay
    return config


def mutate_adam_betas(
    beta1: float,
    beta2: float,
    base_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Generate Adam beta mutations."""
    config = base_config or load_baseline()
    config["name"] = f"adam_b1_{beta1}_b2_{beta2}"
    if "training" not in config:
        config["training"] = {}
    config["training"]["beta1"] = beta1
    config["training"]["beta2"] = beta2
    return config


def mutate_lora_rank(
    r: int,
    base_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Generate LoRA rank mutations."""
    config = base_config or load_baseline()
    config["name"] = f"lora_r{r}"
    if "lora" not in config:
        config["lora"] = {}
    config["lora"]["r"] = r
    config["lora"]["enabled"] = True
    return config


def mutate_lora_alpha(
    lora_alpha: int,
    base_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Generate LoRA alpha mutations."""
    config = base_config or load_baseline()
    config["name"] = f"lora_alpha{lora_alpha}"
    if "lora" not in config:
        config["lora"] = {}
    config["lora"]["lora_alpha"] = lora_alpha
    config["lora"]["enabled"] = True
    return config


def mutate_lora_dropout(
    lora_dropout: float,
    base_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Generate LoRA dropout mutations."""
    config = base_config or load_baseline()
    config["name"] = f"lora_dropout{lora_dropout}"
    if "lora" not in config:
        config["lora"] = {}
    config["lora"]["lora_dropout"] = lora_dropout
    config["lora"]["enabled"] = True
    return config


def mutate_conversation_format(
    format_type: str = "chatml",
    base_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Generate conversation format mutations.
    
    Variants:
        chatml: ChatML format (default)
        llama3: Llama3 chat format
        alpaca: Alpaca instruction format
    """
    config = base_config or load_baseline()
    
    variants = ["chatml", "llama3", "alpaca"]
    if format_type not in variants:
        raise ValueError(f"Unknown format: {format_type}. Choose from {variants}")
    
    config["name"] = f"conv_{format_type}"
    if "data" not in config:
        config["data"] = {}
    config["data"]["conversation_format"] = format_type
    
    return config


def mutate_mla(
    latent_dim: Optional[int] = None,
    base_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Generate MLA (Multi-Latent Attention) mutation.
    
    Args:
        latent_dim: Latent compression dimension (default: n_embd // 2)
    """
    config = base_config or load_baseline()
    
    if latent_dim is None:
        n_embd = config.get("model_args", {}).get("n_embd", 384)
        latent_dim = n_embd // 2
    
    config["name"] = f"mla_latent{latent_dim}"
    if "model_args" not in config:
        config["model_args"] = {}
    config["model_args"]["attention_type"] = "mla"
    config["model_args"]["mla_latent_dim"] = latent_dim
    
    return config


def mutate_moe(
    n_experts: int = 8,
    top_k: int = 2,
    base_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Generate MoE (Mixture of Experts) mutation.
    
    Args:
        n_experts: Number of expert MLPs
        top_k: Number of experts to route each token to
    """
    config = base_config or load_baseline()
    
    config["name"] = f"moe_{n_experts}e{top_k}k"
    if "model_args" not in config:
        config["model_args"] = {}
    config["model_args"]["mlp_type"] = "moe"
    config["model_args"]["moe_n_experts"] = n_experts
    config["model_args"]["moe_top_k"] = top_k
    
    return config


def save_mutation(config: Dict[str, Any], output_dir: str = "experiments") -> Path:
    """Save mutation config to YAML."""
    output_path = Path(output_dir) / f"{config['name']}.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return output_path


def generate_sweep(
    mutation_type: str,
    variants: list,
    output_dir: str = "experiments"
) -> list[Path]:
    """Generate a sweep of mutations.
    
    Args:
        mutation_type: 'attention', 'depth', 'width', or 'lr'
        variants: List of variant parameters
        output_dir: Where to save configs
    
    Returns:
        List of generated config paths
    """
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


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.mutate <type> [variant]")
        print("Types: attention, depth, width, lr, norm, activation, position, loss,")
        print("       batch_size, warmup, grad_clip, weight_decay, adam_betas,")
        print("       lora_rank, lora_alpha, lora_dropout, conversation_format, mla, moe")
        print("\nExamples:")
        print("  python -m src.mutate attention gqa_2kv")
        print("  python -m src.mutate depth 8")
        print("  python -m src.mutate lr 3e-4")
        print("  python -m src.mutate mla 192")
        print("  python -m src.mutate moe 8 2")
        sys.exit(1)
    
    mutation_type = sys.argv[1]
    
    if mutation_type == "attention":
        config = mutate_attention(sys.argv[2])
    elif mutation_type == "depth":
        config = mutate_depth(int(sys.argv[2]))
    elif mutation_type == "width":
        config = mutate_width(int(sys.argv[2]))
    elif mutation_type == "lr":
        config = mutate_lr(float(sys.argv[2]))
    elif mutation_type == "norm":
        config = mutate_norm(sys.argv[2])
    elif mutation_type == "activation":
        config = mutate_activation(sys.argv[2])
    elif mutation_type == "position":
        config = mutate_position_encoding(sys.argv[2])
    elif mutation_type == "loss":
        config = mutate_loss(sys.argv[2])
    elif mutation_type == "batch_size":
        config = mutate_batch_size(int(sys.argv[2]))
    elif mutation_type == "warmup":
        config = mutate_warmup(int(sys.argv[2]))
    elif mutation_type == "grad_clip":
        config = mutate_grad_clip(float(sys.argv[2]))
    elif mutation_type == "weight_decay":
        config = mutate_weight_decay(float(sys.argv[2]))
    elif mutation_type == "adam_betas":
        if len(sys.argv) < 4:
            print("adam_betas requires two arguments: beta1 beta2")
            sys.exit(1)
        config = mutate_adam_betas(float(sys.argv[2]), float(sys.argv[3]))
    elif mutation_type == "lora_rank":
        config = mutate_lora_rank(int(sys.argv[2]))
    elif mutation_type == "lora_alpha":
        config = mutate_lora_alpha(int(sys.argv[2]))
    elif mutation_type == "lora_dropout":
        config = mutate_lora_dropout(float(sys.argv[2]))
    elif mutation_type == "conversation_format":
        config = mutate_conversation_format(sys.argv[2])
    elif mutation_type == "mla":
        latent_dim = int(sys.argv[2]) if len(sys.argv) > 2 else None
        config = mutate_mla(latent_dim)
    elif mutation_type == "moe":
        n_experts = int(sys.argv[2]) if len(sys.argv) > 2 else 8
        top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 2
        config = mutate_moe(n_experts, top_k)
    else:
        print(f"Unknown mutation type: {mutation_type}")
        sys.exit(1)
    
    path = save_mutation(config)
    print(f"\nGenerated: {path}")
    print(f"Run with: python src/train.py {path}")
