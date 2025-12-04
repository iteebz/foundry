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
    
    if len(sys.argv) < 3:
        print("Usage: python -m src.mutate <type> <variant>")
        print("Types: attention, depth, width, lr, norm, activation, position, loss")
        print("\nExamples:")
        print("  python -m src.mutate attention gqa_2kv")
        print("  python -m src.mutate depth 8")
        print("  python -m src.mutate width 512")
        print("  python -m src.mutate lr 3e-4")
        print("  python -m src.mutate norm layernorm")
        print("  python -m src.mutate activation gelu")
        print("  python -m src.mutate position alibi")
        print("  python -m src.mutate loss focal")
        sys.exit(1)
    
    mutation_type = sys.argv[1]
    
    if mutation_type == "attention":
        variant = sys.argv[2]
        config = mutate_attention(variant)
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
    else:
        print(f"Unknown mutation type: {mutation_type}")
        sys.exit(1)
    
    path = save_mutation(config)
    print(f"\nGenerated: {path}")
    print(f"Run with: python src/train.py {path}")
