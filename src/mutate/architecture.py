"""Architecture mutation functions."""

from typing import Any

from .core import load_baseline


def mutate_attention(
    variant: str = "gqa_2kv", base_config: dict[str, Any] | None = None
) -> dict[str, Any]:
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
        raise ValueError(
            f"Unknown attention variant: {variant}. Choose from {list(variants.keys())}"
        )

    config["name"] = f"attn_{variant}"
    config["model_args"]["n_kv_head"] = variants[variant]

    return config


def mutate_depth(n_layers: int, base_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Generate depth mutations."""
    config = base_config or load_baseline()
    config["name"] = f"depth_{n_layers}L"
    config["model_args"]["n_layer"] = n_layers
    return config


def mutate_width(n_embd: int, base_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Generate width mutations."""
    config = base_config or load_baseline()
    config["name"] = f"width_{n_embd}d"
    config["model_args"]["n_embd"] = n_embd

    head_dim = 64
    n_head = n_embd // head_dim
    if n_head < 1:
        n_head = 1
    config["model_args"]["n_head"] = n_head

    if "n_kv_head" in config["model_args"]:
        config["model_args"]["n_kv_head"] = max(1, n_head // 3)

    return config


def mutate_norm(
    norm_type: str = "layernorm", base_config: dict[str, Any] | None = None
) -> dict[str, Any]:
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
    activation: str = "gelu", base_config: dict[str, Any] | None = None
) -> dict[str, Any]:
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
    position_encoding: str = "alibi", base_config: dict[str, Any] | None = None
) -> dict[str, Any]:
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
    loss_type: str = "focal", base_config: dict[str, Any] | None = None
) -> dict[str, Any]:
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


def mutate_mla(
    latent_dim: int | None = None, base_config: dict[str, Any] | None = None
) -> dict[str, Any]:
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
    n_experts: int = 8, top_k: int = 2, base_config: dict[str, Any] | None = None
) -> dict[str, Any]:
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


def mutate_sliding_window(
    window_size: int = 256, base_config: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Generate sliding window attention mutation.

    Args:
        window_size: Size of attention window (attend to last N tokens)
    """
    config = base_config or load_baseline()

    config["name"] = f"sw{window_size}"
    if "model_args" not in config:
        config["model_args"] = {}
    config["model_args"]["sliding_window_size"] = window_size

    return config


def mutate_sparse_attention(
    block_size: int = 64,
    stride: int | None = None,
    base_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate sparse attention mutation.

    Args:
        block_size: Local attention block size
        stride: Stride for global attention (default: same as block_size)
    """
    config = base_config or load_baseline()

    if stride is None:
        stride = block_size

    config["name"] = f"sparse_b{block_size}_s{stride}"
    if "model_args" not in config:
        config["model_args"] = {}
    config["model_args"]["sparse_block_size"] = block_size
    config["model_args"]["sparse_stride"] = stride

    return config
