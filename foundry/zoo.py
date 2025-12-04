"""Model zoo - load pretrained checkpoints from HuggingFace."""

from pathlib import Path
from typing import Any

import torch

from foundry.model import GPT, GPTConfig

MODEL_CONFIGS = {
    "llama3-8b": {
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 32,
        "intermediate_size": 14336,
        "vocab_size": 128256,
        "max_position_embeddings": 8192,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
    },
    "llama3-1b": {
        "hidden_size": 2048,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 16,
        "intermediate_size": 8192,
        "vocab_size": 128256,
        "max_position_embeddings": 8192,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
    },
    "mistral-7b": {
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 32,
        "intermediate_size": 14336,
        "vocab_size": 32000,
        "max_position_embeddings": 32768,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
    },
    "qwen2-7b": {
        "hidden_size": 3584,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "num_hidden_layers": 28,
        "intermediate_size": 18944,
        "vocab_size": 152064,
        "max_position_embeddings": 32768,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
    },
}


def hf_to_foundry_config(hf_config: dict[str, Any]) -> GPTConfig:
    """Convert HuggingFace config to Foundry GPTConfig."""
    return GPTConfig(
        block_size=hf_config.get("max_position_embeddings", 2048),
        vocab_size=hf_config["vocab_size"],
        n_layer=hf_config["num_hidden_layers"],
        n_head=hf_config["num_attention_heads"],
        n_kv_head=hf_config.get("num_key_value_heads", hf_config["num_attention_heads"]),
        n_embd=hf_config["hidden_size"],
        dropout=0.0,
        bias=False,
        norm_type="rmsnorm",
        activation="swiglu",
        position_encoding="rope",
        loss_type="cross_entropy",
    )


def load_pretrained(
    model_name: str, device: str = "cpu", checkpoint_path: Path | None = None
) -> GPT:
    """Load pretrained model from HuggingFace checkpoint.

    Args:
        model_name: Model identifier (e.g., 'llama3-8b', 'mistral-7b')
        device: Device to load model on
        checkpoint_path: Optional local checkpoint path (if None, uses config only)

    Returns:
        GPT model initialized from pretrained weights
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")

    hf_config = MODEL_CONFIGS[model_name]
    config = hf_to_foundry_config(hf_config)

    model = GPT(config)

    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, map_location=device)

        if "model" in state_dict:
            state_dict = state_dict["model"]

        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint from {checkpoint_path}")

    model.to(device)
    return model


def export_checkpoint(
    model: GPT, output_path: Path, metadata: dict[str, Any] | None = None
) -> None:
    """Export model checkpoint.

    Args:
        model: Foundry GPT model
        output_path: Path to save checkpoint
        metadata: Optional metadata to include
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model": model.state_dict(),
        "config": {
            "vocab_size": model.config.vocab_size,
            "n_layer": model.config.n_layer,
            "n_head": model.config.n_head,
            "n_kv_head": model.config.n_kv_head,
            "n_embd": model.config.n_embd,
            "block_size": model.config.block_size,
            "bias": model.config.bias,
            "norm_type": model.config.norm_type,
            "activation": model.config.activation,
            "position_encoding": model.config.position_encoding,
            "loss_type": model.config.loss_type,
        },
    }

    if metadata:
        checkpoint["metadata"] = metadata

    torch.save(checkpoint, output_path)
    print(f"Exported checkpoint to {output_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.zoo <model_name>")
        print(f"Available models: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)

    model_name = sys.argv[1]
    model = load_pretrained(model_name)
    print(f"\nLoaded {model_name}:")
    print(f"  Params: {model.get_num_params() / 1e6:.1f}M")
    print(f"  Layers: {model.config.n_layer}")
    print(f"  Heads: {model.config.n_head} (KV: {model.config.n_kv_head})")
    print(f"  Embedding: {model.config.n_embd}")
    print(f"  Vocab: {model.config.vocab_size}")
