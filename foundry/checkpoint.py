"""Checkpoint bridge for loading pretrained HuggingFace models into foundry."""

import torch


def load_hf_checkpoint(model, hf_model_name: str, cache_dir: str = None) -> None:
    """Load HuggingFace checkpoint weights into foundry model.

    Supports GPT-2 family models. Maps HF state dict to foundry architecture.
    Modifies model in-place.

    Args:
        model: Foundry GPT model instance (from model.py v1)
        hf_model_name: HF model identifier (e.g., 'gpt2', 'gpt2-medium')
        cache_dir: Optional cache directory for HF downloads
    """
    try:
        from transformers import GPT2LMHeadModel
    except ImportError as e:
        raise ImportError(
            "transformers required for checkpoint loading: pip install transformers"
        ) from e

    hf_model = GPT2LMHeadModel.from_pretrained(hf_model_name, cache_dir=cache_dir)
    hf_sd = hf_model.state_dict()

    foundry_sd = model.state_dict()
    foundry_keys = [k for k in foundry_sd if not k.endswith(".attn.bias")]
    hf_keys = [
        k for k in hf_sd if not k.endswith(".attn.masked_bias") and not k.endswith(".attn.bias")
    ]

    transposed = [
        "attn.c_attn.weight",
        "attn.c_proj.weight",
        "mlp.c_fc.weight",
        "mlp.c_proj.weight",
    ]

    assert len(hf_keys) == len(foundry_keys), f"Key mismatch: {len(hf_keys)} vs {len(foundry_keys)}"

    for hf_key in hf_keys:
        foundry_key = (
            hf_key.replace("transformer.h.", "h.")
            .replace("transformer.wte.", "wte.")
            .replace("transformer.wpe.", "wpe.")
            .replace("transformer.ln_f.", "ln_f.")
        )
        if foundry_key not in foundry_sd:
            raise KeyError(f"HF key {hf_key} has no foundry mapping: {foundry_key}")

        if any(hf_key.endswith(w) for w in transposed):
            assert hf_sd[hf_key].shape[::-1] == foundry_sd[foundry_key].shape
            with torch.no_grad():
                foundry_sd[foundry_key].copy_(hf_sd[hf_key].t())
        else:
            assert hf_sd[hf_key].shape == foundry_sd[foundry_key].shape
            with torch.no_grad():
                foundry_sd[foundry_key].copy_(hf_sd[hf_key])

    model.load_state_dict(foundry_sd)


def save_checkpoint(model, optimizer, config: dict, path: str) -> None:
    """Save foundry checkpoint to disk."""
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
    }
    torch.save(checkpoint, path)


def validate_checkpoint(checkpoint: dict) -> None:
    """Validate checkpoint integrity.

    Raises:
        ValueError: If checkpoint is corrupted
    """
    if len(checkpoint.get("model", {})) == 0:
        raise ValueError("Checkpoint has empty model state dict")


def load_checkpoint(model, optimizer, path: str) -> dict:
    """Load foundry checkpoint from disk.

    Returns:
        dict: Checkpoint metadata (config, iter_num, etc.)

    Raises:
        ValueError: If checkpoint is corrupted
    """
    checkpoint = torch.load(path, map_location="cpu")

    validate_checkpoint(checkpoint)

    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    model.load_state_dict(state_dict)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint.get("config", {})
