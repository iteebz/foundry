"""Data pipeline mutation functions."""

from typing import Any

from .core import load_baseline


def mutate_data_filter(
    min_len: int,
    max_len: int | None = None,
    dedupe: bool = True,
    base_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
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


def mutate_conversation_format(
    format_type: str = "chatml", base_config: dict[str, Any] | None = None
) -> dict[str, Any]:
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
