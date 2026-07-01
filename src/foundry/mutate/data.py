"""Data pipeline mutation functions."""

from pathlib import Path
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


def mutate_curriculum(
    strategy: str = "length",
    schedule: str = "linear",
    num_stages: int = 4,
    base_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate curriculum learning mutations.

    Args:
        strategy: Difficulty scoring ('length' or 'perplexity')
        schedule: Curriculum schedule ('linear' or 'step')
        num_stages: Number of curriculum stages
    """
    config = base_config or load_baseline()

    strategies = ["length", "perplexity"]
    if strategy not in strategies:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from {strategies}")

    schedules = ["linear", "step"]
    if schedule not in schedules:
        raise ValueError(f"Unknown schedule: {schedule}. Choose from {schedules}")

    config["name"] = f"curriculum_{strategy}_{schedule}_{num_stages}stages"
    if "data" not in config:
        config["data"] = {}
    config["data"]["curriculum"] = {
        "enabled": True,
        "strategy": strategy,
        "schedule": schedule,
        "num_stages": num_stages,
    }

    return config


def mutate_constitution(
    constitution_path: str, base_config: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Generate constitution injection mutation.

    Args:
        constitution_path: Path to constitution dataset (JSONL with preference pairs)
    """
    config = base_config or load_baseline()

    config["name"] = f"constitution_{Path(constitution_path).stem}"
    if "data" not in config:
        config["data"] = {}
    config["data"]["constitution"] = {
        "enabled": True,
        "path": constitution_path,
    }

    return config
