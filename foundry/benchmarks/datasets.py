"""Lazy dataset loading from HuggingFace with version pins."""

from collections.abc import Iterator
from typing import Any

from datasets import load_dataset

DATASET_CONFIGS = {
    "gsm8k": {
        "path": "openai/gsm8k",
        "revision": "refs/convert/parquet",
        "split": "test",
        "name": "main",
    },
    "mmlu": {
        "path": "cais/mmlu",
        "revision": "refs/convert/parquet",
        "split": "test",
        "name": "all",
    },
    "humaneval": {
        "path": "openai/openai_humaneval",
        "revision": "refs/convert/parquet",
        "split": "test",
    },
}


def load_benchmark_dataset(
    task: str,
    max_samples: int | None = None,
    streaming: bool = True,
) -> Iterator[dict[str, Any]]:
    """Load benchmark dataset lazily from HuggingFace.

    Uses streaming by default to avoid downloading entire dataset.
    Version pinned via revision for reproducibility.
    """
    if task not in DATASET_CONFIGS:
        raise ValueError(f"Unknown task: {task}. Available: {list(DATASET_CONFIGS.keys())}")

    config = DATASET_CONFIGS[task]

    ds = load_dataset(
        config["path"],
        name=config.get("name"),
        split=config["split"],
        revision=config["revision"],
        streaming=streaming,
        trust_remote_code=False,
    )

    for count, item in enumerate(ds):
        if max_samples and count >= max_samples:
            break
        yield dict(item)


def get_gsm8k(max_samples: int | None = None) -> Iterator[dict[str, Any]]:
    """Load GSM8K test set."""
    return load_benchmark_dataset("gsm8k", max_samples)


def get_mmlu(max_samples: int | None = None) -> Iterator[dict[str, Any]]:
    """Load MMLU test set.

    Transforms integer answer (0-3) to letter (A-D) for eval compatibility.
    """
    for item in load_benchmark_dataset("mmlu", max_samples):
        item["answer"] = chr(65 + item["answer"])
        yield item


def get_humaneval(max_samples: int | None = None) -> Iterator[dict[str, Any]]:
    """Load HumanEval test set."""
    return load_benchmark_dataset("humaneval", max_samples)
