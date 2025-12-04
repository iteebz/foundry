"""Constitution dataset formatting for alignment training."""

import json
from pathlib import Path
from typing import Any


def format_preference_pair(
    prompt: str, chosen: str, rejected: str, metadata: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Format preference pair for training.
    
    Args:
        prompt: Input prompt
        chosen: Preferred response
        rejected: Non-preferred response
        metadata: Optional metadata (e.g., source, category)
    
    Returns:
        Formatted preference pair
    """
    pair = {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }

    if metadata:
        pair["metadata"] = metadata

    return pair


def load_constitution(constitution_path: str | Path) -> list[dict[str, Any]]:
    """Load constitution dataset.
    
    Format: JSONL with {prompt, chosen, rejected} per line
    """
    constitution_path = Path(constitution_path)
    if not constitution_path.exists():
        raise FileNotFoundError(f"Constitution not found: {constitution_path}")

    pairs = []
    with open(constitution_path) as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))

    return pairs


def save_constitution(pairs: list[dict[str, Any]], output_path: str | Path) -> None:
    """Save constitution dataset to JSONL."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")


def validate_constitution(pairs: list[dict[str, Any]]) -> bool:
    """Validate constitution dataset format."""
    for pair in pairs:
        if not isinstance(pair, dict):
            return False
        if not all(k in pair for k in ["prompt", "chosen", "rejected"]):
            return False
        if not all(isinstance(pair[k], str) for k in ["prompt", "chosen", "rejected"]):
            return False

    return True
