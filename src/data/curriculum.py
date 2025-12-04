"""Curriculum learning: order training data by difficulty."""

import torch
from typing import Callable


def score_by_length(tokens: list[int]) -> float:
    """Score difficulty by sequence length (longer = harder)."""
    return float(len(tokens))


def score_by_perplexity(tokens: list[int], model, device: str = "cpu") -> float:
    """Score difficulty by model perplexity (higher = harder)."""
    if len(tokens) == 0:
        return 0.0

    model.eval()
    inputs = torch.tensor([tokens], device=device)

    with torch.no_grad():
        try:
            logits, loss = model(inputs, inputs)
            return loss.item()
        except Exception:
            return 0.0


def order_by_difficulty(
    dataset: list[list[int]], score_fn: Callable[[list[int]], float], reverse: bool = False
) -> list[list[int]]:
    """Order dataset by difficulty score."""
    scored = [(item, score_fn(item)) for item in dataset]
    scored.sort(key=lambda x: x[1], reverse=reverse)
    return [item for item, _ in scored]


def curriculum_schedule(
    dataset: list[list[int]],
    score_fn: Callable[[list[int]], float],
    schedule: str = "linear",
    num_stages: int = 4,
) -> list[list[list[int]]]:
    """Create curriculum schedule (easy → hard stages).
    
    Args:
        dataset: List of tokenized sequences
        score_fn: Function to score difficulty
        schedule: 'linear' or 'step'
        num_stages: Number of curriculum stages
    
    Returns:
        List of dataset stages (each stage is a list of sequences)
    """
    ordered = order_by_difficulty(dataset, score_fn, reverse=False)
    stage_size = len(ordered) // num_stages

    stages = []
    for i in range(num_stages):
        start = 0 if schedule == "linear" else i * stage_size
        end = (i + 1) * stage_size if i < num_stages - 1 else len(ordered)
        stages.append(ordered[start:end])

    return stages


def get_curriculum_stage(epoch: int, total_epochs: int, num_stages: int = 4) -> int:
    """Get curriculum stage for current epoch."""
    stage = int((epoch / total_epochs) * num_stages)
    return min(stage, num_stages - 1)
