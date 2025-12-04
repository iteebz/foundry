"""Curriculum learning: order training data by difficulty."""

from collections.abc import Callable

import torch


def score_by_length(tokens: list[int]) -> float:
    """Score difficulty by sequence length (longer = harder)."""
    return float(len(tokens))


def score_by_perplexity(tokens: list[int], model, device: str = "cpu") -> float:
    if len(tokens) == 0:
        return 0.0

    if len(tokens) < 2:
        raise ValueError(f"Sequence too short for perplexity: {len(tokens)} tokens")

    model.eval()
    inputs = torch.tensor([tokens], device=device)

    with torch.no_grad():
        try:
            output = model(inputs, inputs)
            if not isinstance(output, tuple) or len(output) < 2:
                raise ValueError(f"Model must return (logits, loss), got {type(output)}")
            logits, loss = output
            return float(loss.item())
        except Exception as e:
            raise ValueError(f"Model evaluation failed: {e}") from e


def order_by_difficulty(
    dataset: list[list[int]], score_fn: Callable[[list[int]], float], reverse: bool = False
) -> tuple[list[list[int]], list[float]]:
    scored = [(item, score_fn(item)) for item in dataset]
    scored.sort(key=lambda x: x[1], reverse=reverse)
    return [item for item, _ in scored], [score for _, score in scored]


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
    ordered, _ = order_by_difficulty(dataset, score_fn, reverse=False)
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
