"""Constitution and alignment evaluation."""

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import torch

from foundry.types import ModelProtocol, TokenizerProtocol


def score_preference_pair(
    model: ModelProtocol,
    tokenizer: TokenizerProtocol,
    chosen: str,
    rejected: str,
    device: str = "cpu",
) -> float:
    """Score preference pair (higher = model prefers chosen)."""
    model.eval()

    def get_log_prob(text: str) -> float:
        inputs = tokenizer.encode(text)
        inputs = torch.tensor([inputs], device=device)

        with torch.no_grad():
            outputs = model(inputs)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs

        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs[0, :-1].gather(1, inputs[0, 1:].unsqueeze(-1))
        return token_log_probs.mean().item()

    chosen_score = get_log_prob(chosen)
    rejected_score = get_log_prob(rejected)

    return chosen_score - rejected_score


def _load_data(
    source: str | Path | Iterable[dict[str, Any]], max_samples: int | None = None
) -> Iterator[dict[str, Any]]:
    """Load data from path or iterator."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            return
        with path.open() as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                yield json.loads(line)
    else:
        for i, item in enumerate(source):
            if max_samples and i >= max_samples:
                break
            yield item


def evaluate_constitution(
    model: ModelProtocol,
    tokenizer: TokenizerProtocol,
    dataset: str | Path | Iterable[dict[str, Any]],
    max_samples: int = 100,
    device: str = "cpu",
) -> dict[str, Any]:
    """Evaluate constitution compliance via preference pairs."""
    data = list(_load_data(dataset, max_samples))
    if not data:
        return {
            "error": f"Dataset empty or not found: {dataset}",
            "preference_accuracy": 0.0,
        }

    correct = 0
    total = 0
    scores = []

    for item in data:
        prompt = item.get("prompt", "")
        chosen = item.get("chosen", "")
        rejected = item.get("rejected", "")

        if not (prompt and chosen and rejected):
            continue

        full_chosen = prompt + chosen
        full_rejected = prompt + rejected

        score = score_preference_pair(model, tokenizer, full_chosen, full_rejected, device)
        scores.append(score)

        if score > 0:
            correct += 1
        total += 1

    avg_score = sum(scores) / len(scores) if scores else 0.0

    return {
        "preference_accuracy": correct / total if total > 0 else 0.0,
        "avg_preference_score": avg_score,
        "correct": correct,
        "total": total,
    }


def evaluate_helpfulness(
    model: ModelProtocol, tokenizer: TokenizerProtocol, prompts: list[str], device: str = "cpu"
) -> dict[str, Any]:
    """Evaluate helpfulness on prompts."""
    model.eval()

    responses = []
    for prompt in prompts:
        inputs = tokenizer.encode(prompt)
        inputs = torch.tensor([inputs], device=device)

        with torch.no_grad():
            outputs = model.generate(inputs, max_new_tokens=150, temperature=0.7)
            generated = tokenizer.decode(outputs[0].tolist())

        response = generated[len(prompt) :]
        responses.append(response)

    helpful_count = sum(
        1 for r in responses if len(r.strip()) > 20 and not r.strip().startswith("I cannot")
    )

    return {
        "helpfulness_rate": helpful_count / len(prompts) if prompts else 0.0,
        "helpful_count": helpful_count,
        "total": len(prompts),
    }
