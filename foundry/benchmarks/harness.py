"""Multi-task evaluation harness."""

import json
from pathlib import Path
from typing import Any

from foundry.benchmarks.constitution import evaluate_constitution
from foundry.benchmarks.tasks import evaluate_gsm8k, evaluate_humaneval, evaluate_mmlu


def run_benchmark_suite(
    model,
    tokenizer,
    tasks: list[str],
    dataset_dir: str | Path = "data/eval",
    max_samples: int = 100,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run multi-task evaluation suite."""
    dataset_dir = Path(dataset_dir)
    results = {}

    task_map = {
        "gsm8k": ("gsm8k_test.jsonl", evaluate_gsm8k),
        "mmlu": ("mmlu_test.jsonl", evaluate_mmlu),
        "humaneval": ("humaneval.jsonl", evaluate_humaneval),
        "constitution": ("constitution.jsonl", evaluate_constitution),
    }

    for task in tasks:
        if task not in task_map:
            results[task] = {"error": f"Unknown task: {task}"}
            continue

        dataset_file, eval_fn = task_map[task]
        dataset_path = dataset_dir / dataset_file

        try:
            task_results = eval_fn(model, tokenizer, dataset_path, max_samples, device)
            results[task] = task_results
        except Exception as e:
            results[task] = {"error": str(e)}

    aggregate = compute_aggregate_metrics(results)
    results["aggregate"] = aggregate

    return results


def compute_aggregate_metrics(results: dict[str, Any]) -> dict[str, Any]:
    """Compute aggregate metrics across tasks."""
    scores = []

    if "gsm8k" in results and "accuracy" in results["gsm8k"]:
        scores.append(results["gsm8k"]["accuracy"])

    if "mmlu" in results and "accuracy" in results["mmlu"]:
        scores.append(results["mmlu"]["accuracy"])

    if "humaneval" in results and "pass_at_1" in results["humaneval"]:
        scores.append(results["humaneval"]["pass_at_1"])

    if "constitution" in results and "preference_accuracy" in results["constitution"]:
        scores.append(results["constitution"]["preference_accuracy"])

    return {
        "mean_score": sum(scores) / len(scores) if scores else 0.0,
        "num_tasks": len(scores),
    }


def save_eval_results(results: dict[str, Any], output_path: str | Path) -> None:
    """Save evaluation results to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
