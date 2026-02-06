"""Multi-task evaluation harness."""

import json
from pathlib import Path
from typing import Any

from foundry.benchmarks.constitution import evaluate_constitution
from foundry.benchmarks.datasets import get_gsm8k, get_humaneval, get_mmlu
from foundry.benchmarks.tasks import evaluate_gsm8k, evaluate_humaneval, evaluate_mmlu
from foundry.types import ModelProtocol, TokenizerProtocol


def run_benchmark_suite(
    model: ModelProtocol,
    tokenizer: TokenizerProtocol,
    tasks: list[str],
    dataset_dir: str | Path | None = None,
    max_samples: int = 100,
    device: str = "cpu",
    use_hf: bool = True,
) -> dict[str, Any]:
    """Run multi-task evaluation suite.

    By default, streams datasets from HuggingFace (lazy download).
    Set use_hf=False and provide dataset_dir to use local JSONL files.
    """
    results = {}

    hf_loaders = {
        "gsm8k": (get_gsm8k, evaluate_gsm8k),
        "mmlu": (get_mmlu, evaluate_mmlu),
        "humaneval": (get_humaneval, evaluate_humaneval),
    }

    local_files = {
        "gsm8k": "gsm8k_test.jsonl",
        "mmlu": "mmlu_test.jsonl",
        "humaneval": "humaneval.jsonl",
        "constitution": "constitution.jsonl",
    }

    eval_fns = {
        "gsm8k": evaluate_gsm8k,
        "mmlu": evaluate_mmlu,
        "humaneval": evaluate_humaneval,
        "constitution": evaluate_constitution,
    }

    for task in tasks:
        if task not in eval_fns:
            results[task] = {"error": f"Unknown task: {task}"}
            continue

        try:
            if use_hf and task in hf_loaders:
                loader, eval_fn = hf_loaders[task]
                dataset = loader(max_samples)
                task_results = eval_fn(model, tokenizer, dataset, max_samples, device)
            else:
                if dataset_dir is None:
                    results[task] = {"error": f"No dataset_dir for local task: {task}"}
                    continue
                dataset_path = Path(dataset_dir) / local_files[task]
                eval_fn = eval_fns[task]
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

    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
