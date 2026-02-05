"""Benchmark task evaluation (GSM8K, MMLU, HumanEval)."""

import json
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import torch


def extract_answer(text: str, task_type: str = "math") -> str | None:
    """Extract answer from model output."""
    if task_type == "math":
        matches = re.findall(r"####\s*([0-9,\.]+)", text)
        if matches:
            return matches[-1].replace(",", "")
        matches = re.findall(r"(?:answer is|equals?)\s*([0-9,\.]+)", text.lower())
        if matches:
            return matches[-1].replace(",", "")
        numbers = re.findall(r"\b([0-9,\.]+)\b", text)
        if numbers:
            return numbers[-1].replace(",", "")
    elif task_type == "multiple_choice":
        matches = re.findall(r"\b([A-D])\b", text.upper())
        if matches:
            return matches[0]
    return None


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


def evaluate_gsm8k(
    model,
    tokenizer,
    dataset: str | Path | Iterable[dict[str, Any]],
    max_samples: int = 100,
    device: str = "cpu",
) -> dict[str, Any]:
    """Evaluate on GSM8K math reasoning."""
    data = list(_load_data(dataset, max_samples))
    if not data:
        return {"error": f"Dataset empty or not found: {dataset}", "accuracy": 0.0}

    correct = 0
    total = 0

    model.eval()
    for item in data:
        question = item["question"]
        answer = item["answer"].split("####")[-1].strip().replace(",", "")

        prompt = f"Question: {question}\nAnswer:"
        inputs = tokenizer.encode(prompt)
        inputs = torch.tensor([inputs], device=device)

        with torch.no_grad():
            outputs = model.generate(inputs, max_new_tokens=200, temperature=0.7)
            generated = tokenizer.decode(outputs[0].tolist())

        pred_answer = extract_answer(generated, task_type="math")

        if pred_answer and pred_answer == answer:
            correct += 1
        total += 1

    return {"accuracy": correct / total if total > 0 else 0.0, "correct": correct, "total": total}


def evaluate_mmlu(
    model,
    tokenizer,
    dataset: str | Path | Iterable[dict[str, Any]],
    max_samples: int = 100,
    device: str = "cpu",
) -> dict[str, Any]:
    """Evaluate on MMLU multiple choice knowledge."""
    data = list(_load_data(dataset, max_samples))
    if not data:
        return {"error": f"Dataset empty or not found: {dataset}", "accuracy": 0.0}

    correct = 0
    total = 0

    model.eval()
    for item in data:
        question = item["question"]
        choices = item["choices"]
        answer = item["answer"]

        prompt = f"Question: {question}\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65 + i)}. {choice}\n"
        prompt += "Answer:"

        inputs = tokenizer.encode(prompt)
        inputs = torch.tensor([inputs], device=device)

        with torch.no_grad():
            outputs = model.generate(inputs, max_new_tokens=50, temperature=0.0)
            generated = tokenizer.decode(outputs[0].tolist())

        pred_answer = extract_answer(generated, task_type="multiple_choice")

        if pred_answer and pred_answer == answer:
            correct += 1
        total += 1

    return {"accuracy": correct / total if total > 0 else 0.0, "correct": correct, "total": total}


def evaluate_humaneval(
    model,
    tokenizer,
    dataset: str | Path | Iterable[dict[str, Any]],
    max_samples: int = 50,
    device: str = "cpu",
) -> dict[str, Any]:
    """Evaluate on HumanEval code generation."""
    data = list(_load_data(dataset, max_samples))
    if not data:
        return {"error": f"Dataset empty or not found: {dataset}", "pass_at_1": 0.0}

    passed = 0
    total = 0

    model.eval()
    for item in data:
        prompt = item["prompt"]
        test = item.get("test", "")
        entry_point = item.get("entry_point", "")

        inputs = tokenizer.encode(prompt)
        inputs = torch.tensor([inputs], device=device)

        with torch.no_grad():
            outputs = model.generate(inputs, max_new_tokens=300, temperature=0.2)
            generated = tokenizer.decode(outputs[0].tolist())

        code = generated[len(prompt) :]

        try:
            exec_globals = {}
            exec(code, exec_globals)  # noqa: S102 - benchmark requires code execution
            if entry_point and entry_point in exec_globals:
                exec(test, exec_globals)  # noqa: S102 - benchmark requires code execution
                passed += 1
        except Exception:  # noqa: S110 - benchmark silently skips failed tests
            pass

        total += 1

    return {
        "pass_at_1": passed / total if total > 0 else 0.0,
        "passed": passed,
        "total": total,
    }
