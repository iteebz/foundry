"""Benchmark task evaluation (GSM8K, MMLU, HumanEval)."""

import json
import re
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


def evaluate_gsm8k(
    model, tokenizer, dataset_path: str | Path, max_samples: int = 100, device: str = "cpu"
) -> dict[str, Any]:
    """Evaluate on GSM8K math reasoning."""
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        return {"error": f"Dataset not found: {dataset_path}", "accuracy": 0.0}

    with open(dataset_path) as f:
        data = [json.loads(line) for line in f]

    if max_samples:
        data = data[:max_samples]

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
    model, tokenizer, dataset_path: str | Path, max_samples: int = 100, device: str = "cpu"
) -> dict[str, Any]:
    """Evaluate on MMLU multiple choice knowledge."""
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        return {"error": f"Dataset not found: {dataset_path}", "accuracy": 0.0}

    with open(dataset_path) as f:
        data = [json.loads(line) for line in f]

    if max_samples:
        data = data[:max_samples]

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
    model, tokenizer, dataset_path: str | Path, max_samples: int = 50, device: str = "cpu"
) -> dict[str, Any]:
    """Evaluate on HumanEval code generation."""
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        return {"error": f"Dataset not found: {dataset_path}", "pass_at_1": 0.0}

    with open(dataset_path) as f:
        data = [json.loads(line) for line in f]

    if max_samples:
        data = data[:max_samples]

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
            exec(code, exec_globals)
            if entry_point and entry_point in exec_globals:
                exec(test, exec_globals)
                passed += 1
        except Exception:
            pass

        total += 1

    return {
        "pass_at_1": passed / total if total > 0 else 0.0,
        "passed": passed,
        "total": total,
    }
