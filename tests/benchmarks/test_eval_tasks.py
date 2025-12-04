"""Tests for evaluation tasks."""

import json
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from foundry.benchmarks.tasks import (
    evaluate_gsm8k,
    evaluate_humaneval,
    evaluate_mmlu,
    extract_answer,
)


class DummyTokenizer:
    def encode(self, text):
        return [ord(c) % 100 for c in text[:20]]

    def decode(self, tokens):
        return "The answer is 42"


class DummyModel(nn.Module):
    def __init__(self, response="The answer is 42"):
        super().__init__()
        self.response = response

    def eval(self):
        return self

    def generate(self, inputs, max_new_tokens=100, temperature=0.7):
        response_tokens = [ord(c) % 100 for c in self.response]
        return torch.tensor([response_tokens])


def test_extract_answer_math():
    """Extract numerical answer from text."""
    assert extract_answer("The answer is 42", task_type="math") == "42"
    assert extract_answer("It equals 3.14", task_type="math") == "3.14"
    assert extract_answer("#### 1,234", task_type="math") == "1234"


def test_extract_answer_multiple_choice():
    """Extract multiple choice answer."""
    assert extract_answer("The answer is B", task_type="multiple_choice") == "B"
    assert extract_answer("I choose C", task_type="multiple_choice") == "C"


def test_evaluate_gsm8k():
    """GSM8K evaluation with dummy model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "gsm8k_test.jsonl"
        with open(dataset_path, "w") as f:
            f.write(json.dumps({"question": "What is 2+2?", "answer": "#### 42"}) + "\n")
            f.write(json.dumps({"question": "What is 3+3?", "answer": "#### 6"}) + "\n")

        model = DummyModel("The answer is 42")
        tokenizer = DummyTokenizer()

        results = evaluate_gsm8k(model, tokenizer, dataset_path, max_samples=2)

        assert "accuracy" in results
        assert "total" in results
        assert results["total"] == 2


def test_evaluate_mmlu():
    """MMLU evaluation with dummy model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "mmlu_test.jsonl"
        with open(dataset_path, "w") as f:
            f.write(
                json.dumps(
                    {
                        "question": "What is the capital of France?",
                        "choices": ["London", "Paris", "Berlin", "Madrid"],
                        "answer": "B",
                    }
                )
                + "\n"
            )

        model = DummyModel("B")
        tokenizer = DummyTokenizer()

        results = evaluate_mmlu(model, tokenizer, dataset_path, max_samples=1)

        assert "accuracy" in results
        assert "total" in results
        assert results["total"] == 1


def test_evaluate_humaneval():
    """HumanEval evaluation with dummy model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "humaneval.jsonl"
        with open(dataset_path, "w") as f:
            f.write(
                json.dumps(
                    {
                        "prompt": "def add(a, b):\n",
                        "test": "assert add(2, 3) == 5",
                        "entry_point": "add",
                    }
                )
                + "\n"
            )

        model = DummyModel("    return a + b")
        tokenizer = DummyTokenizer()

        results = evaluate_humaneval(model, tokenizer, dataset_path, max_samples=1)

        assert "pass_at_1" in results
        assert "total" in results


def test_missing_dataset():
    """Handle missing dataset gracefully."""
    model = DummyModel()
    tokenizer = DummyTokenizer()

    results = evaluate_gsm8k(model, tokenizer, "nonexistent.jsonl", max_samples=1)

    assert "error" in results
    assert results["accuracy"] == 0.0


if __name__ == "__main__":
    test_extract_answer_math()
    test_extract_answer_multiple_choice()
    test_evaluate_gsm8k()
    test_evaluate_mmlu()
    test_evaluate_humaneval()
    test_missing_dataset()
    print("\n✓ All eval task tests passed")
