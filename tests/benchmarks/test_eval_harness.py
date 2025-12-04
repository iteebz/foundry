"""Tests for eval harness."""

import json
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from foundry.benchmarks.harness import (
    compute_aggregate_metrics,
    run_benchmark_suite,
    save_eval_results,
)


class DummyTokenizer:
    def encode(self, text):
        return [ord(c) % 100 for c in text[:20]]

    def decode(self, tokens):
        return "The answer is 42"


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def eval(self):
        return self

    def forward(self, inputs):
        return torch.randn(inputs.size(0), inputs.size(1), 100)

    def generate(self, inputs, max_new_tokens=100, temperature=0.7):
        return torch.tensor([[42] * 20])


def test_compute_aggregate_metrics():
    """Compute aggregate from task results."""
    results = {
        "gsm8k": {"accuracy": 0.5},
        "mmlu": {"accuracy": 0.6},
        "humaneval": {"pass_at_1": 0.4},
    }

    agg = compute_aggregate_metrics(results)

    assert "mean_score" in agg
    assert "num_tasks" in agg
    assert agg["num_tasks"] == 3
    assert 0.0 <= agg["mean_score"] <= 1.0


def test_run_benchmark_suite():
    """Run multi-task eval suite."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_dir = Path(tmpdir)
        (dataset_dir / "gsm8k_test.jsonl").write_text(
            json.dumps({"question": "What is 2+2?", "answer": "#### 4"}) + "\n"
        )

        model = DummyModel()
        tokenizer = DummyTokenizer()

        results = run_benchmark_suite(
            model, tokenizer, tasks=["gsm8k"], dataset_dir=dataset_dir, max_samples=1
        )

        assert "gsm8k" in results
        assert "aggregate" in results


def test_save_eval_results():
    """Save eval results to JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "results.json"
        results = {"gsm8k": {"accuracy": 0.5}, "aggregate": {"mean_score": 0.5}}

        save_eval_results(results, output_path)

        assert output_path.exists()

        with open(output_path) as f:
            loaded = json.load(f)

        assert loaded["gsm8k"]["accuracy"] == 0.5


def test_unknown_task():
    """Handle unknown task gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = DummyModel()
        tokenizer = DummyTokenizer()

        results = run_benchmark_suite(model, tokenizer, tasks=["unknown_task"], dataset_dir=tmpdir)

        assert "unknown_task" in results
        assert "error" in results["unknown_task"]


if __name__ == "__main__":
    test_compute_aggregate_metrics()
    test_run_benchmark_suite()
    test_save_eval_results()
    test_unknown_task()
    print("\n✓ All eval harness tests passed")
