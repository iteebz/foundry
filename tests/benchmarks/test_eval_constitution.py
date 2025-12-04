"""Tests for constitution evaluation."""

import json
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from foundry.benchmarks.constitution import (
    evaluate_constitution,
    evaluate_helpfulness,
    score_preference_pair,
)


class DummyTokenizer:
    def encode(self, text):
        return [ord(c) % 100 for c in text[:20]]

    def decode(self, tokens):
        return "This is a helpful response that answers the question."


class DummyModel(nn.Module):
    def __init__(self, prefer_chosen=True):
        super().__init__()
        self.prefer_chosen = prefer_chosen
        self.linear = nn.Linear(10, 10)

    def eval(self):
        return self

    def forward(self, inputs):
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        logits = torch.randn(batch_size, seq_len, 100)

        if self.prefer_chosen and "chosen" in str(inputs):
            logits = logits + 0.5

        return logits

    def generate(self, inputs, max_new_tokens=100, temperature=0.7):
        response = "This is a helpful response that answers the question."
        response_tokens = [ord(c) % 100 for c in response]
        return torch.tensor([response_tokens])


def test_score_preference_pair():
    """Score preference between chosen and rejected."""
    model = DummyModel(prefer_chosen=True)
    tokenizer = DummyTokenizer()

    score = score_preference_pair(model, tokenizer, chosen="Good answer", rejected="Bad answer")

    assert isinstance(score, float)


def test_evaluate_constitution():
    """Constitution evaluation with dummy model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "constitution.jsonl"
        with open(dataset_path, "w") as f:
            f.write(
                json.dumps(
                    {
                        "prompt": "How do I help someone?",
                        "chosen": "Be kind and supportive.",
                        "rejected": "Ignore them.",
                    }
                )
                + "\n"
            )

        model = DummyModel()
        tokenizer = DummyTokenizer()

        results = evaluate_constitution(model, tokenizer, dataset_path, max_samples=1)

        assert "preference_accuracy" in results
        assert "avg_preference_score" in results
        assert "total" in results


def test_evaluate_helpfulness():
    """Helpfulness evaluation with dummy model."""
    model = DummyModel()
    tokenizer = DummyTokenizer()

    prompts = ["How do I cook pasta?", "What is Python?"]

    results = evaluate_helpfulness(model, tokenizer, prompts)

    assert "helpfulness_rate" in results
    assert "helpful_count" in results
    assert results["total"] == 2


def test_constitution_missing_dataset():
    """Handle missing constitution dataset."""
    model = DummyModel()
    tokenizer = DummyTokenizer()

    results = evaluate_constitution(model, tokenizer, "nonexistent.jsonl")

    assert "error" in results
    assert results["preference_accuracy"] == 0.0


if __name__ == "__main__":
    test_score_preference_pair()
    test_evaluate_constitution()
    test_evaluate_helpfulness()
    test_constitution_missing_dataset()
    print("\n✓ All constitution eval tests passed")
