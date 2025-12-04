"""Tests for evaluation harness."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eval import evaluate


class DummyModel:
    """Mock model for testing."""

    def eval(self):
        pass

    def train(self):
        pass

    def __call__(self, X, Y):
        # Return dummy logits and a fixed loss
        return None, torch.tensor(2.5)


def dummy_get_batch(split):
    """Mock batch generator."""
    return torch.zeros((4, 8)), torch.zeros((4, 8))


def test_evaluate_basic():
    """Eval returns loss and perplexity."""
    model = DummyModel()
    results = evaluate(model, dummy_get_batch, max_iters=10)

    assert "loss" in results
    assert "perplexity" in results
    assert results["loss"] == 2.5
    assert np.isclose(results["perplexity"], np.exp(2.5))


def test_evaluate_shape():
    """Eval works with different batch shapes."""
    model = DummyModel()
    results = evaluate(model, dummy_get_batch, max_iters=5)

    assert isinstance(results["loss"], float)
    assert isinstance(results["perplexity"], float | np.floating)


def test_evaluate_iterations():
    """Eval respects max_iters parameter."""
    call_count = 0

    class CountingModel(DummyModel):
        def __call__(self, X, Y):
            nonlocal call_count
            call_count += 1
            return None, torch.tensor(1.0)

    model = CountingModel()
    evaluate(model, dummy_get_batch, max_iters=20)

    assert call_count == 20
