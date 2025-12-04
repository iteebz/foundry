"""Tests for curriculum learning."""

import torch
import torch.nn as nn

from foundry.data.curriculum import (
    curriculum_schedule,
    get_curriculum_stage,
    order_by_difficulty,
    score_by_length,
    score_by_perplexity,
)


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def eval(self):
        return self

    def __call__(self, inputs, targets):
        loss = torch.tensor(float(inputs.size(1)) / 10.0)
        return None, loss


def test_score_by_length():
    """Score by length returns sequence length."""
    tokens = [1, 2, 3, 4, 5]
    score = score_by_length(tokens)
    assert score == 5.0


def test_score_by_perplexity():
    """Score by perplexity uses model loss."""
    model = DummyModel()
    tokens = [1, 2, 3, 4, 5]

    score = score_by_perplexity(tokens, model)
    assert isinstance(score, float)
    assert score >= 0.0


def test_order_by_difficulty():
    """Order dataset by difficulty."""
    dataset = [[1, 2], [1, 2, 3, 4, 5], [1, 2, 3]]

    ordered = order_by_difficulty(dataset, score_by_length)

    assert len(ordered[0]) == 2
    assert len(ordered[1]) == 3
    assert len(ordered[2]) == 5


def test_curriculum_schedule_linear():
    """Linear curriculum includes all previous stages."""
    dataset = [[i] * (i + 1) for i in range(8)]

    stages = curriculum_schedule(dataset, score_by_length, schedule="linear", num_stages=4)

    assert len(stages) == 4
    assert len(stages[0]) == 2
    assert len(stages[1]) == 4
    assert len(stages[2]) == 6
    assert len(stages[3]) == 8


def test_curriculum_schedule_step():
    """Step curriculum has disjoint stages."""
    dataset = [[i] * (i + 1) for i in range(8)]

    stages = curriculum_schedule(dataset, score_by_length, schedule="step", num_stages=4)

    assert len(stages) == 4
    for stage in stages[:3]:
        assert len(stage) == 2


def test_get_curriculum_stage():
    """Get stage for current epoch."""
    assert get_curriculum_stage(0, 100, num_stages=4) == 0
    assert get_curriculum_stage(25, 100, num_stages=4) == 1
    assert get_curriculum_stage(50, 100, num_stages=4) == 2
    assert get_curriculum_stage(75, 100, num_stages=4) == 3
    assert get_curriculum_stage(99, 100, num_stages=4) == 3


if __name__ == "__main__":
    test_score_by_length()
    test_score_by_perplexity()
    test_order_by_difficulty()
    test_curriculum_schedule_linear()
    test_curriculum_schedule_step()
    test_get_curriculum_stage()
    print("\n✓ All curriculum tests passed")
