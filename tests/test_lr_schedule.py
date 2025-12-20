"""Tests for learning rate schedule math.

Invariants:
- Warmup: LR increases linearly from ~0 to learning_rate
- Decay: LR follows cosine from learning_rate to min_lr
- Post-decay: LR stays at min_lr
"""

import math
from dataclasses import dataclass

import pytest

from foundry.train import get_lr


@dataclass
class MockTrainingConfig:
    learning_rate: float = 1e-3
    min_lr: float = 1e-5
    warmup_iters: int = 100
    lr_decay_iters: int = 1000


@dataclass
class MockConfig:
    training: MockTrainingConfig = None

    def __post_init__(self):
        if self.training is None:
            self.training = MockTrainingConfig()


def test_lr_at_start_is_small():
    config = MockConfig()
    lr = get_lr(0, config)
    expected = config.training.learning_rate * 1 / (config.training.warmup_iters + 1)
    assert math.isclose(lr, expected, rel_tol=1e-6)


def test_lr_increases_during_warmup():
    config = MockConfig()
    lr_0 = get_lr(0, config)
    lr_50 = get_lr(50, config)
    lr_99 = get_lr(99, config)
    assert lr_0 < lr_50 < lr_99


def test_lr_at_warmup_end():
    config = MockConfig()
    lr = get_lr(config.training.warmup_iters, config)
    expected = config.training.learning_rate
    assert math.isclose(lr, expected, rel_tol=1e-2)


def test_lr_decays_after_warmup():
    config = MockConfig()
    lr_warmup = get_lr(config.training.warmup_iters, config)
    lr_mid = get_lr(500, config)
    lr_end = get_lr(config.training.lr_decay_iters, config)
    assert lr_warmup > lr_mid > lr_end


def test_lr_at_decay_end_is_min():
    config = MockConfig()
    lr = get_lr(config.training.lr_decay_iters, config)
    assert math.isclose(lr, config.training.min_lr, rel_tol=1e-2)


def test_lr_stays_at_min_after_decay():
    config = MockConfig()
    lr_at_end = get_lr(config.training.lr_decay_iters, config)
    lr_after = get_lr(config.training.lr_decay_iters + 1000, config)
    assert math.isclose(lr_at_end, lr_after, rel_tol=1e-6)
    assert math.isclose(lr_after, config.training.min_lr, rel_tol=1e-6)


def test_lr_cosine_midpoint():
    config = MockConfig()
    midpoint = (config.training.warmup_iters + config.training.lr_decay_iters) // 2
    lr = get_lr(midpoint, config)
    expected_mid = (config.training.learning_rate + config.training.min_lr) / 2
    assert math.isclose(lr, expected_mid, rel_tol=0.1)


@pytest.mark.parametrize("it", [0, 50, 100, 500, 1000, 2000])
def test_lr_always_positive(it):
    config = MockConfig()
    lr = get_lr(it, config)
    assert lr > 0


@pytest.mark.parametrize("it", [0, 50, 100, 500, 1000, 2000])
def test_lr_never_exceeds_max(it):
    config = MockConfig()
    lr = get_lr(it, config)
    assert lr <= config.training.learning_rate + 1e-9
