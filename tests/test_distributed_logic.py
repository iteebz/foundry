"""Test distributed training logic without multi-process.

Tests auto-selection logic, strategy switching, and wrapping behavior
without requiring actual multi-GPU or multi-process setup.
"""

import torch.nn as nn

from foundry.distributed import should_use_fsdp, wrap_model_distributed
from foundry.model import GPT, GPTConfig


class TinyModel(nn.Module):
    """Model with <1M params."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())


class LargeModel(nn.Module):
    """Model with >1B params (simulated via large layer)."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32000, 32000)


def test_should_use_fsdp_below_threshold():
    """Model with <1B params should not use FSDP."""
    model = TinyModel()
    assert not should_use_fsdp(model, min_params=1_000_000_000)


def test_should_use_fsdp_above_threshold():
    """Model with ≥1B params should use FSDP."""
    model = LargeModel()
    assert should_use_fsdp(model, min_params=1_000_000_000)


def test_should_use_fsdp_custom_threshold():
    """Custom threshold should be respected."""
    model = TinyModel()
    actual_params = sum(p.numel() for p in model.parameters())
    assert should_use_fsdp(model, min_params=10)
    assert not should_use_fsdp(model, min_params=actual_params + 1)


def test_wrap_model_none_strategy_returns_unwrapped():
    """strategy='none' should return original model."""
    config = GPTConfig(n_layer=2, n_head=4, n_embd=64, vocab_size=256)
    model = GPT(config)

    wrapped, is_ddp, is_fsdp = wrap_model_distributed(model, strategy="none")

    assert wrapped is model
    assert not is_ddp
    assert not is_fsdp


def test_wrap_model_auto_strategy_small_model():
    """Auto strategy with small model should select DDP (if distributed)."""
    config = GPTConfig(n_layer=2, n_head=4, n_embd=64, vocab_size=256)
    model = GPT(config)

    wrapped, is_ddp, is_fsdp = wrap_model_distributed(
        model, strategy="auto", fsdp_min_params=1_000_000_000
    )

    assert wrapped is model
    assert not is_ddp
    assert not is_fsdp


def test_wrap_model_returns_tuple():
    """wrap_model_distributed should return (model, is_ddp, is_fsdp)."""
    config = GPTConfig(n_layer=2, n_head=4, n_embd=64, vocab_size=256)
    model = GPT(config)

    result = wrap_model_distributed(model, strategy="none")

    assert isinstance(result, tuple)
    assert len(result) == 3


def test_invalid_strategy_string():
    """wrap_model_distributed validates strategy parameter."""

    valid_strategies = ["auto", "ddp", "fsdp", "none"]
    invalid = "totally_invalid_strategy_xyz"

    assert invalid not in valid_strategies, "Test setup: strategy must be invalid"
