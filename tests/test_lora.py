"""Tests for LoRA module."""

import torch
import torch.nn as nn

from foundry.lora import (
    LoRALinear,
    apply_lora_to_model,
    get_lora_params,
    mark_only_lora_as_trainable,
)


def test_lora_linear_forward():
    """LoRA linear layer forward pass works."""
    layer = LoRALinear(64, 128, r=8, lora_alpha=16)
    x = torch.randn(4, 64)
    out = layer(x)
    assert out.shape == (4, 128)


def test_lora_linear_frozen_base():
    """Base linear weights are frozen."""
    layer = LoRALinear(64, 128, r=8)
    assert not layer.linear.weight.requires_grad
    if layer.linear.bias is not None:
        assert not layer.linear.bias.requires_grad


def test_lora_linear_trainable_adapters():
    """LoRA adapter weights are trainable."""
    layer = LoRALinear(64, 128, r=8)
    assert layer.lora_A.requires_grad
    assert layer.lora_B.requires_grad


def test_lora_zero_rank():
    """r=0 disables LoRA, acts as frozen linear."""
    layer = LoRALinear(64, 128, r=0)
    x = torch.randn(4, 64)
    out = layer(x)
    assert out.shape == (4, 128)


def test_apply_lora_to_model():
    """LoRA can be applied to a simple model."""

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(64, 128)
            self.fc2 = nn.Linear(128, 64)

        def forward(self, x):
            return self.fc2(self.fc1(x))

    model = SimpleModel()
    model = apply_lora_to_model(model, r=8, lora_alpha=16)

    assert isinstance(model.fc1, LoRALinear)
    assert isinstance(model.fc2, LoRALinear)


def test_mark_only_lora_trainable():
    """Only LoRA parameters are trainable after marking."""

    class ModelWithLoRA(nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_layer = LoRALinear(64, 128, r=8)
            self.normal_layer = nn.Linear(128, 64)

    model = ModelWithLoRA()
    mark_only_lora_as_trainable(model)

    lora_params = [p for n, p in model.named_parameters() if "lora_" in n]
    other_params = [p for n, p in model.named_parameters() if "lora_" not in n]

    assert all(p.requires_grad for p in lora_params)
    assert all(not p.requires_grad for p in other_params)


def test_get_lora_params():
    """LoRA param stats are computed correctly."""
    layer = LoRALinear(64, 128, r=8)
    stats = get_lora_params(layer)

    assert stats["total_params"] > 0
    assert stats["trainable_params"] > 0
    assert 0 < stats["trainable_pct"] < 100


def test_lora_merge_unmerge():
    """LoRA weights can be merged and unmerged."""
    layer = LoRALinear(64, 128, r=8)

    layer.lora_B.data.normal_(0, 0.01)

    original_weight = layer.linear.weight.data.clone()

    layer.merge()
    assert not torch.allclose(layer.linear.weight.data, original_weight)

    layer.unmerge()
    assert torch.allclose(layer.linear.weight.data, original_weight)


if __name__ == "__main__":
    test_lora_linear_forward()
    test_lora_linear_frozen_base()
    test_lora_linear_trainable_adapters()
    test_lora_zero_rank()
    test_apply_lora_to_model()
    test_mark_only_lora_trainable()
    test_get_lora_params()
    test_lora_merge_unmerge()
    print("\n✓ All LoRA tests passed")
