"""Tests for GELU activation."""

import torch

from foundry.modules.gelu import GELU


def test_gelu_creation():
    """GELU module initializes correctly."""
    gelu = GELU(n_embd=256)
    assert gelu.fc.in_features == 256
    assert gelu.fc.out_features == 1024
    assert gelu.proj.in_features == 1024
    assert gelu.proj.out_features == 256


def test_gelu_creation_with_bias():
    """GELU accepts bias parameter."""
    gelu_no_bias = GELU(n_embd=128, bias=False)
    gelu_bias = GELU(n_embd=128, bias=True)
    assert gelu_no_bias.fc.bias is None
    assert gelu_bias.fc.bias is not None


def test_gelu_forward():
    """GELU forward pass preserves shape."""
    gelu = GELU(n_embd=256)
    x = torch.randn(2, 16, 256)
    y = gelu(x)
    assert y.shape == (2, 16, 256)


def test_gelu_expansion():
    """GELU expands to 4x then projects back."""
    gelu = GELU(n_embd=64)
    assert gelu.fc.out_features == 256
    assert gelu.proj.in_features == 256
    assert gelu.proj.out_features == 64


def test_gelu_deterministic():
    """GELU is deterministic (no dropout)."""
    gelu = GELU(n_embd=128)
    gelu.eval()
    x = torch.randn(1, 8, 128)
    y1 = gelu(x)
    y2 = gelu(x)
    assert torch.allclose(y1, y2)


if __name__ == "__main__":
    test_gelu_creation()
    test_gelu_creation_with_bias()
    test_gelu_forward()
    test_gelu_expansion()
    test_gelu_deterministic()
    print("\n✓ All GELU tests passed")
