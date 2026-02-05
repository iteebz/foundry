"""Tests for LayerNorm."""

import torch

from foundry.modules.layernorm import LayerNorm


def test_layernorm_creation():
    """LayerNorm initializes correctly."""
    ln = LayerNorm(ndim=256)
    assert ln.weight.shape == (256,)
    assert ln.bias.shape == (256,)


def test_layernorm_creation_no_bias():
    """LayerNorm works without bias."""
    ln = LayerNorm(ndim=256, bias=False)
    assert ln.weight.shape == (256,)
    assert ln.bias is None


def test_layernorm_forward():
    """LayerNorm forward pass preserves shape."""
    ln = LayerNorm(ndim=128)
    x = torch.randn(2, 16, 128)
    y = ln(x)
    assert y.shape == (2, 16, 128)


def test_layernorm_normalizes():
    """LayerNorm produces roughly zero mean, unit variance."""
    ln = LayerNorm(ndim=256)
    x = torch.randn(4, 32, 256) * 10 + 5
    y = ln(x)
    assert y.mean().abs() < 0.1
    assert (y.std() - 1.0).abs() < 0.1


def test_layernorm_deterministic():
    """LayerNorm is deterministic."""
    ln = LayerNorm(ndim=64)
    x = torch.randn(1, 8, 64)
    y1 = ln(x)
    y2 = ln(x)
    assert torch.allclose(y1, y2)


if __name__ == "__main__":
    test_layernorm_creation()
    test_layernorm_creation_no_bias()
    test_layernorm_forward()
    test_layernorm_normalizes()
    test_layernorm_deterministic()
    print("\n✓ All LayerNorm tests passed")
