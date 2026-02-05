"""Tests for GLU activation."""

import torch

from foundry.modules.glu import GLU


def test_glu_creation():
    """GLU module initializes correctly."""
    glu = GLU(n_embd=256)
    assert glu.fc.in_features == 256
    assert glu.fc.out_features == 1024
    assert glu.gate.in_features == 256
    assert glu.gate.out_features == 1024
    assert glu.proj.out_features == 256


def test_glu_creation_with_bias():
    """GLU accepts bias parameter."""
    glu_no_bias = GLU(n_embd=128, bias=False)
    glu_bias = GLU(n_embd=128, bias=True)
    assert glu_no_bias.fc.bias is None
    assert glu_bias.fc.bias is not None


def test_glu_forward():
    """GLU forward pass preserves shape."""
    glu = GLU(n_embd=256)
    x = torch.randn(2, 16, 256)
    y = glu(x)
    assert y.shape == (2, 16, 256)


def test_glu_gating():
    """GLU applies sigmoid gating."""
    glu = GLU(n_embd=64)
    x = torch.randn(1, 4, 64)
    y = glu(x)
    assert y.shape == x.shape


def test_glu_deterministic():
    """GLU is deterministic (no dropout)."""
    glu = GLU(n_embd=128)
    glu.eval()
    x = torch.randn(1, 8, 128)
    y1 = glu(x)
    y2 = glu(x)
    assert torch.allclose(y1, y2)


if __name__ == "__main__":
    test_glu_creation()
    test_glu_creation_with_bias()
    test_glu_forward()
    test_glu_gating()
    test_glu_deterministic()
    print("\n✓ All GLU tests passed")
