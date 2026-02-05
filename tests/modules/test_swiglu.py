"""Tests for SwiGLU activation."""

import torch

from foundry.modules.swiglu import SwiGLU


def test_swiglu_creation():
    """SwiGLU initializes with default hidden_dim."""
    swiglu = SwiGLU(dim=256)
    expected_hidden = int(2 * 4 * 256 / 3)
    assert swiglu.w1.in_features == 256
    assert swiglu.w1.out_features == expected_hidden
    assert swiglu.w2.out_features == 256


def test_swiglu_creation_custom_hidden():
    """SwiGLU accepts custom hidden_dim."""
    swiglu = SwiGLU(dim=256, hidden_dim=512)
    assert swiglu.w1.out_features == 512
    assert swiglu.w3.out_features == 512


def test_swiglu_creation_with_bias():
    """SwiGLU accepts bias parameter."""
    swiglu_no_bias = SwiGLU(dim=128, bias=False)
    swiglu_bias = SwiGLU(dim=128, bias=True)
    assert swiglu_no_bias.w1.bias is None
    assert swiglu_bias.w1.bias is not None


def test_swiglu_forward():
    """SwiGLU forward pass preserves shape."""
    swiglu = SwiGLU(dim=256)
    x = torch.randn(2, 16, 256)
    y = swiglu(x)
    assert y.shape == (2, 16, 256)


def test_swiglu_three_weights():
    """SwiGLU has three weight matrices (w1, w2, w3)."""
    swiglu = SwiGLU(dim=128)
    assert hasattr(swiglu, "w1")
    assert hasattr(swiglu, "w2")
    assert hasattr(swiglu, "w3")


def test_swiglu_deterministic():
    """SwiGLU is deterministic (no dropout)."""
    swiglu = SwiGLU(dim=128)
    swiglu.eval()
    x = torch.randn(1, 8, 128)
    y1 = swiglu(x)
    y2 = swiglu(x)
    assert torch.allclose(y1, y2)


if __name__ == "__main__":
    test_swiglu_creation()
    test_swiglu_creation_custom_hidden()
    test_swiglu_creation_with_bias()
    test_swiglu_forward()
    test_swiglu_three_weights()
    test_swiglu_deterministic()
    print("\n✓ All SwiGLU tests passed")
