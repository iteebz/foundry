"""Tests for RMSNorm."""

import torch

from foundry.modules.rmsnorm import RMSNorm


def test_rmsnorm_creation():
    """RMSNorm initializes correctly."""
    rms = RMSNorm(dim=256)
    assert rms.weight.shape == (256,)
    assert rms.eps == 1e-6


def test_rmsnorm_creation_custom_eps():
    """RMSNorm accepts custom epsilon."""
    rms = RMSNorm(dim=128, eps=1e-5)
    assert rms.eps == 1e-5


def test_rmsnorm_forward():
    """RMSNorm forward pass preserves shape."""
    rms = RMSNorm(dim=128)
    x = torch.randn(2, 16, 128)
    y = rms(x)
    assert y.shape == (2, 16, 128)


def test_rmsnorm_scales():
    """RMSNorm scales to roughly unit RMS."""
    rms = RMSNorm(dim=256)
    x = torch.randn(4, 32, 256) * 10
    y = rms(x)
    rms_val = y.pow(2).mean(-1).sqrt()
    assert (rms_val.mean() - 1.0).abs() < 0.2


def test_rmsnorm_deterministic():
    """RMSNorm is deterministic."""
    rms = RMSNorm(dim=64)
    x = torch.randn(1, 8, 64)
    y1 = rms(x)
    y2 = rms(x)
    assert torch.allclose(y1, y2)


def test_rmsnorm_no_bias():
    """RMSNorm has no bias parameter (unlike LayerNorm)."""
    rms = RMSNorm(dim=128)
    params = list(rms.parameters())
    assert len(params) == 1
    assert params[0].shape == (128,)


if __name__ == "__main__":
    test_rmsnorm_creation()
    test_rmsnorm_creation_custom_eps()
    test_rmsnorm_forward()
    test_rmsnorm_scales()
    test_rmsnorm_deterministic()
    test_rmsnorm_no_bias()
    print("\n✓ All RMSNorm tests passed")
