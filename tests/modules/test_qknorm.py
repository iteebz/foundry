"""Tests for QK Normalization."""

import torch

from foundry.modules.qknorm import QKNorm


def test_qknorm_creation():
    """QKNorm initializes correctly."""
    qknorm = QKNorm(dim=64)
    assert qknorm.query_norm.normalized_shape == (64,)
    assert qknorm.key_norm.normalized_shape == (64,)


def test_qknorm_forward():
    """QKNorm forward pass preserves shapes."""
    qknorm = QKNorm(dim=64)
    q = torch.randn(2, 8, 16, 64)
    k = torch.randn(2, 8, 16, 64)
    q_norm, k_norm = qknorm(q, k)
    assert q_norm.shape == (2, 8, 16, 64)
    assert k_norm.shape == (2, 8, 16, 64)


def test_qknorm_normalizes():
    """QKNorm produces normalized outputs."""
    qknorm = QKNorm(dim=128)
    q = torch.randn(1, 4, 8, 128) * 10 + 5
    k = torch.randn(1, 4, 8, 128) * 10 + 5
    q_norm, k_norm = qknorm(q, k)
    assert q_norm.mean().abs() < 0.1
    assert k_norm.mean().abs() < 0.1


def test_qknorm_deterministic():
    """QKNorm is deterministic."""
    qknorm = QKNorm(dim=64)
    q = torch.randn(1, 2, 4, 64)
    k = torch.randn(1, 2, 4, 64)
    q1, k1 = qknorm(q, k)
    q2, k2 = qknorm(q, k)
    assert torch.allclose(q1, q2)
    assert torch.allclose(k1, k2)


def test_qknorm_independent():
    """QKNorm applies independent norms to Q and K."""
    qknorm = QKNorm(dim=32)
    assert qknorm.query_norm is not qknorm.key_norm


if __name__ == "__main__":
    test_qknorm_creation()
    test_qknorm_forward()
    test_qknorm_normalizes()
    test_qknorm_deterministic()
    test_qknorm_independent()
    print("\n✓ All QKNorm tests passed")
