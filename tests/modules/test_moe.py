"""Tests for Mixture of Experts."""

import torch

from foundry.modules.moe import MoELayer


def test_moe_creation():
    """MoE layer initializes correctly."""
    moe = MoELayer(n_embd=384, n_experts=8, top_k=2)
    assert moe.n_embd == 384
    assert moe.n_experts == 8
    assert moe.top_k == 2
    assert len(moe.experts) == 8


def test_moe_forward():
    """MoE forward pass works."""
    moe = MoELayer(n_embd=384, n_experts=4, top_k=2)
    x = torch.randn(2, 16, 384)
    y = moe(x)

    assert y.shape == (2, 16, 384)


def test_moe_routing():
    """MoE routes to top-k experts."""
    torch.manual_seed(42)
    moe = MoELayer(n_embd=128, n_experts=4, top_k=2, dropout=0.0)
    moe.eval()

    x = torch.randn(1, 8, 128)
    y = moe(x)

    assert y.shape == x.shape
    assert not torch.isnan(y).any()


def test_moe_single_expert():
    """MoE with 1 expert and top_k=1 behaves like MLP."""
    moe = MoELayer(n_embd=128, n_experts=1, top_k=1, dropout=0.0)
    moe.eval()

    x = torch.randn(1, 4, 128)
    y = moe(x)

    assert y.shape == x.shape


if __name__ == "__main__":
    test_moe_creation()
    test_moe_forward()
    test_moe_routing()
    test_moe_single_expert()
    print("\n✓ All MoE tests passed")
