"""Tests for Grouped Query Attention."""

import torch

from foundry.modules.gqa import GroupedQueryAttention


def test_gqa_creation():
    """GQA module initializes correctly."""
    gqa = GroupedQueryAttention(n_embd=256, n_head=8, n_kv_head=2)
    assert gqa.n_head == 8
    assert gqa.n_kv_head == 2
    assert gqa.n_rep == 4
    assert gqa.head_dim == 32


def test_gqa_creation_defaults():
    """GQA defaults to n_kv_head=n_head (MHA)."""
    gqa = GroupedQueryAttention(n_embd=256, n_head=8)
    assert gqa.n_kv_head == 8
    assert gqa.n_rep == 1


def test_gqa_forward():
    """GQA forward pass preserves shape."""
    gqa = GroupedQueryAttention(n_embd=256, n_head=8, n_kv_head=2, dropout=0.0)
    gqa.eval()
    x = torch.randn(2, 16, 256)
    y = gqa(x)
    assert y.shape == (2, 16, 256)


def test_gqa_kv_projection_sizes():
    """GQA KV projections use reduced heads."""
    gqa = GroupedQueryAttention(n_embd=256, n_head=8, n_kv_head=2)
    assert gqa.q_proj.out_features == 256
    assert gqa.k_proj.out_features == 64
    assert gqa.v_proj.out_features == 64


def test_gqa_deterministic():
    """GQA produces deterministic output in eval mode."""
    gqa = GroupedQueryAttention(n_embd=128, n_head=4, n_kv_head=2, dropout=0.0)
    gqa.eval()
    x = torch.randn(1, 8, 128)
    y1 = gqa(x)
    y2 = gqa(x)
    assert torch.allclose(y1, y2)


def test_gqa_single_kv_head():
    """GQA works with single KV head (MQA)."""
    gqa = GroupedQueryAttention(n_embd=128, n_head=4, n_kv_head=1, dropout=0.0)
    gqa.eval()
    x = torch.randn(1, 8, 128)
    y = gqa(x)
    assert y.shape == (1, 8, 128)


if __name__ == "__main__":
    test_gqa_creation()
    test_gqa_creation_defaults()
    test_gqa_forward()
    test_gqa_kv_projection_sizes()
    test_gqa_deterministic()
    test_gqa_single_kv_head()
    print("\n✓ All GQA tests passed")
