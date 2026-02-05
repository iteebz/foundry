"""Tests for Rotary Position Embedding."""

import torch

from foundry.modules.rope import RotaryEmbedding, apply_rotary_emb, rotate_half


def test_rope_creation():
    """RotaryEmbedding initializes correctly."""
    rope = RotaryEmbedding(dim=64, max_seq_len=2048)
    assert rope.dim == 64
    assert rope.max_seq_len == 2048
    assert rope.inv_freq.shape == (32,)


def test_rope_creation_custom_base():
    """RotaryEmbedding accepts custom base."""
    rope = RotaryEmbedding(dim=64, base=500000.0)
    assert rope.base == 500000.0


def test_rope_forward():
    """RotaryEmbedding returns cos and sin of correct shape."""
    rope = RotaryEmbedding(dim=64, max_seq_len=1024)
    x = torch.randn(1, 8, 32, 64)
    cos, sin = rope(x, seq_len=32)
    assert cos.shape == (32, 64)
    assert sin.shape == (32, 64)


def test_rope_caching():
    """RotaryEmbedding caches cos/sin for same seq_len."""
    rope = RotaryEmbedding(dim=64)
    x = torch.randn(1, 8, 16, 64)
    cos1, sin1 = rope(x, seq_len=16)
    cos2, sin2 = rope(x, seq_len=16)
    assert torch.allclose(cos1, cos2)
    assert torch.allclose(sin1, sin2)
    assert rope._seq_len_cached == 16


def test_rotate_half():
    """rotate_half rotates correctly."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    rotated = rotate_half(x)
    expected = torch.tensor([-3.0, -4.0, 1.0, 2.0])
    assert torch.allclose(rotated, expected)


def test_apply_rotary_emb():
    """apply_rotary_emb preserves shape."""
    q = torch.randn(2, 8, 16, 64)
    k = torch.randn(2, 8, 16, 64)
    cos = torch.randn(16, 64)
    sin = torch.randn(16, 64)
    q_rot, k_rot = apply_rotary_emb(q, k, cos, sin)
    assert q_rot.shape == (2, 8, 16, 64)
    assert k_rot.shape == (2, 8, 16, 64)


def test_rope_deterministic():
    """RotaryEmbedding is deterministic."""
    rope = RotaryEmbedding(dim=64)
    x = torch.randn(1, 4, 8, 64)
    cos1, sin1 = rope(x, seq_len=8)
    cos2, sin2 = rope(x, seq_len=8)
    assert torch.allclose(cos1, cos2)
    assert torch.allclose(sin1, sin2)


if __name__ == "__main__":
    test_rope_creation()
    test_rope_creation_custom_base()
    test_rope_forward()
    test_rope_caching()
    test_rotate_half()
    test_apply_rotary_emb()
    test_rope_deterministic()
    print("\n✓ All RoPE tests passed")
