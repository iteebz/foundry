"""Tests for ALiBi position encoding."""

import torch

from foundry.modules.alibi import ALiBi, get_alibi_slopes


def test_alibi_slopes_power_of_2():
    """ALiBi slopes correct for power-of-2 heads."""
    slopes = get_alibi_slopes(8)
    assert slopes.shape == (8,)
    assert (slopes > 0).all()
    assert (slopes <= 1).all()


def test_alibi_slopes_non_power_of_2():
    """ALiBi slopes handles non-power-of-2 heads."""
    slopes = get_alibi_slopes(6)
    assert slopes.shape == (6,)


def test_alibi_creation():
    """ALiBi module initializes correctly."""
    alibi = ALiBi(n_heads=8, max_seq_len=1024)
    assert alibi.n_heads == 8
    assert alibi.max_seq_len == 1024
    assert alibi.slopes.shape == (8, 1, 1)


def test_alibi_forward():
    """ALiBi forward returns correct shape."""
    alibi = ALiBi(n_heads=8, max_seq_len=1024)
    bias = alibi(seq_len=64)
    assert bias.shape == (1, 8, 64, 64)


def test_alibi_bias_causal():
    """ALiBi bias is negative (penalizes distance)."""
    alibi = ALiBi(n_heads=4, max_seq_len=128)
    bias = alibi(seq_len=16)
    assert (bias <= 0).all()


def test_alibi_bias_diagonal_zero():
    """ALiBi diagonal (self-attention) is zero."""
    alibi = ALiBi(n_heads=4, max_seq_len=128)
    bias = alibi(seq_len=8)
    diag = torch.diagonal(bias[0, 0])
    assert torch.allclose(diag, torch.zeros_like(diag))


if __name__ == "__main__":
    test_alibi_slopes_power_of_2()
    test_alibi_slopes_non_power_of_2()
    test_alibi_creation()
    test_alibi_forward()
    test_alibi_bias_causal()
    test_alibi_bias_diagonal_zero()
    print("\n✓ All ALiBi tests passed")
