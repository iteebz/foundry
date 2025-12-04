"""Tests for Multi-Latent Attention."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modules.mla import MultiLatentAttention


def test_mla_creation():
    """MLA module initializes correctly."""
    mla = MultiLatentAttention(n_embd=384, n_head=6, latent_dim=192)
    assert mla.n_embd == 384
    assert mla.n_head == 6
    assert mla.latent_dim == 192
    assert mla.head_dim == 64


def test_mla_forward():
    """MLA forward pass works."""
    mla = MultiLatentAttention(n_embd=384, n_head=6, latent_dim=192)
    x = torch.randn(2, 16, 384)
    y = mla(x)

    assert y.shape == (2, 16, 384)


def test_mla_default_latent_dim():
    """MLA uses n_embd//2 as default latent dim."""
    mla = MultiLatentAttention(n_embd=384, n_head=6)
    assert mla.latent_dim == 192


def test_mla_output_deterministic():
    """MLA produces deterministic output in eval mode."""
    torch.manual_seed(42)
    mla = MultiLatentAttention(n_embd=128, n_head=4, dropout=0.0)
    mla.eval()

    x = torch.randn(1, 8, 128)
    y1 = mla(x)
    y2 = mla(x)

    assert torch.allclose(y1, y2)


def test_mla_compression():
    """MLA compresses KV to smaller dimension."""
    mla = MultiLatentAttention(n_embd=384, n_head=6, latent_dim=192)

    assert mla.c_down.out_features == 192
    assert mla.kv_up.out_features == 2 * 64


if __name__ == "__main__":
    test_mla_creation()
    test_mla_forward()
    test_mla_default_latent_dim()
    test_mla_output_deterministic()
    test_mla_compression()
    print("\n✓ All MLA tests passed")
