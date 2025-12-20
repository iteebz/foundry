"""Rotary Position Embedding from Llama."""

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """RoPE: Rotary Position Embedding."""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if seq_len != self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos().to(dtype)
            self._sin_cached = emb.sin().to(dtype)

    def forward(self, x: torch.Tensor, seq_len: int | None = None):
        if seq_len is None:
            seq_len = x.shape[-2]
        self._update_cache(seq_len, x.device, x.dtype)
        return self._cos_cached[:seq_len], self._sin_cached[:seq_len]


def apply_rotary_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotary embeddings to query and key tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x: torch.Tensor):
    """Rotate half the hidden dims of the input."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
