"""Sliding window attention - local attention with fixed window size."""

import torch
import torch.nn as nn


def sliding_window_mask(seq_len: int, window_size: int, device="cpu") -> torch.Tensor:
    """Generate sliding window attention mask.

    Args:
        seq_len: Sequence length
        window_size: Size of attention window
        device: Device to create mask on

    Returns:
        Attention mask [seq_len, seq_len] with 1s in sliding window, 0s elsewhere
    """
    mask = torch.zeros(seq_len, seq_len, device=device)

    for i in range(seq_len):
        start = max(0, i - window_size + 1)
        mask[i, start : i + 1] = 1

    return mask


class SlidingWindowMask(nn.Module):
    """Sliding window attention mask generator."""

    def __init__(self, window_size: int, max_seq_len: int = 2048):
        super().__init__()
        self.window_size = window_size
        self.max_seq_len = max_seq_len

        self.register_buffer("mask", sliding_window_mask(max_seq_len, window_size))

    def forward(self, seq_len: int) -> torch.Tensor:
        """Get sliding window mask for given sequence length."""
        return self.mask[:seq_len, :seq_len]
