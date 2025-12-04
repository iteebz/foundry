"""Tests for sliding window attention."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modules.sliding_window import SlidingWindowMask, sliding_window_mask


def test_sliding_window_mask():
    """Sliding window mask generates correct pattern."""
    mask = sliding_window_mask(seq_len=5, window_size=3)

    expected = torch.tensor(
        [
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
        ],
        dtype=torch.float32,
    )

    assert torch.equal(mask, expected)


def test_sliding_window_module():
    """SlidingWindowMask module works."""
    sw = SlidingWindowMask(window_size=2, max_seq_len=4)
    mask = sw(3)

    assert mask.shape == (3, 3)
    assert mask[0, 0] == 1
    assert mask[1, 0] == 1
    assert mask[1, 1] == 1
    assert mask[2, 0] == 0


def test_sliding_window_different_sizes():
    """Different window sizes produce different masks."""
    mask2 = sliding_window_mask(seq_len=4, window_size=2)
    mask3 = sliding_window_mask(seq_len=4, window_size=3)

    assert not torch.equal(mask2, mask3)
    assert mask2.sum() < mask3.sum()


if __name__ == "__main__":
    test_sliding_window_mask()
    test_sliding_window_module()
    test_sliding_window_different_sizes()
    print("\n✓ All sliding window tests passed")
