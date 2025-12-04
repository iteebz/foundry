"""Tests for sparse attention."""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modules.sparse_attention import sparse_attention_mask, SparseAttentionMask


def test_sparse_attention_mask():
    """Sparse attention mask generates local + strided pattern."""
    mask = sparse_attention_mask(seq_len=8, block_size=2, stride=4)
    
    assert mask[0, 0] == 1
    assert mask[1, 0] == 1
    assert mask[1, 1] == 1
    
    assert mask[5, 4] == 1
    assert mask[5, 5] == 1
    assert mask[5, 0] == 1
    assert mask[5, 4] == 1


def test_sparse_attention_module():
    """SparseAttentionMask module works."""
    sparse = SparseAttentionMask(block_size=2, stride=3, max_seq_len=6)
    mask = sparse(4)
    
    assert mask.shape == (4, 4)
    assert mask[0, 0] == 1
    assert mask[3, 2] == 1
    assert mask[3, 3] == 1
    assert mask[3, 0] == 1


def test_sparse_attention_vs_full():
    """Sparse attention has fewer connections than full attention."""
    sparse_mask = sparse_attention_mask(seq_len=128, block_size=16, stride=32)
    full_mask = torch.tril(torch.ones(128, 128))
    
    assert sparse_mask.sum() < full_mask.sum()


if __name__ == "__main__":
    test_sparse_attention_mask()
    test_sparse_attention_module()
    test_sparse_attention_vs_full()
    print("\n✓ All sparse attention tests passed")
