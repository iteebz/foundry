"""Sparse attention - attend to strided positions (local + global)."""

import torch
import torch.nn as nn


def sparse_attention_mask(
    seq_len: int,
    block_size: int = 64,
    stride: int = 64,
    device='cpu'
) -> torch.Tensor:
    """Generate sparse attention mask.
    
    Pattern: Local attention (block_size) + strided global attention (every stride)
    
    Args:
        seq_len: Sequence length
        block_size: Size of local attention block
        stride: Stride for global attention
        device: Device to create mask on
    
    Returns:
        Attention mask [seq_len, seq_len]
    """
    mask = torch.zeros(seq_len, seq_len, device=device)
    
    for i in range(seq_len):
        start_local = max(0, i - block_size + 1)
        mask[i, start_local:i+1] = 1
        
        strided_positions = list(range(0, i+1, stride))
        mask[i, strided_positions] = 1
    
    return mask


class SparseAttentionMask(nn.Module):
    """Sparse attention mask generator."""
    
    def __init__(
        self,
        block_size: int = 64,
        stride: int = 64,
        max_seq_len: int = 2048
    ):
        super().__init__()
        self.block_size = block_size
        self.stride = stride
        self.max_seq_len = max_seq_len
        
        self.register_buffer(
            "mask",
            sparse_attention_mask(max_seq_len, block_size, stride)
        )
    
    def forward(self, seq_len: int) -> torch.Tensor:
        """Get sparse attention mask for given sequence length."""
        return self.mask[:seq_len, :seq_len]
