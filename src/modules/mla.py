"""Multi-Latent Attention (MLA) - DeepSeek innovation.

Compresses Q/K/V into shared latent space, then decompresses.
Reduces KV cache size while maintaining model capacity.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class MultiLatentAttention(nn.Module):
    """Multi-Latent Attention with shared compression."""
    
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        latent_dim: int = None,
        bias: bool = False,
        dropout: float = 0.0,
        block_size: int = 1024,
    ):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.latent_dim = latent_dim or (n_embd // 2)
        self.dropout = dropout
        
        self.c_down = nn.Linear(n_embd, self.latent_dim, bias=bias)
        
        self.q_up = nn.Linear(self.latent_dim, n_embd, bias=bias)
        self.kv_up = nn.Linear(self.latent_dim, 2 * self.head_dim, bias=bias)
        
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )
    
    def forward(self, x):
        B, T, C = x.size()
        
        latent = self.c_down(x)
        
        q = self.q_up(latent).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        kv = self.kv_up(latent)
        k, v = kv.split(self.head_dim, dim=-1)
        k = k.unsqueeze(1).expand(B, self.n_head, T, self.head_dim)
        v = v.unsqueeze(1).expand(B, self.n_head, T, self.head_dim)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        y = self.resid_dropout(self.c_proj(y))
        return y
