"""Grouped Query Attention from Llama."""

import torch
import torch.nn as nn
from torch.nn import functional as F


class GroupedQueryAttention(nn.Module):
    """GQA: num_kv_heads < num_heads, with kv shared across groups."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_kv_head: int | None = None,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert n_embd % n_head == 0
        if n_kv_head is None:
            n_kv_head = n_head
        assert n_head % n_kv_head == 0

        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.n_rep = n_head // n_kv_head

        self.q_proj = nn.Linear(n_embd, n_head * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(n_embd, n_embd, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        B, T, C = x.size()

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        result = self.resid_dropout(self.o_proj(y))
        assert result.shape == x.shape, f"GQA shape contract violated: {result.shape} != {x.shape}"
        return result
