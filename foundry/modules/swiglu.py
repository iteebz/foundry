"""SwiGLU MLP from Llama."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU: Swish-Gated Linear Unit MLP."""

    def __init__(self, dim: int, hidden_dim: int | None = None, bias: bool = False):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)

    def forward(self, x: torch.Tensor):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
