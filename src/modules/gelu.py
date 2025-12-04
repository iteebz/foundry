"""GELU activation variants."""

import torch.nn as nn
from torch.nn import functional as F


class GELU(nn.Module):
    """Standard GELU activation."""

    def __init__(self, n_embd, bias=False):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.proj = nn.Linear(4 * n_embd, n_embd, bias=bias)

    def forward(self, x):
        x = self.fc(x)
        x = F.gelu(x)
        return self.proj(x)
