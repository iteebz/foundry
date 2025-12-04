"""GLU activation variant."""

import torch
import torch.nn as nn
from torch.nn import functional as F


class GLU(nn.Module):
    """Gated Linear Unit."""

    def __init__(self, n_embd, bias=False):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gate = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.proj = nn.Linear(4 * n_embd, n_embd, bias=bias)

    def forward(self, x):
        return self.proj(self.fc(x) * torch.sigmoid(self.gate(x)))
