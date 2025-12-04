"""Standard LayerNorm implementation."""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """LayerNorm with optional bias."""

    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return nn.functional.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)
