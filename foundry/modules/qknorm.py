"""QK Normalization for attention stability."""

import torch.nn as nn


class QKNorm(nn.Module):
    """Normalize queries and keys before attention."""

    def __init__(self, dim):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.key_norm = nn.LayerNorm(dim)

    def forward(self, q, k):
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q, k
