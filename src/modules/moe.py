"""Mixture of Experts (MoE) layer.

Routes tokens to subset of expert MLPs for efficient scaling.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class MoELayer(nn.Module):
    """Mixture of Experts with top-k routing."""

    def __init__(
        self,
        n_embd: int,
        n_experts: int = 8,
        top_k: int = 2,
        expert_hidden_dim: int = None,
        bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_embd = n_embd
        self.n_experts = n_experts
        self.top_k = top_k
        expert_hidden_dim = expert_hidden_dim or (4 * n_embd)

        self.router = nn.Linear(n_embd, n_experts, bias=False)

        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(n_embd, expert_hidden_dim, bias=bias),
                    nn.GELU(),
                    nn.Linear(expert_hidden_dim, n_embd, bias=bias),
                    nn.Dropout(dropout),
                )
                for _ in range(n_experts)
            ]
        )

    def forward(self, x):
        B, T, C = x.size()

        x_flat = x.view(-1, C)

        router_logits = self.router(x_flat)
        routing_weights = F.softmax(router_logits, dim=-1)

        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        output = torch.zeros_like(x_flat)

        for i in range(self.n_experts):
            expert_mask = (top_k_indices == i).any(dim=-1)
            if expert_mask.any():
                expert_input = x_flat[expert_mask]
                expert_output = self.experts[i](expert_input)

                expert_weight_mask = top_k_indices == i
                weights = top_k_weights[expert_weight_mask].unsqueeze(-1)

                output[expert_mask] += expert_output * weights

        return output.view(B, T, C)
