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
        expert_hidden_dim: int | None = None,
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
        num_tokens = x_flat.size(0)

        router_logits = self.router(x_flat)
        routing_weights = F.softmax(router_logits, dim=-1)

        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        top_k_weights_flat = top_k_weights.view(-1)
        top_k_indices_flat = top_k_indices.view(-1)

        token_indices = torch.arange(num_tokens, device=x.device).repeat_interleave(self.top_k)

        sort_indices = torch.argsort(top_k_indices_flat, stable=True)
        sorted_token_indices = token_indices[sort_indices]
        sorted_expert_indices = top_k_indices_flat[sort_indices]
        sorted_weights = top_k_weights_flat[sort_indices]

        output = torch.zeros_like(x_flat)

        expert_boundaries = torch.where(
            torch.cat(
                [
                    torch.tensor([True], device=x.device),
                    sorted_expert_indices[1:] != sorted_expert_indices[:-1],
                ]
            )
        )[0]
        expert_boundaries = torch.cat(
            [expert_boundaries, torch.tensor([len(sorted_expert_indices)], device=x.device)]
        )

        for i in range(len(expert_boundaries) - 1):
            start_idx = expert_boundaries[i]
            end_idx = expert_boundaries[i + 1]

            if start_idx >= end_idx:
                continue

            expert_id = sorted_expert_indices[start_idx].item()
            batch_tokens = sorted_token_indices[start_idx:end_idx]
            batch_weights = sorted_weights[start_idx:end_idx]

            expert_input = x_flat[batch_tokens]
            expert_output = self.experts[expert_id](expert_input)

            output.index_add_(0, batch_tokens, expert_output * batch_weights.unsqueeze(-1))

        return output.view(B, T, C)
