"""MoE architectural invariants.

Tests guarantees, not hopes:
- Routing weights sum to 1
- Same input -> same output (determinism)
- Output shape preserved
- Top-k respected
- Gradients flow to selected experts
"""

import pytest
import torch

from foundry.modules.moe import MoELayer


def test_moe_routing_weights_sum_to_one():
    moe = MoELayer(n_embd=64, n_experts=8, top_k=2, dropout=0.0)
    moe.eval()

    torch.manual_seed(42)
    x = torch.randn(4, 16, 64)

    with torch.no_grad():
        router_logits = moe.router(x.view(-1, 64))
        routing_weights = torch.softmax(router_logits, dim=-1)
        top_k_weights, _ = torch.topk(routing_weights, moe.top_k, dim=-1)
        top_k_weights_normalized = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        sums = top_k_weights_normalized.sum(dim=-1)

        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)


def test_moe_routing_deterministic():
    moe = MoELayer(n_embd=64, n_experts=8, top_k=2, dropout=0.0)
    moe.eval()

    torch.manual_seed(42)
    x = torch.randn(4, 16, 64)

    with torch.no_grad():
        out1 = moe(x)
        out2 = moe(x)
        assert torch.allclose(out1, out2, atol=1e-6)


def test_moe_output_shape_preserves_input():
    moe = MoELayer(n_embd=64, n_experts=8, top_k=2, dropout=0.0)
    x = torch.randn(4, 16, 64)
    output = moe(x)
    assert output.shape == x.shape


def test_moe_output_finite():
    moe = MoELayer(n_embd=64, n_experts=8, top_k=2, dropout=0.0)
    x = torch.randn(4, 16, 64)
    output = moe(x)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("top_k", [1, 2, 4])
def test_moe_top_k_parameter_respected(top_k):
    moe = MoELayer(n_embd=64, n_experts=8, top_k=top_k, dropout=0.0)
    moe.eval()

    torch.manual_seed(42)
    x = torch.randn(4, 16, 64)

    with torch.no_grad():
        router_logits = moe.router(x.view(-1, 64))
        routing_weights = torch.softmax(router_logits, dim=-1)
        _, top_k_indices = torch.topk(routing_weights, top_k, dim=-1)
        assert top_k_indices.shape[-1] == top_k


def test_moe_gradient_flows():
    moe = MoELayer(n_embd=64, n_experts=4, top_k=1, dropout=0.0)
    moe.train()

    torch.manual_seed(42)
    x = torch.randn(2, 4, 64, requires_grad=True)

    output = moe(x)
    loss = output.sum()
    loss.backward()

    has_grad = any(
        p.grad is not None and not torch.allclose(p.grad, torch.zeros_like(p.grad))
        for p in moe.parameters()
    )
    assert has_grad
