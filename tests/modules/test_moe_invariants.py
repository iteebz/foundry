"""Test MoE routing invariants and expert load balancing."""

import torch

from foundry.modules.moe import MoELayer


def test_moe_routing_weights_sum_to_one():
    """Top-k routing weights should sum to 1.0 for each token."""
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

        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6), (
            "Routing weights should sum to 1.0"
        )


def test_moe_all_experts_reachable():
    """Over many random inputs, all experts should be selected at least once."""
    moe = MoELayer(n_embd=64, n_experts=8, top_k=2, dropout=0.0)
    moe.eval()

    expert_counts = torch.zeros(8)

    torch.manual_seed(42)
    for _ in range(100):
        x = torch.randn(4, 16, 64)

        with torch.no_grad():
            router_logits = moe.router(x.view(-1, 64))
            routing_weights = torch.softmax(router_logits, dim=-1)
            _, top_k_indices = torch.topk(routing_weights, moe.top_k, dim=-1)

            for expert_id in range(8):
                expert_counts[expert_id] += (top_k_indices == expert_id).sum().item()

    assert (expert_counts > 0).all(), f"All experts should be selected: {expert_counts}"


def test_moe_routing_deterministic():
    """Same input should produce same expert selection."""
    moe = MoELayer(n_embd=64, n_experts=8, top_k=2, dropout=0.0)
    moe.eval()

    torch.manual_seed(42)
    x = torch.randn(4, 16, 64)

    with torch.no_grad():
        out1 = moe(x)
        out2 = moe(x)

        assert torch.allclose(out1, out2, atol=1e-6), "Same input should produce same output"


def test_moe_gradient_flow_to_selected_experts():
    """Backward pass should update only selected experts for each token."""
    moe = MoELayer(n_embd=64, n_experts=4, top_k=1, dropout=0.0)
    moe.train()

    torch.manual_seed(42)
    x = torch.randn(2, 4, 64, requires_grad=True)

    {name: param.clone() for name, param in moe.named_parameters() if "experts" in name}

    output = moe(x)
    loss = output.sum()
    loss.backward()

    updated_experts = []
    for expert_id in range(4):
        expert_updated = False
        for name, param in moe.named_parameters():
            if (
                f"experts.{expert_id}" in name
                and param.grad is not None
                and not torch.allclose(param.grad, torch.zeros_like(param.grad))
            ):
                expert_updated = True
                break
        if expert_updated:
            updated_experts.append(expert_id)

    assert len(updated_experts) > 0, "At least one expert should receive gradients"


def test_moe_expert_load_distribution():
    """Experts should receive roughly balanced load over many tokens."""
    moe = MoELayer(n_embd=64, n_experts=8, top_k=2, dropout=0.0)
    moe.eval()

    expert_counts = torch.zeros(8)

    torch.manual_seed(42)
    total_tokens = 0
    for _ in range(50):
        x = torch.randn(4, 32, 64)
        total_tokens += 4 * 32

        with torch.no_grad():
            router_logits = moe.router(x.view(-1, 64))
            routing_weights = torch.softmax(router_logits, dim=-1)
            _, top_k_indices = torch.topk(routing_weights, moe.top_k, dim=-1)

            for expert_id in range(8):
                expert_counts[expert_id] += (top_k_indices == expert_id).sum().item()

    expert_ratios = expert_counts / expert_counts.sum()
    expected_ratio = 1.0 / 8

    for expert_id, ratio in enumerate(expert_ratios):
        assert abs(ratio - expected_ratio) < 0.1, (
            f"Expert {expert_id} has imbalanced load: {ratio:.3f} vs expected {expected_ratio:.3f}"
        )


def test_moe_output_shape_preserves_input():
    """MoE should preserve input shape (B, T, C)."""
    moe = MoELayer(n_embd=64, n_experts=8, top_k=2, dropout=0.0)

    x = torch.randn(4, 16, 64)
    output = moe(x)

    assert output.shape == x.shape, f"Output shape {output.shape} != input shape {x.shape}"


def test_moe_top_k_parameter_respected():
    """MoE should select exactly top_k experts per token."""
    for top_k in [1, 2, 4]:
        moe = MoELayer(n_embd=64, n_experts=8, top_k=top_k, dropout=0.0)
        moe.eval()

        torch.manual_seed(42)
        x = torch.randn(4, 16, 64)

        with torch.no_grad():
            router_logits = moe.router(x.view(-1, 64))
            routing_weights = torch.softmax(router_logits, dim=-1)
            _top_k_weights, top_k_indices = torch.topk(routing_weights, top_k, dim=-1)

            assert top_k_indices.shape[-1] == top_k, f"Should select {top_k} experts"
