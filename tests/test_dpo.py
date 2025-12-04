"""Tests for DPO loss."""

import torch

from foundry.modules.dpo_loss import DPOLoss, compute_log_probs


def test_dpo_loss_basic():
    """DPO loss computes correctly."""
    dpo = DPOLoss(beta=0.1)

    policy_chosen = torch.tensor([1.0, 2.0])
    policy_rejected = torch.tensor([0.5, 1.5])
    reference_chosen = torch.tensor([0.9, 1.8])
    reference_rejected = torch.tensor([0.6, 1.4])

    loss = dpo(policy_chosen, policy_rejected, reference_chosen, reference_rejected)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss.item() >= 0.0


def test_dpo_loss_with_label_smoothing():
    """DPO loss with label smoothing."""
    dpo = DPOLoss(beta=0.1, label_smoothing=0.1)

    policy_chosen = torch.tensor([1.0])
    policy_rejected = torch.tensor([0.5])
    reference_chosen = torch.tensor([0.9])
    reference_rejected = torch.tensor([0.6])

    loss = dpo(policy_chosen, policy_rejected, reference_chosen, reference_rejected)

    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0.0


def test_compute_log_probs():
    """Compute log probabilities for labels."""
    batch_size = 2
    seq_len = 5
    vocab_size = 100

    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    log_probs = compute_log_probs(logits, labels)

    assert log_probs.shape == (batch_size,)
    assert torch.all(log_probs <= 0.0)


def test_dpo_loss_prefers_chosen():
    """DPO loss is lower when policy prefers chosen over rejected."""
    dpo = DPOLoss(beta=0.1)

    policy_chosen_high = torch.tensor([2.0])
    policy_rejected_low = torch.tensor([0.5])
    reference_chosen = torch.tensor([1.0])
    reference_rejected = torch.tensor([1.0])

    loss_high = dpo(policy_chosen_high, policy_rejected_low, reference_chosen, reference_rejected)

    policy_chosen_low = torch.tensor([0.5])
    policy_rejected_high = torch.tensor([2.0])

    loss_low = dpo(policy_chosen_low, policy_rejected_high, reference_chosen, reference_rejected)

    assert loss_high < loss_low


if __name__ == "__main__":
    test_dpo_loss_basic()
    test_dpo_loss_with_label_smoothing()
    test_compute_log_probs()
    test_dpo_loss_prefers_chosen()
    print("\n✓ All DPO tests passed")
