"""Tests for Focal Loss."""

import torch

from foundry.modules.focal_loss import FocalLoss


def test_focal_loss_creation():
    """FocalLoss initializes with defaults."""
    loss_fn = FocalLoss()
    assert loss_fn.alpha == 1.0
    assert loss_fn.gamma == 2.0
    assert loss_fn.ignore_index == -1


def test_focal_loss_creation_custom():
    """FocalLoss accepts custom parameters."""
    loss_fn = FocalLoss(alpha=0.5, gamma=1.5, ignore_index=-100)
    assert loss_fn.alpha == 0.5
    assert loss_fn.gamma == 1.5
    assert loss_fn.ignore_index == -100


def test_focal_loss_forward():
    """FocalLoss computes loss."""
    loss_fn = FocalLoss()
    logits = torch.randn(4, 10)
    targets = torch.randint(0, 10, (4,))
    loss = loss_fn(logits, targets)
    assert loss.ndim == 0
    assert loss.item() >= 0


def test_focal_loss_downweights_easy():
    """FocalLoss gives lower loss for confident correct predictions."""
    loss_fn = FocalLoss(gamma=2.0)
    easy_logits = torch.tensor([[10.0, -10.0]])
    hard_logits = torch.tensor([[0.1, -0.1]])
    targets = torch.tensor([0])

    easy_loss = loss_fn(easy_logits, targets)
    hard_loss = loss_fn(hard_logits, targets)
    assert easy_loss < hard_loss


def test_focal_loss_gamma_zero_is_ce():
    """FocalLoss with gamma=0 approximates cross entropy."""
    focal = FocalLoss(gamma=0.0, alpha=1.0)
    logits = torch.randn(8, 5)
    targets = torch.randint(0, 5, (8,))

    focal_loss = focal(logits, targets)
    ce_loss = torch.nn.functional.cross_entropy(logits, targets)
    assert torch.isclose(focal_loss, ce_loss, atol=1e-5)


def test_focal_loss_ignore_index():
    """FocalLoss respects ignore_index."""
    loss_fn = FocalLoss(ignore_index=-1)
    logits = torch.randn(4, 10)
    targets = torch.tensor([0, 1, -1, 2])
    loss = loss_fn(logits, targets)
    assert loss.item() >= 0


if __name__ == "__main__":
    test_focal_loss_creation()
    test_focal_loss_creation_custom()
    test_focal_loss_forward()
    test_focal_loss_downweights_easy()
    test_focal_loss_gamma_zero_is_ce()
    test_focal_loss_ignore_index()
    print("\n✓ All FocalLoss tests passed")
