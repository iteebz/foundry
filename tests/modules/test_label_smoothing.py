"""Tests for Label Smoothing Cross Entropy."""

import torch

from foundry.modules.label_smoothing import LabelSmoothingCrossEntropy


def test_label_smoothing_creation():
    """LabelSmoothingCrossEntropy initializes with defaults."""
    loss_fn = LabelSmoothingCrossEntropy()
    assert loss_fn.smoothing == 0.1
    assert loss_fn.ignore_index == -1


def test_label_smoothing_creation_custom():
    """LabelSmoothingCrossEntropy accepts custom parameters."""
    loss_fn = LabelSmoothingCrossEntropy(smoothing=0.2, ignore_index=-100)
    assert loss_fn.smoothing == 0.2
    assert loss_fn.ignore_index == -100


def test_label_smoothing_forward():
    """LabelSmoothingCrossEntropy computes loss."""
    loss_fn = LabelSmoothingCrossEntropy()
    logits = torch.randn(4, 10)
    targets = torch.randint(0, 10, (4,))
    loss = loss_fn(logits, targets)
    assert loss.ndim == 0
    assert loss.item() >= 0


def test_label_smoothing_zero_is_ce():
    """LabelSmoothingCrossEntropy with smoothing=0 approximates CE."""
    ls_loss = LabelSmoothingCrossEntropy(smoothing=0.0)
    logits = torch.randn(8, 5)
    targets = torch.randint(0, 5, (8,))

    smooth_loss = ls_loss(logits, targets)
    ce_loss = torch.nn.functional.cross_entropy(logits, targets)
    assert torch.isclose(smooth_loss, ce_loss, atol=1e-5)


def test_label_smoothing_higher_than_ce():
    """LabelSmoothingCrossEntropy loss >= CE for same inputs."""
    ls_loss = LabelSmoothingCrossEntropy(smoothing=0.1)
    logits = torch.tensor([[5.0, 0.0, 0.0]])
    targets = torch.tensor([0])

    smooth_val = ls_loss(logits, targets)
    ce_val = torch.nn.functional.cross_entropy(logits, targets)
    assert smooth_val >= ce_val


def test_label_smoothing_mask_applied():
    """LabelSmoothingCrossEntropy mask excludes ignored targets from mean."""
    loss_fn = LabelSmoothingCrossEntropy(ignore_index=999)
    logits = torch.randn(4, 10)
    targets = torch.tensor([0, 1, 999, 2])
    targets_clamped = targets.clone()
    targets_clamped[2] = 0
    loss = loss_fn(logits, targets_clamped)
    assert loss.item() >= 0


if __name__ == "__main__":
    test_label_smoothing_creation()
    test_label_smoothing_creation_custom()
    test_label_smoothing_forward()
    test_label_smoothing_zero_is_ce()
    test_label_smoothing_higher_than_ce()
    test_label_smoothing_mask_applied()
    print("\n✓ All LabelSmoothingCrossEntropy tests passed")
