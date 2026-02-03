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


def test_preference_dataset(tmp_path):
    """PreferenceDataset loads JSONL correctly."""
    import json

    from foundry.data.dataset import PreferenceDataset

    data_path = tmp_path / "prefs.jsonl"
    pairs = [
        {"prompt": "Hello", "chosen": " world", "rejected": " there"},
        {"prompt": "How are", "chosen": " you?", "rejected": " they?"},
    ]
    with data_path.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    class MockTokenizer:
        def encode(self, text):
            return [ord(c) for c in text]

    dataset = PreferenceDataset(data_path, MockTokenizer(), max_length=32)
    assert len(dataset) == 2

    item = dataset[0]
    assert "chosen_ids" in item
    assert "rejected_ids" in item
    assert "prompt_len" in item


def test_collate_preference_batch(tmp_path):
    """Collate function pads correctly."""
    import json

    from foundry.data.dataset import PreferenceDataset, collate_preference_batch

    data_path = tmp_path / "prefs.jsonl"
    pairs = [
        {"prompt": "A", "chosen": "BC", "rejected": "D"},
        {"prompt": "EF", "chosen": "G", "rejected": "HIJK"},
    ]
    with data_path.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    class MockTokenizer:
        def encode(self, text):
            return [ord(c) for c in text]

    dataset = PreferenceDataset(data_path, MockTokenizer(), max_length=32)
    batch = collate_preference_batch([dataset[0], dataset[1]])

    assert batch["chosen_ids"].shape[0] == 2
    assert batch["rejected_ids"].shape[0] == 2
    assert batch["chosen_mask"].shape == batch["chosen_ids"].shape
    assert batch["prompt_lens"].shape == (2,)
