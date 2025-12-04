"""Tests for DPO mutations."""

import tempfile

import yaml

from foundry.mutate import mutate_dpo, mutate_loss, save_mutation


def test_mutate_loss_dpo():
    """DPO loss mutation."""
    config = mutate_loss("dpo")

    assert config["name"] == "loss_dpo"
    assert config["model_args"]["loss_type"] == "dpo"


def test_mutate_dpo_default():
    """DPO mutation with default params."""
    config = mutate_dpo()

    assert config["name"] == "dpo_beta0.1"
    assert config["model_args"]["loss_type"] == "dpo"
    assert config["model_args"]["dpo_beta"] == 0.1
    assert config["model_args"]["dpo_label_smoothing"] == 0.0


def test_mutate_dpo_custom():
    """DPO mutation with custom params."""
    config = mutate_dpo(beta=0.5, label_smoothing=0.1)

    assert config["name"] == "dpo_beta0.5_smooth0.1"
    assert config["model_args"]["dpo_beta"] == 0.5
    assert config["model_args"]["dpo_label_smoothing"] == 0.1


def test_save_dpo_mutation():
    """DPO mutation saves to valid YAML."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = mutate_dpo(beta=0.2)
        path = save_mutation(config, tmpdir)

        assert path.exists()

        with open(path) as f:
            loaded = yaml.safe_load(f)

        assert loaded["name"] == config["name"]
        assert loaded["model_args"]["loss_type"] == "dpo"


if __name__ == "__main__":
    test_mutate_loss_dpo()
    test_mutate_dpo_default()
    test_mutate_dpo_custom()
    test_save_dpo_mutation()
    print("\n✓ All DPO mutation tests passed")
