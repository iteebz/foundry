"""Tests for loss function mutations."""

import tempfile

import yaml

from foundry.mutate import mutate_loss, save_mutation


def test_mutate_loss_focal():
    """Focal loss mutation generates valid config."""
    config = mutate_loss("focal")
    assert config["name"] == "loss_focal"
    assert config["model_args"]["loss_type"] == "focal"


def test_mutate_loss_label_smoothing():
    """Label smoothing mutation generates valid config."""
    config = mutate_loss("label_smoothing")
    assert config["name"] == "loss_label_smoothing"
    assert config["model_args"]["loss_type"] == "label_smoothing"


def test_mutate_loss_cross_entropy():
    """Cross entropy mutation generates valid config."""
    config = mutate_loss("cross_entropy")
    assert config["name"] == "loss_cross_entropy"
    assert config["model_args"]["loss_type"] == "cross_entropy"


def test_mutate_loss_invalid():
    """Invalid loss types raise errors."""
    try:
        mutate_loss("invalid_loss")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "Unknown loss type" in str(e)


def test_save_loss_mutation():
    """Loss mutations save to valid YAML."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = mutate_loss("focal")
        path = save_mutation(config, tmpdir)

        assert path.exists()

        with open(path) as f:
            loaded = yaml.safe_load(f)

        assert loaded["name"] == config["name"]
        assert loaded["model_args"]["loss_type"] == config["model_args"]["loss_type"]


if __name__ == "__main__":
    test_mutate_loss_focal()
    test_mutate_loss_label_smoothing()
    test_mutate_loss_cross_entropy()
    test_mutate_loss_invalid()
    test_save_loss_mutation()
    print("\n✓ All loss mutation tests passed")
