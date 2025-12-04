"""Tests for optimizer mutations."""

import sys
import tempfile
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mutate import mutate_weight_decay, mutate_adam_betas, save_mutation


def test_mutate_weight_decay():
    """Weight decay mutation generates valid config."""
    config = mutate_weight_decay(1e-2)
    assert config["name"] == "wd_1e-02"
    assert config["training"]["weight_decay"] == 1e-2


def test_mutate_adam_betas():
    """Adam betas mutation generates valid config."""
    config = mutate_adam_betas(0.9, 0.999)
    assert config["name"] == "adam_b1_0.9_b2_0.999"
    assert config["training"]["beta1"] == 0.9
    assert config["training"]["beta2"] == 0.999


def test_save_weight_decay_mutation():
    """Weight decay mutations save to valid YAML."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = mutate_weight_decay(1e-2)
        path = save_mutation(config, tmpdir)
        
        assert path.exists()
        
        with open(path) as f:
            loaded = yaml.safe_load(f)
        
        assert loaded["name"] == config["name"]
        assert loaded["training"]["weight_decay"] == config["training"]["weight_decay"]


def test_save_adam_betas_mutation():
    """Adam betas mutations save to valid YAML."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = mutate_adam_betas(0.9, 0.999)
        path = save_mutation(config, tmpdir)
        
        assert path.exists()
        
        with open(path) as f:
            loaded = yaml.safe_load(f)
        
        assert loaded["name"] == config["name"]
        assert loaded["training"]["beta1"] == config["training"]["beta1"]
        assert loaded["training"]["beta2"] == config["training"]["beta2"]


if __name__ == "__main__":
    test_mutate_weight_decay()
    test_mutate_adam_betas()
    test_save_weight_decay_mutation()
    test_save_adam_betas_mutation()
    print("\n✓ All optimizer mutation tests passed")
