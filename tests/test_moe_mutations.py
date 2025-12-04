"""Tests for MoE mutations."""

import sys
import tempfile
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mutate import mutate_moe, save_mutation


def test_mutate_moe_default():
    """MoE mutation with default params."""
    config = mutate_moe()
    assert config["name"] == "moe_8e2k"
    assert config["model_args"]["mlp_type"] == "moe"
    assert config["model_args"]["moe_n_experts"] == 8
    assert config["model_args"]["moe_top_k"] == 2


def test_mutate_moe_custom():
    """MoE mutation with custom params."""
    config = mutate_moe(n_experts=16, top_k=4)
    assert config["name"] == "moe_16e4k"
    assert config["model_args"]["mlp_type"] == "moe"
    assert config["model_args"]["moe_n_experts"] == 16
    assert config["model_args"]["moe_top_k"] == 4


def test_save_moe_mutation():
    """MoE mutation saves to valid YAML."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = mutate_moe(4, 2)
        path = save_mutation(config, tmpdir)
        
        assert path.exists()
        
        with open(path) as f:
            loaded = yaml.safe_load(f)
        
        assert loaded["name"] == config["name"]
        assert loaded["model_args"]["mlp_type"] == "moe"
        assert loaded["model_args"]["moe_n_experts"] == 4


if __name__ == "__main__":
    test_mutate_moe_default()
    test_mutate_moe_custom()
    test_save_moe_mutation()
    print("\n✓ All MoE mutation tests passed")
