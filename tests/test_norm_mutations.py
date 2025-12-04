"""Tests for normalization mutations."""

import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mutate import mutate_norm, save_mutation


def test_mutate_norm_layernorm():
    """LayerNorm mutation generates valid config."""
    config = mutate_norm("layernorm")
    assert config["name"] == "norm_layernorm"
    assert config["model_args"]["norm_type"] == "layernorm"


def test_mutate_norm_rmsnorm():
    """RMSNorm mutation generates valid config."""
    config = mutate_norm("rmsnorm")
    assert config["name"] == "norm_rmsnorm"
    assert config["model_args"]["norm_type"] == "rmsnorm"


def test_mutate_norm_invalid():
    """Invalid norm types raise errors."""
    try:
        mutate_norm("invalid_norm")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "Unknown norm type" in str(e)


def test_save_norm_mutation():
    """Norm mutations save to valid YAML."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = mutate_norm("layernorm")
        path = save_mutation(config, tmpdir)

        assert path.exists()

        with open(path) as f:
            loaded = yaml.safe_load(f)

        assert loaded["name"] == config["name"]
        assert loaded["model_args"]["norm_type"] == config["model_args"]["norm_type"]


if __name__ == "__main__":
    test_mutate_norm_layernorm()
    test_mutate_norm_rmsnorm()
    test_mutate_norm_invalid()
    test_save_norm_mutation()
    print("\n✓ All normalization mutation tests passed")
