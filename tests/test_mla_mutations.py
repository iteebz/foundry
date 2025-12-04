"""Tests for MLA mutations."""

import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mutate import mutate_mla, save_mutation


def test_mutate_mla_default():
    """MLA mutation with default latent dim."""
    config = mutate_mla()
    assert config["name"] == "mla_latent192"
    assert config["model_args"]["attention_type"] == "mla"
    assert config["model_args"]["mla_latent_dim"] == 192


def test_mutate_mla_custom():
    """MLA mutation with custom latent dim."""
    config = mutate_mla(256)
    assert config["name"] == "mla_latent256"
    assert config["model_args"]["attention_type"] == "mla"
    assert config["model_args"]["mla_latent_dim"] == 256


def test_save_mla_mutation():
    """MLA mutation saves to valid YAML."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = mutate_mla(128)
        path = save_mutation(config, tmpdir)

        assert path.exists()

        with open(path) as f:
            loaded = yaml.safe_load(f)

        assert loaded["name"] == config["name"]
        assert loaded["model_args"]["attention_type"] == "mla"
        assert loaded["model_args"]["mla_latent_dim"] == 128


if __name__ == "__main__":
    test_mutate_mla_default()
    test_mutate_mla_custom()
    test_save_mla_mutation()
    print("\n✓ All MLA mutation tests passed")
