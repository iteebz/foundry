"""Tests for sparse attention mutations."""

import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mutate import mutate_sparse_attention, save_mutation


def test_mutate_sparse_attention_default():
    """Sparse attention mutation with default params."""
    config = mutate_sparse_attention()
    assert config["name"] == "sparse_b64_s64"
    assert config["model_args"]["sparse_block_size"] == 64
    assert config["model_args"]["sparse_stride"] == 64


def test_mutate_sparse_attention_custom():
    """Sparse attention mutation with custom params."""
    config = mutate_sparse_attention(block_size=128, stride=256)
    assert config["name"] == "sparse_b128_s256"
    assert config["model_args"]["sparse_block_size"] == 128
    assert config["model_args"]["sparse_stride"] == 256


def test_save_sparse_attention_mutation():
    """Sparse attention mutation saves to valid YAML."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = mutate_sparse_attention(32, 64)
        path = save_mutation(config, tmpdir)

        assert path.exists()

        with open(path) as f:
            loaded = yaml.safe_load(f)

        assert loaded["name"] == config["name"]
        assert loaded["model_args"]["sparse_block_size"] == 32
        assert loaded["model_args"]["sparse_stride"] == 64


if __name__ == "__main__":
    test_mutate_sparse_attention_default()
    test_mutate_sparse_attention_custom()
    test_save_sparse_attention_mutation()
    print("\n✓ All sparse attention mutation tests passed")
