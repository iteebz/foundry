"""Tests for training hyperparameter mutations."""

import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mutate import mutate_batch_size, mutate_grad_clip, mutate_warmup, save_mutation


def test_mutate_batch_size():
    """Batch size mutation generates valid config."""
    config = mutate_batch_size(128)
    assert config["name"] == "batch_128"
    assert config["training"]["batch_size"] == 128


def test_mutate_warmup():
    """Warmup mutation generates valid config."""
    config = mutate_warmup(500)
    assert config["name"] == "warmup_500"
    assert config["training"]["warmup_iters"] == 500


def test_mutate_grad_clip():
    """Grad clip mutation generates valid config."""
    config = mutate_grad_clip(0.5)
    assert config["name"] == "gradclip_0.5"
    assert config["training"]["grad_clip"] == 0.5


def test_save_batch_size_mutation():
    """Batch size mutations save to valid YAML."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = mutate_batch_size(128)
        path = save_mutation(config, tmpdir)

        assert path.exists()

        with open(path) as f:
            loaded = yaml.safe_load(f)

        assert loaded["name"] == config["name"]
        assert loaded["training"]["batch_size"] == config["training"]["batch_size"]


def test_save_warmup_mutation():
    """Warmup mutations save to valid YAML."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = mutate_warmup(500)
        path = save_mutation(config, tmpdir)

        assert path.exists()

        with open(path) as f:
            loaded = yaml.safe_load(f)

        assert loaded["name"] == config["name"]
        assert loaded["training"]["warmup_iters"] == config["training"]["warmup_iters"]


def test_save_grad_clip_mutation():
    """Grad clip mutations save to valid YAML."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = mutate_grad_clip(0.5)
        path = save_mutation(config, tmpdir)

        assert path.exists()

        with open(path) as f:
            loaded = yaml.safe_load(f)

        assert loaded["name"] == config["name"]
        assert loaded["training"]["grad_clip"] == config["training"]["grad_clip"]


if __name__ == "__main__":
    test_mutate_batch_size()
    test_mutate_warmup()
    test_mutate_grad_clip()
    test_save_batch_size_mutation()
    test_save_warmup_mutation()
    test_save_grad_clip_mutation()
    print("\n✓ All training hyperparameter mutation tests passed")
