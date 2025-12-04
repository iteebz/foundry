"""Tests for LoRA mutations."""

import tempfile

import yaml

from foundry.mutate import (
    mutate_lora_alpha,
    mutate_lora_dropout,
    mutate_lora_rank,
    save_mutation,
)


def test_mutate_lora_rank():
    """LoRA rank mutation generates valid config."""
    config = mutate_lora_rank(16)
    assert config["name"] == "lora_r16"
    assert config["lora"]["r"] == 16
    assert config["lora"]["enabled"] is True


def test_mutate_lora_alpha():
    """LoRA alpha mutation generates valid config."""
    config = mutate_lora_alpha(32)
    assert config["name"] == "lora_alpha32"
    assert config["lora"]["lora_alpha"] == 32
    assert config["lora"]["enabled"] is True


def test_mutate_lora_dropout():
    """LoRA dropout mutation generates valid config."""
    config = mutate_lora_dropout(0.1)
    assert config["name"] == "lora_dropout0.1"
    assert config["lora"]["lora_dropout"] == 0.1
    assert config["lora"]["enabled"] is True


def test_save_lora_mutation():
    """LoRA mutations save to valid YAML."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = mutate_lora_rank(16)
        path = save_mutation(config, tmpdir)

        assert path.exists()

        with open(path) as f:
            loaded = yaml.safe_load(f)

        assert loaded["name"] == config["name"]
        assert loaded["lora"]["r"] == config["lora"]["r"]


if __name__ == "__main__":
    test_mutate_lora_rank()
    test_mutate_lora_alpha()
    test_mutate_lora_dropout()
    test_save_lora_mutation()
    print("\n✓ All LoRA mutation tests passed")
