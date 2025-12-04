"""Tests for position encoding mutations."""

import tempfile

import yaml

from foundry.mutate import mutate_position_encoding, save_mutation


def test_mutate_position_rope():
    """RoPE mutation generates valid config."""
    config = mutate_position_encoding("rope")
    assert config["name"] == "pos_rope"
    assert config["model_args"]["position_encoding"] == "rope"


def test_mutate_position_alibi():
    """ALiBi mutation generates valid config."""
    config = mutate_position_encoding("alibi")
    assert config["name"] == "pos_alibi"
    assert config["model_args"]["position_encoding"] == "alibi"


def test_mutate_position_invalid():
    """Invalid position encodings raise errors."""
    try:
        mutate_position_encoding("invalid_pos")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "Unknown position encoding" in str(e)


def test_save_position_mutation():
    """Position mutations save to valid YAML."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = mutate_position_encoding("alibi")
        path = save_mutation(config, tmpdir)

        assert path.exists()

        with open(path) as f:
            loaded = yaml.safe_load(f)

        assert loaded["name"] == config["name"]
        assert (
            loaded["model_args"]["position_encoding"] == config["model_args"]["position_encoding"]
        )


if __name__ == "__main__":
    test_mutate_position_rope()
    test_mutate_position_alibi()
    test_mutate_position_invalid()
    test_save_position_mutation()
    print("\n✓ All position encoding mutation tests passed")
