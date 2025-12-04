"""Tests for constitution mutations."""

import tempfile

import yaml

from foundry.mutate import mutate_constitution, save_mutation


def test_mutate_constitution():
    """Constitution mutation generates config."""
    config = mutate_constitution("data/constitutions/helpful.jsonl")

    assert config["name"] == "constitution_helpful"
    assert config["data"]["constitution"]["enabled"] is True
    assert config["data"]["constitution"]["path"] == "data/constitutions/helpful.jsonl"


def test_save_constitution_mutation():
    """Constitution mutation saves to valid YAML."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = mutate_constitution("data/constitutions/harmless.jsonl")
        path = save_mutation(config, tmpdir)

        assert path.exists()

        with open(path) as f:
            loaded = yaml.safe_load(f)

        assert loaded["name"] == config["name"]
        assert loaded["data"]["constitution"]["enabled"] is True


if __name__ == "__main__":
    test_mutate_constitution()
    test_save_constitution_mutation()
    print("\n✓ All constitution mutation tests passed")
