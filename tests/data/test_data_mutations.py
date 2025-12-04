"""Tests for data pipeline mutations."""

import tempfile

import yaml

from foundry.mutate import mutate_data_filter, save_mutation


def test_mutate_data_filter_min_only():
    """Data filter with min_len generates valid config."""
    config = mutate_data_filter(min_len=10)
    assert config["name"] == "filter_min10_dedupe"
    assert config["data"]["min_len"] == 10
    assert config["data"]["dedupe"] is True


def test_mutate_data_filter_min_max():
    """Data filter with min and max generates valid config."""
    config = mutate_data_filter(min_len=10, max_len=1000)
    assert config["name"] == "filter_min10_max1000_dedupe"
    assert config["data"]["min_len"] == 10
    assert config["data"]["max_len"] == 1000


def test_mutate_data_filter_no_dedupe():
    """Data filter without dedupe generates valid config."""
    config = mutate_data_filter(min_len=10, dedupe=False)
    assert config["name"] == "filter_min10"
    assert config["data"]["dedupe"] is False


def test_save_data_filter_mutation():
    """Data filter mutations save to valid YAML."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = mutate_data_filter(min_len=10, max_len=1000)
        path = save_mutation(config, tmpdir)

        assert path.exists()

        with open(path) as f:
            loaded = yaml.safe_load(f)

        assert loaded["name"] == config["name"]
        assert loaded["data"]["min_len"] == config["data"]["min_len"]
        assert loaded["data"]["max_len"] == config["data"]["max_len"]


if __name__ == "__main__":
    test_mutate_data_filter_min_only()
    test_mutate_data_filter_min_max()
    test_mutate_data_filter_no_dedupe()
    test_save_data_filter_mutation()
    print("\n✓ All data pipeline mutation tests passed")
