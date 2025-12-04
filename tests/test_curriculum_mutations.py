"""Tests for curriculum mutations."""

import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mutate import mutate_curriculum, save_mutation


def test_mutate_curriculum_length():
    """Curriculum mutation with length strategy."""
    config = mutate_curriculum(strategy="length", schedule="linear", num_stages=4)
    
    assert config["name"] == "curriculum_length_linear_4stages"
    assert config["data"]["curriculum"]["enabled"] is True
    assert config["data"]["curriculum"]["strategy"] == "length"
    assert config["data"]["curriculum"]["schedule"] == "linear"
    assert config["data"]["curriculum"]["num_stages"] == 4


def test_mutate_curriculum_perplexity():
    """Curriculum mutation with perplexity strategy."""
    config = mutate_curriculum(strategy="perplexity", schedule="step", num_stages=8)
    
    assert config["name"] == "curriculum_perplexity_step_8stages"
    assert config["data"]["curriculum"]["strategy"] == "perplexity"
    assert config["data"]["curriculum"]["schedule"] == "step"
    assert config["data"]["curriculum"]["num_stages"] == 8


def test_save_curriculum_mutation():
    """Curriculum mutation saves to valid YAML."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = mutate_curriculum(strategy="length", schedule="linear")
        path = save_mutation(config, tmpdir)
        
        assert path.exists()
        
        with open(path) as f:
            loaded = yaml.safe_load(f)
        
        assert loaded["name"] == config["name"]
        assert loaded["data"]["curriculum"]["enabled"] is True


def test_invalid_strategy():
    """Invalid strategy raises error."""
    try:
        mutate_curriculum(strategy="invalid")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "Unknown strategy" in str(e)


def test_invalid_schedule():
    """Invalid schedule raises error."""
    try:
        mutate_curriculum(schedule="invalid")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "Unknown schedule" in str(e)


if __name__ == "__main__":
    test_mutate_curriculum_length()
    test_mutate_curriculum_perplexity()
    test_save_curriculum_mutation()
    test_invalid_strategy()
    test_invalid_schedule()
    print("\n✓ All curriculum mutation tests passed")
