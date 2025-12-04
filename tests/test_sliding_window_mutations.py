"""Tests for sliding window mutations."""

import sys
import tempfile
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mutate import mutate_sliding_window, save_mutation


def test_mutate_sliding_window_default():
    """Sliding window mutation with default size."""
    config = mutate_sliding_window()
    assert config["name"] == "sw256"
    assert config["model_args"]["sliding_window_size"] == 256


def test_mutate_sliding_window_custom():
    """Sliding window mutation with custom size."""
    config = mutate_sliding_window(512)
    assert config["name"] == "sw512"
    assert config["model_args"]["sliding_window_size"] == 512


def test_save_sliding_window_mutation():
    """Sliding window mutation saves to valid YAML."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = mutate_sliding_window(1024)
        path = save_mutation(config, tmpdir)
        
        assert path.exists()
        
        with open(path) as f:
            loaded = yaml.safe_load(f)
        
        assert loaded["name"] == config["name"]
        assert loaded["model_args"]["sliding_window_size"] == 1024


if __name__ == "__main__":
    test_mutate_sliding_window_default()
    test_mutate_sliding_window_custom()
    test_save_sliding_window_mutation()
    print("\n✓ All sliding window mutation tests passed")
