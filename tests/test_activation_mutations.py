"""Tests for activation mutations."""

import sys
import tempfile
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mutate import mutate_activation, save_mutation


def test_mutate_activation_gelu():
    """GELU mutation generates valid config."""
    config = mutate_activation("gelu")
    assert config["name"] == "act_gelu"
    assert config["model_args"]["activation"] == "gelu"


def test_mutate_activation_glu():
    """GLU mutation generates valid config."""
    config = mutate_activation("glu")
    assert config["name"] == "act_glu"
    assert config["model_args"]["activation"] == "glu"


def test_mutate_activation_swiglu():
    """SwiGLU mutation generates valid config."""
    config = mutate_activation("swiglu")
    assert config["name"] == "act_swiglu"
    assert config["model_args"]["activation"] == "swiglu"


def test_mutate_activation_invalid():
    """Invalid activations raise errors."""
    try:
        mutate_activation("invalid_act")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown activation" in str(e)


def test_save_activation_mutation():
    """Activation mutations save to valid YAML."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = mutate_activation("gelu")
        path = save_mutation(config, tmpdir)
        
        assert path.exists()
        
        with open(path) as f:
            loaded = yaml.safe_load(f)
        
        assert loaded["name"] == config["name"]
        assert loaded["model_args"]["activation"] == config["model_args"]["activation"]


if __name__ == "__main__":
    test_mutate_activation_gelu()
    test_mutate_activation_glu()
    test_mutate_activation_swiglu()
    test_mutate_activation_invalid()
    test_save_activation_mutation()
    print("\n✓ All activation mutation tests passed")
