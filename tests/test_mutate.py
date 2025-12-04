"""Tests for mutation generator."""

import sys
import tempfile
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mutate import (
    mutate_attention,
    mutate_depth,
    mutate_width,
    mutate_lr,
    save_mutation,
    load_baseline,
)


def test_load_baseline():
    """Baseline config loads successfully."""
    config = load_baseline()
    assert "name" in config
    assert "training" in config
    assert "model_args" in config


def test_mutate_attention():
    """Attention mutations generate valid configs."""
    for variant in ["gqa_2kv", "gqa_1kv", "mha"]:
        config = mutate_attention(variant)
        assert config["name"] == f"attn_{variant}"
        assert "n_kv_head" in config["model_args"]
        assert config["model_args"]["n_kv_head"] >= 1


def test_mutate_depth():
    """Depth mutations generate valid configs."""
    config = mutate_depth(8)
    assert config["name"] == "depth_8L"
    assert config["model_args"]["n_layer"] == 8


def test_mutate_width():
    """Width mutations generate valid configs."""
    config = mutate_width(512)
    assert config["name"] == "width_512d"
    assert config["model_args"]["n_embd"] == 512
    assert config["model_args"]["n_embd"] % config["model_args"]["n_head"] == 0


def test_mutate_lr():
    """LR mutations generate valid configs."""
    config = mutate_lr(3e-4)
    assert config["name"] == "lr_3e-04"
    assert config["training"]["learning_rate"] == 3e-4


def test_save_mutation():
    """Mutations save to valid YAML."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = mutate_attention("gqa_2kv")
        path = save_mutation(config, tmpdir)
        
        assert path.exists()
        
        with open(path) as f:
            loaded = yaml.safe_load(f)
        
        assert loaded["name"] == config["name"]
        assert loaded["model_args"]["n_kv_head"] == config["model_args"]["n_kv_head"]


def test_attention_variant_validation():
    """Invalid attention variants raise errors."""
    try:
        mutate_attention("invalid_variant")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown attention variant" in str(e)


if __name__ == "__main__":
    test_load_baseline()
    test_mutate_attention()
    test_mutate_depth()
    test_mutate_width()
    test_mutate_lr()
    test_save_mutation()
    test_attention_variant_validation()
    print("\n✓ All mutation tests passed")
