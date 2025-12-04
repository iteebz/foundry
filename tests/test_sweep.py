"""Tests for sweep runner."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sweep import generate_mutation


def test_generate_mutation_attention():
    """Attention mutation generation works."""
    path = generate_mutation("attention", "gqa_2kv")
    assert path.exists()
    assert "attn_gqa_2kv" in path.stem
    path.unlink()


def test_generate_mutation_depth():
    """Depth mutation generation works."""
    path = generate_mutation("depth", "8")
    assert path.exists()
    assert "depth_8L" in path.stem
    path.unlink()


def test_generate_mutation_lr():
    """LR mutation generation works."""
    path = generate_mutation("lr", "3e-4")
    assert path.exists()
    assert "lr_3e-04" in path.stem
    path.unlink()


if __name__ == "__main__":
    test_generate_mutation_attention()
    test_generate_mutation_depth()
    test_generate_mutation_lr()
    print("\n✓ All sweep tests passed")
