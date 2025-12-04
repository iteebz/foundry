"""Tests for conversation format mutations."""

import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mutate import mutate_conversation_format, save_mutation


def test_mutate_conversation_format_chatml():
    """ChatML format mutation generates valid config."""
    config = mutate_conversation_format("chatml")
    assert config["name"] == "conv_chatml"
    assert config["data"]["conversation_format"] == "chatml"


def test_mutate_conversation_format_llama3():
    """Llama3 format mutation generates valid config."""
    config = mutate_conversation_format("llama3")
    assert config["name"] == "conv_llama3"
    assert config["data"]["conversation_format"] == "llama3"


def test_mutate_conversation_format_alpaca():
    """Alpaca format mutation generates valid config."""
    config = mutate_conversation_format("alpaca")
    assert config["name"] == "conv_alpaca"
    assert config["data"]["conversation_format"] == "alpaca"


def test_save_conversation_mutation():
    """Conversation format mutations save to valid YAML."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = mutate_conversation_format("chatml")
        path = save_mutation(config, tmpdir)

        assert path.exists()

        with open(path) as f:
            loaded = yaml.safe_load(f)

        assert loaded["name"] == config["name"]
        assert loaded["data"]["conversation_format"] == "chatml"


def test_unknown_conversation_format():
    """Unknown format raises ValueError."""
    try:
        mutate_conversation_format("unknown")
        raise AssertionError("Should raise ValueError")
    except ValueError as e:
        assert "Unknown format" in str(e)


if __name__ == "__main__":
    test_mutate_conversation_format_chatml()
    test_mutate_conversation_format_llama3()
    test_mutate_conversation_format_alpaca()
    test_save_conversation_mutation()
    test_unknown_conversation_format()
    print("\n✓ All conversation mutation tests passed")
