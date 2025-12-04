"""Tests for constitution formatting."""

import tempfile
from pathlib import Path

from foundry.data.constitution_format import (
    format_preference_pair,
    load_constitution,
    save_constitution,
    validate_constitution,
)


def test_format_preference_pair():
    """Format preference pair correctly."""
    pair = format_preference_pair(
        prompt="How do I help?",
        chosen="Be supportive and kind.",
        rejected="Ignore them.",
    )

    assert pair["prompt"] == "How do I help?"
    assert pair["chosen"] == "Be supportive and kind."
    assert pair["rejected"] == "Ignore them."


def test_format_preference_pair_with_metadata():
    """Format preference pair with metadata."""
    pair = format_preference_pair(
        prompt="Test",
        chosen="Good",
        rejected="Bad",
        metadata={"source": "test", "category": "helpfulness"},
    )

    assert "metadata" in pair
    assert pair["metadata"]["source"] == "test"


def test_save_and_load_constitution():
    """Save and load constitution dataset."""
    pairs = [
        {
            "prompt": "How do I learn?",
            "chosen": "Study consistently.",
            "rejected": "Give up.",
        },
        {
            "prompt": "How do I grow?",
            "chosen": "Embrace challenges.",
            "rejected": "Avoid difficulty.",
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "constitution.jsonl"

        save_constitution(pairs, path)
        assert path.exists()

        loaded = load_constitution(path)

        assert len(loaded) == 2
        assert loaded[0]["prompt"] == "How do I learn?"
        assert loaded[1]["chosen"] == "Embrace challenges."


def test_validate_constitution():
    """Validate constitution format."""
    valid_pairs = [
        {"prompt": "Q1", "chosen": "A1", "rejected": "B1"},
        {"prompt": "Q2", "chosen": "A2", "rejected": "B2"},
    ]

    assert validate_constitution(valid_pairs) is True

    invalid_pairs = [
        {"prompt": "Q1", "chosen": "A1"},
        {"prompt": "Q2", "chosen": "A2", "rejected": "B2"},
    ]

    assert validate_constitution(invalid_pairs) is False


def test_load_missing_constitution():
    """Handle missing constitution file."""
    try:
        load_constitution("nonexistent.jsonl")
        raise AssertionError("Should have raised FileNotFoundError")
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    test_format_preference_pair()
    test_format_preference_pair_with_metadata()
    test_save_and_load_constitution()
    test_validate_constitution()
    test_load_missing_constitution()
    print("\n✓ All constitution format tests passed")
