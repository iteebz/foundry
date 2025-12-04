"""Tests for synthetic data generation."""

import pytest

from foundry.data.synthetic import (
    evol_instruct,
    generate_math_problems,
    self_instruct,
)
from foundry.data.tokenize import CharTokenizer
from foundry.model import GPT, GPTConfig


@pytest.fixture
def tiny_model():
    """Tiny model for testing."""
    config = GPTConfig(
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
        block_size=128,
        vocab_size=100,
        dropout=0.0,
    )
    model = GPT(config)
    model.eval()
    return model


@pytest.fixture
def char_tokenizer():
    """Simple character tokenizer."""
    return CharTokenizer()


def test_self_instruct_structure(tiny_model, char_tokenizer):
    """Self-instruct returns list of instruction-response dicts."""
    seed_tasks = [
        {"instruction": "Add 2+2", "response": "4"},
        {"instruction": "Spell 'cat'", "response": "c-a-t"},
    ]

    result = self_instruct(
        tiny_model,
        char_tokenizer,
        seed_tasks,
        num_samples=2,
        max_new_tokens=20,
    )

    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, dict)
        if "instruction" in item:
            assert "response" in item


def test_evol_instruct_structure(tiny_model, char_tokenizer):
    """Evol-instruct returns evolved tasks."""
    base_tasks = [
        {"instruction": "Count to 5", "response": "1, 2, 3, 4, 5"},
    ]

    result = evol_instruct(
        tiny_model,
        char_tokenizer,
        base_tasks,
        num_iterations=1,
        temperature=0.7,
    )

    assert isinstance(result, list)
    assert len(result) >= len(base_tasks)


def test_generate_math_problems_structure(tiny_model, char_tokenizer):
    """Math problem generation returns structured output."""
    result = generate_math_problems(
        tiny_model,
        char_tokenizer,
        difficulty="easy",
        num_problems=2,
    )

    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, dict)
        if "problem" in item:
            assert "solution" in item
            assert "answer" in item


def test_self_instruct_handles_empty_seeds(tiny_model, char_tokenizer):
    """Self-instruct handles edge case of empty seed tasks."""
    with pytest.raises((ValueError, IndexError)):
        self_instruct(tiny_model, char_tokenizer, [], num_samples=1)


def test_evol_instruct_preserves_base_tasks(tiny_model, char_tokenizer):
    """Evol-instruct includes original tasks in output."""
    base = [{"instruction": "test", "response": "answer"}]
    result = evol_instruct(tiny_model, char_tokenizer, base, num_iterations=1)

    assert any(t["instruction"] == "test" for t in result)
