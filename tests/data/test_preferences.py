"""Tests for preference pair generation."""

import pytest

from foundry.data.preferences import (
    PreferencePair,
    generate_constitution_pairs,
    generate_pairs_from_models,
    generate_pairs_from_samples,
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


def test_preference_pair_dataclass():
    """PreferencePair stores required fields."""
    pair = PreferencePair(
        prompt="test",
        chosen="good response",
        rejected="bad response",
        chosen_score=1.0,
        rejected_score=0.0,
    )

    assert pair.prompt == "test"
    assert pair.chosen == "good response"
    assert pair.rejected == "bad response"
    assert pair.chosen_score == 1.0
    assert pair.rejected_score == 0.0


def test_generate_pairs_from_samples_structure(tiny_model, char_tokenizer):
    """Best-of-N sampling generates preference pairs."""
    prompts = ["test prompt"]

    pairs = generate_pairs_from_samples(
        tiny_model,
        char_tokenizer,
        prompts,
        num_samples_per_prompt=2,
        max_new_tokens=10,
    )

    assert isinstance(pairs, list)
    assert len(pairs) == len(prompts)

    for pair in pairs:
        assert isinstance(pair, PreferencePair)
        assert pair.prompt in prompts
        assert isinstance(pair.chosen, str)
        assert isinstance(pair.rejected, str)


def test_generate_pairs_from_models_structure(tiny_model, char_tokenizer):
    """Model comparison generates preference pairs."""
    prompts = ["compare these models"]

    model_a = tiny_model
    model_b = tiny_model

    pairs = generate_pairs_from_models(
        model_a,
        model_b,
        char_tokenizer,
        prompts,
        prefer_model="a",
    )

    assert isinstance(pairs, list)
    assert len(pairs) == len(prompts)

    for pair in pairs:
        assert isinstance(pair, PreferencePair)


def test_generate_constitution_pairs_structure(tiny_model, char_tokenizer):
    """Constitutional AI generates critique-revision pairs."""
    prompts = ["test"]
    principles = ["Be helpful and harmless"]

    pairs = generate_constitution_pairs(
        tiny_model,
        char_tokenizer,
        prompts,
        principles,
        max_new_tokens=10,
    )

    assert isinstance(pairs, list)
    assert len(pairs) == len(prompts)

    for pair in pairs:
        assert isinstance(pair, PreferencePair)
        assert pair.prompt in prompts


def test_custom_reward_function(tiny_model, char_tokenizer):
    """generate_pairs_from_samples accepts custom reward."""

    def custom_reward(prompt, response):
        return float(len(response))

    pairs = generate_pairs_from_samples(
        tiny_model,
        char_tokenizer,
        ["test"],
        num_samples_per_prompt=2,
        reward_fn=custom_reward,
        max_new_tokens=10,
    )

    assert len(pairs) == 1
    assert pairs[0].chosen_score >= pairs[0].rejected_score
