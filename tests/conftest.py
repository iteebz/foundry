"""Pytest configuration for all tests."""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import pytest  # noqa: E402

from foundry.model import GPT, GPTConfig  # noqa: E402


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self):
        self.vocab_size = 100

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 100 for c in text[:10]]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i + 32) for i in ids[:20])


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
def mock_tokenizer():
    """Mock tokenizer."""
    return MockTokenizer()
