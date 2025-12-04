"""Pytest configuration for all tests."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

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


@pytest.fixture
def token_bin_file():
    """10-token uint16 binary file."""
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.uint16)
        data.tofile(f)
        path = Path(f.name)
    yield path
    path.unlink()


@pytest.fixture
def small_datasets():
    """Two small TensorDatasets for mixture testing."""
    ds1 = torch.utils.data.TensorDataset(torch.tensor([[1, 2], [3, 4]]))
    ds2 = torch.utils.data.TensorDataset(torch.tensor([[5, 6]]))
    return [ds1, ds2]
