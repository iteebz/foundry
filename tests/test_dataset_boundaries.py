"""Tests for TokenDataset boundary conditions and edge cases."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from foundry.data.dataset import TokenDataset


def test_token_dataset_rejects_too_small():
    """Dataset with fewer tokens than block_size should raise ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "tiny.bin"
        tokens = np.array([1, 2, 3, 4, 5], dtype=np.uint16)
        tokens.tofile(path)

        with pytest.raises(ValueError, match="too small"):
            TokenDataset(path, block_size=10)


def test_token_dataset_rejects_empty():
    """Empty token file should raise ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "empty.bin"
        path.touch()

        with pytest.raises(ValueError, match="Empty"):
            TokenDataset(path, block_size=10)


def test_token_dataset_accepts_boundary():
    """Dataset exactly block_size + 1 tokens should work."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "boundary.bin"
        tokens = np.arange(11, dtype=np.uint16)
        tokens.tofile(path)

        ds = TokenDataset(path, block_size=10)
        x, y = ds[0]
        assert x.shape == (10,)
        assert y.shape == (10,)


def test_token_dataset_streaming_threshold():
    """File size > threshold uses streaming, < uses memmap."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path_small = Path(tmpdir) / "small.bin"
        path_large = Path(tmpdir) / "large.bin"

        small_tokens = np.arange(1000, dtype=np.uint16)
        large_tokens = np.arange(100000, dtype=np.uint16)

        small_tokens.tofile(path_small)
        large_tokens.tofile(path_large)

        ds_small = TokenDataset(path_small, block_size=10, streaming_threshold=10000)
        ds_large = TokenDataset(path_large, block_size=10, streaming_threshold=10000)

        assert ds_small._mode == "memmap"
        assert ds_large._mode == "streaming"


def test_token_dataset_memmap_streaming_equivalence():
    """Memmap and streaming modes return identical data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.bin"
        tokens = np.arange(1000, dtype=np.uint16)
        tokens.tofile(path)

        ds_memmap = TokenDataset(path, block_size=10, streaming_threshold=100000)
        ds_streaming = TokenDataset(path, block_size=10, streaming_threshold=100)

        for i in range(min(10, len(ds_memmap))):
            x_mem, y_mem = ds_memmap[i]
            x_stream, y_stream = ds_streaming[i]

            assert np.array_equal(x_mem, x_stream)
            assert np.array_equal(y_mem, y_stream)
