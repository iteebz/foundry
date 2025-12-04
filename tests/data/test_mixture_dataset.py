"""Tests for MixtureDataset behavior and properties."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from foundry.data.dataset import MixtureDataset, TokenDataset


def test_mixture_dataset_normalizes_weights():
    """Mixture weights normalize to sum=1."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path1 = Path(tmpdir) / "ds1.bin"
        path2 = Path(tmpdir) / "ds2.bin"

        np.array([1, 2, 3, 4, 5, 6], dtype=np.uint16).tofile(path1)
        np.array([100, 101, 102, 103, 104, 105], dtype=np.uint16).tofile(path2)

        ds1 = TokenDataset(path1, block_size=2)
        ds2 = TokenDataset(path2, block_size=2)

        mixture = MixtureDataset([ds1, ds2], weights=[2.0, 3.0], seed=42)

        assert np.isclose(mixture.weights.sum(), 1.0)
        assert np.isclose(mixture.weights[0], 0.4)
        assert np.isclose(mixture.weights[1], 0.6)


def test_mixture_dataset_deterministic_seeding():
    """Same seed produces same sampling distribution."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path1 = Path(tmpdir) / "ds1.bin"
        path2 = Path(tmpdir) / "ds2.bin"

        np.array([1, 2, 3, 4, 5, 6], dtype=np.uint16).tofile(path1)
        np.array([100, 101, 102, 103, 104, 105], dtype=np.uint16).tofile(path2)

        ds1a = TokenDataset(path1, block_size=2)
        ds2a = TokenDataset(path2, block_size=2)

        ds1b = TokenDataset(path1, block_size=2)
        ds2b = TokenDataset(path2, block_size=2)

        mix_a = MixtureDataset([ds1a, ds2a], weights=[0.5, 0.5], seed=42)
        mix_b = MixtureDataset([ds1b, ds2b], weights=[0.5, 0.5], seed=42)

        assert mix_a.indices == mix_b.indices


def test_mixture_dataset_rejects_weight_mismatch():
    """Mixture with mismatched weights/datasets raises ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "ds.bin"
        np.array([1, 2, 3, 4], dtype=np.uint16).tofile(path)

        ds = TokenDataset(path, block_size=2)

        with pytest.raises(ValueError):
            MixtureDataset([ds], weights=[0.5, 0.5])


def test_mixture_dataset_resample_changes_indices():
    """Resampling changes the indices."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path1 = Path(tmpdir) / "ds1.bin"
        path2 = Path(tmpdir) / "ds2.bin"
        np.array([1, 2, 3, 4, 5, 6], dtype=np.uint16).tofile(path1)
        np.array([7, 8, 9, 10, 11, 12], dtype=np.uint16).tofile(path2)

        ds1 = TokenDataset(path1, block_size=2)
        ds2 = TokenDataset(path2, block_size=2)

        mixture = MixtureDataset([ds1, ds2], weights=[0.5, 0.5], seed=42)
        original_indices = mixture.indices[:]

        mixture.resample()
        assert mixture.indices != original_indices
