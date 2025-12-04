import tempfile
from pathlib import Path

import numpy as np
import pytest

from foundry.data.dataset import MixtureDataset, TokenDataset
from foundry.data.filter import dedupe, length_filter
from foundry.data.pack import compute_packing_efficiency, load_bin, pack_sequences, pack_to_bin


def test_dedupe():
    texts = ["a", "b", "a", "c", "b"]
    result = list(dedupe(iter(texts)))
    assert result == ["a", "b", "c"]


def test_length_filter():
    texts = ["a", "ab", "abc", "abcd"]
    result = list(length_filter(iter(texts), min_len=2, max_len=3))
    assert result == ["ab", "abc"]


def test_pack_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.bin"
        tokens = [1, 2, 3, 4, 5]
        pack_to_bin(tokens, path)
        loaded = load_bin(path)
        assert list(loaded) == tokens


class TestTokenDataset:
    def test_memmap_mode_small_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "small.bin"
            tokens = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.uint16)
            tokens.tofile(path)

            ds = TokenDataset(path, block_size=3)
            assert ds._mode == "memmap"
            assert len(ds) == 7

    def test_streaming_mode_large_threshold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "large.bin"
            tokens = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.uint16)
            tokens.tofile(path)

            ds = TokenDataset(path, block_size=3, streaming_threshold=10)
            assert ds._mode == "streaming"

    def test_memmap_streaming_identical_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.bin"
            tokens = np.array(list(range(100)), dtype=np.uint16)
            tokens.tofile(path)

            ds_memmap = TokenDataset(path, block_size=10, streaming_threshold=10000)
            ds_streaming = TokenDataset(path, block_size=10, streaming_threshold=10)

            assert ds_memmap._mode == "memmap"
            assert ds_streaming._mode == "streaming"

            for i in range(10):
                x1, y1 = ds_memmap[i]
                x2, y2 = ds_streaming[i]
                assert (x1 == x2).all()
                assert (y1 == y2).all()

    def test_rejects_empty_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.bin"
            path.touch()

            with pytest.raises(ValueError, match="Empty token file"):
                TokenDataset(path, block_size=10)

    def test_rejects_missing_file(self):
        with pytest.raises(FileNotFoundError, match="Token data not found"):
            TokenDataset("/nonexistent/path.bin", block_size=10)

    def test_rejects_too_small_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tiny.bin"
            tokens = np.array([1, 2], dtype=np.uint16)
            tokens.tofile(path)

            with pytest.raises(ValueError, match="Dataset too small"):
                TokenDataset(path, block_size=10)


class TestMixtureDataset:
    def test_weight_normalization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = Path(tmpdir) / "ds1.bin"
            path2 = Path(tmpdir) / "ds2.bin"
            np.array([1, 2, 3, 4, 5, 6], dtype=np.uint16).tofile(path1)
            np.array([7, 8, 9, 10, 11, 12], dtype=np.uint16).tofile(path2)

            ds1 = TokenDataset(path1, block_size=2)
            ds2 = TokenDataset(path2, block_size=2)

            mixture = MixtureDataset([ds1, ds2], weights=[2.0, 3.0])
            assert np.isclose(mixture.weights.sum(), 1.0)
            assert np.isclose(mixture.weights[0], 0.4)
            assert np.isclose(mixture.weights[1], 0.6)

    def test_deterministic_with_seed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = Path(tmpdir) / "ds1.bin"
            path2 = Path(tmpdir) / "ds2.bin"
            np.array([1, 2, 3, 4, 5, 6], dtype=np.uint16).tofile(path1)
            np.array([7, 8, 9, 10, 11, 12], dtype=np.uint16).tofile(path2)

            ds1 = TokenDataset(path1, block_size=2)
            ds2 = TokenDataset(path2, block_size=2)

            m1 = MixtureDataset([ds1, ds2], weights=[0.5, 0.5], seed=42)
            m2 = MixtureDataset([ds1, ds2], weights=[0.5, 0.5], seed=42)

            assert m1.indices == m2.indices

    def test_resampling_changes_indices(self):
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

    def test_rejects_mismatched_lengths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ds.bin"
            np.array([1, 2, 3, 4], dtype=np.uint16).tofile(path)
            ds = TokenDataset(path, block_size=2)

            with pytest.raises(ValueError, match="must match"):
                MixtureDataset([ds], weights=[0.5, 0.5])


class TestPackSequences:
    def test_all_sequences_within_max_length(self):
        seqs = [[1, 2, 3], [4, 5], [6, 7, 8, 9, 10]]
        packed = pack_sequences(seqs, max_length=10)

        for seq in packed:
            assert len(seq) == 10

    def test_no_token_loss(self):
        seqs = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
        packed = pack_sequences(seqs, max_length=10, eos_token=0, pad_token=0)

        original_tokens = sum(len(s) for s in seqs)
        packed_non_pad = sum(1 for p in packed for t in p if t != 0)

        assert packed_non_pad >= original_tokens

    def test_eos_separates_sequences(self):
        seqs = [[1, 2], [3, 4]]
        packed = pack_sequences(seqs, max_length=10, eos_token=99, pad_token=0)

        assert 99 in packed[0]

    def test_efficiency_metrics(self):
        seqs = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
        metrics = compute_packing_efficiency(seqs, max_length=10)

        assert 0 <= metrics["efficiency"] <= 1.0
        assert 0 <= metrics["waste"] <= 1.0
        assert metrics["efficiency"] + metrics["waste"] == pytest.approx(1.0)
        assert metrics["num_sequences"] == len(seqs)
        assert metrics["total_tokens"] == sum(len(s) for s in seqs)

    def test_empty_sequences(self):
        packed = pack_sequences([], max_length=10)
        assert packed == []

        metrics = compute_packing_efficiency([], max_length=10)
        assert metrics["efficiency"] == 0.0
