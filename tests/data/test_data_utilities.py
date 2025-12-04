"""Tests for data filtering, packing, and efficiency utilities."""

import tempfile
from pathlib import Path

import pytest

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


def test_pack_sequences_within_max_length():
    """Packed sequences don't exceed max_length."""
    seqs = [[1, 2, 3], [4, 5], [6, 7, 8, 9, 10]]
    packed = pack_sequences(seqs, max_length=10)

    for seq in packed:
        assert len(seq) == 10


def test_pack_sequences_no_token_loss():
    """Packing preserves original tokens."""
    seqs = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    packed = pack_sequences(seqs, max_length=10, eos_token=0, pad_token=0)

    original_tokens = sum(len(s) for s in seqs)
    packed_non_pad = sum(1 for p in packed for t in p if t != 0)

    assert packed_non_pad >= original_tokens


def test_pack_sequences_eos_separates():
    """EOS token appears between sequences."""
    seqs = [[1, 2], [3, 4]]
    packed = pack_sequences(seqs, max_length=10, eos_token=99, pad_token=0)

    assert 99 in packed[0]


def test_packing_efficiency_metrics():
    """Efficiency metrics are valid."""
    seqs = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    metrics = compute_packing_efficiency(seqs, max_length=10)

    assert 0 <= metrics["efficiency"] <= 1.0
    assert 0 <= metrics["waste"] <= 1.0
    assert metrics["efficiency"] + metrics["waste"] == pytest.approx(1.0)
    assert metrics["num_sequences"] == len(seqs)
    assert metrics["total_tokens"] == sum(len(s) for s in seqs)


def test_pack_sequences_empty():
    """Empty sequence list returns empty result."""
    packed = pack_sequences([], max_length=10)
    assert packed == []

    metrics = compute_packing_efficiency([], max_length=10)
    assert metrics["efficiency"] == 0.0
