"""Data pipeline tests: validation, boundaries, contracts, invariants."""

import numpy as np
import pytest
import torch

from foundry.data.curriculum import order_by_difficulty, score_by_length
from foundry.data.dataset import MixtureDataset, TokenDataset, validate_bin_file
from foundry.data.filter import dedupe, length_filter
from foundry.data.pack import compute_packing_efficiency, pack_sequences
from foundry.data.tokenize import BPETokenizer


def test_bpe_encode_decode_roundtrip():
    tok = BPETokenizer(vocab_size=256)
    text = "hello world"
    assert text == tok.decode(tok.encode(text))


def test_bpe_fit_and_encode():
    tok = BPETokenizer(vocab_size=300)
    tok.fit("the quick brown fox" * 10)
    encoded = tok.encode("the quick brown")
    assert len(encoded) > 0
    assert all(isinstance(t, int | np.integer) for t in encoded)


def test_bpe_max_applies_learned_merges():
    tok = BPETokenizer(vocab_size=300)
    tok.fit("aabbccdd" * 100)
    assert len(tok.encode("aabbccdd")) <= 8


@pytest.mark.parametrize(
    "seqs,max_len",
    [
        ([[1, 2, 3], [4, 5], [6, 7, 8, 9]], 10),
        ([[1] * 5, [2] * 5, [3] * 5], 10),
    ],
)
def test_pack_respects_max_length(seqs, max_len):
    packed = pack_sequences(seqs, max_length=max_len)
    assert all(len(p) == max_len for p in packed)


def test_pack_preserves_all_tokens():
    seqs = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    packed = pack_sequences(seqs, max_length=10)
    total_tokens = sum(len(s) for s in seqs)
    packed_tokens = sum(len(p) - p.count(0) for p in packed)
    assert packed_tokens == total_tokens


def test_pack_empty_input():
    assert pack_sequences([], max_length=10) == []


def test_pack_single_oversized_sequence():
    packed = pack_sequences([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], max_length=10)
    assert len(packed) == 1 and len(packed[0]) == 10
    assert packed[0][:10] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def test_pack_list_copy_safety():
    seqs = [[1, 2], [3, 4]]
    packed = pack_sequences(seqs, max_length=10)
    original = packed[0][0]
    packed[0][0] = 999
    repacked = pack_sequences(seqs, max_length=10)
    assert repacked[0][0] == original and repacked[0][0] != 999


def test_packing_efficiency_with_padding():
    metrics = compute_packing_efficiency([[1, 2], [3, 4]], max_length=10)
    assert metrics["efficiency"] < 1.0 and metrics["waste"] > 0.0


def test_packing_efficiency_from_packed():
    metrics = compute_packing_efficiency(packed_sequences=[[1, 2, 0, 0, 0], [3, 4, 5, 0, 0]])
    assert metrics["total_capacity"] == 10 and metrics["num_packed"] == 2


@pytest.mark.parametrize(
    "texts,expected",
    [
        (["hello", "world", "hello", "foo", "world"], ["hello", "world", "foo"]),
        (["z", "a", "z", "b"], ["z", "a", "b"]),
        ([], []),
    ],
)
def test_dedupe(texts, expected):
    assert list(dedupe(iter(texts))) == expected


@pytest.mark.parametrize("algo", ["sha256", "md5"])
def test_dedupe_hash_algo(algo):
    assert len(list(dedupe(iter(["same", "same"]), hash_algo=algo))) == 1


@pytest.mark.parametrize(
    "texts,min_len,max_len,count_tokens,expected",
    [
        (["a", "hello", "world"], 2, 5, False, ["hello", "world"]),
        (["one", "one two", "one two three"], 2, 2, True, ["one two"]),
        (["a", "hello world", "xyz"], 2, None, False, ["hello world", "xyz"]),
    ],
)
def test_length_filter(texts, min_len, max_len, count_tokens, expected):
    result = list(
        length_filter(iter(texts), min_len=min_len, max_len=max_len, count_tokens=count_tokens)
    )
    assert result == expected


def test_validate_bin_file_success(token_bin_file):
    result = validate_bin_file(token_bin_file)
    assert result["num_tokens"] == 10 and "sha256" in result


def test_validate_bin_file_not_found():
    from pathlib import Path

    with pytest.raises(FileNotFoundError):
        validate_bin_file(Path("/nonexistent/file.bin"))


def test_validate_bin_file_empty():
    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        path = Path(f.name)
    with pytest.raises(ValueError, match="Empty"):
        validate_bin_file(path)


def test_token_dataset_loads_and_indexes(token_bin_file):
    ds = TokenDataset(token_bin_file, block_size=3)
    assert len(ds) == 7
    x, y = ds[0]
    assert len(x) == 3 and len(y) == 3


def test_token_dataset_bounds_enforced(token_bin_file):
    ds = TokenDataset(token_bin_file, block_size=3)
    with pytest.raises(IndexError):
        ds[100]
    with pytest.raises(IndexError):
        ds[-1]


def test_token_dataset_validation_optional(token_bin_file):
    ds = TokenDataset(token_bin_file, block_size=3, validate=False)
    assert not hasattr(ds, "_validation")


def test_token_dataset_correct_target_shift(token_bin_file):
    ds = TokenDataset(token_bin_file, block_size=3)
    x, y = ds[0]
    assert torch.equal(x, torch.tensor([1, 2, 3], dtype=torch.int64))
    assert torch.equal(y, torch.tensor([2, 3, 4], dtype=torch.int64))


def test_mixture_dataset_has_samples(small_datasets):
    mixture = MixtureDataset(small_datasets, weights=[0.5, 0.5], seed=42)
    assert len(mixture) > 0


def test_mixture_dataset_deterministic(small_datasets):
    m1 = MixtureDataset(small_datasets, weights=[0.7, 0.3], seed=1337)
    m2 = MixtureDataset(small_datasets, weights=[0.7, 0.3], seed=1337)
    assert m1.indices == m2.indices


def test_mixture_dataset_different_seeds(small_datasets):
    m1 = MixtureDataset(small_datasets, weights=[0.7, 0.3], seed=1)
    m2 = MixtureDataset(small_datasets, weights=[0.7, 0.3], seed=2)
    assert m1.indices != m2.indices


def test_mixture_dataset_weight_normalization(small_datasets):
    mixture = MixtureDataset(small_datasets, weights=[2.0, 8.0], seed=42)
    assert np.isclose(mixture.weights.sum(), 1.0)


def test_mixture_dataset_rejects_negative_weights(small_datasets):
    with pytest.raises(ValueError, match="positive"):
        MixtureDataset(small_datasets, weights=[0.5, -0.5])


def test_mixture_dataset_rejects_weight_mismatch(small_datasets):
    with pytest.raises(ValueError, match="must match"):
        MixtureDataset(small_datasets, weights=[0.5])


def test_mixture_dataset_resample(small_datasets):
    mixture = MixtureDataset(small_datasets, weights=[0.5, 0.5], seed=1337)
    original = list(mixture.indices)
    mixture.resample(seed=9999)
    assert mixture.indices != original


def test_curriculum_order_by_length():
    seqs = [[1, 2, 3], [1], [1, 2, 3, 4, 5]]
    ordered, scores = order_by_difficulty(seqs, score_fn=score_by_length)
    assert ordered[0] == [1] and ordered[-1] == [1, 2, 3, 4, 5]
    assert len(scores) == 3


def test_curriculum_order_reverse():
    seqs = [[1], [1, 2], [1, 2, 3]]
    ordered, _ = order_by_difficulty(seqs, score_fn=score_by_length, reverse=True)
    assert ordered[0] == [1, 2, 3] and ordered[-1] == [1]


def test_curriculum_order_preserves_data():
    seqs = [[1, 2], [3, 4], [5, 6]]
    ordered, _ = order_by_difficulty(seqs, score_fn=score_by_length)
    assert {tuple(s) for s in ordered} == {tuple(s) for s in seqs}
