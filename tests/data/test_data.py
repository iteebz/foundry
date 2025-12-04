import tempfile
from pathlib import Path

from foundry.data.filter import dedupe, length_filter
from foundry.data.pack import load_bin, pack_to_bin


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
