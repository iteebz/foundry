import tempfile
from pathlib import Path

from foundry.data.filter import dedupe, length_filter
from foundry.data.pack import load_bin, pack_to_bin, prepare_dataset
from foundry.data.tokenize import CharTokenizer


def test_char_tokenizer():
    tok = CharTokenizer("hello world")
    assert tok.vocab_size == 8
    encoded = tok.encode("hello")
    decoded = tok.decode(encoded)
    assert decoded == "hello"


def test_tokenizer_save_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "meta.pkl"
        tok = CharTokenizer("abc")
        tok.save(path)
        loaded = CharTokenizer.load(path)
        assert loaded.vocab_size == tok.vocab_size
        assert loaded.encode("abc") == tok.encode("abc")


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


def test_prepare_dataset():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        text = "hello world" * 100
        result = prepare_dataset(text, out_dir)
        assert result["vocab_size"] == 8
        assert result["train_tokens"] + result["val_tokens"] == len(text)
        assert (out_dir / "train.bin").exists()
        assert (out_dir / "val.bin").exists()
        assert (out_dir / "meta.pkl").exists()
