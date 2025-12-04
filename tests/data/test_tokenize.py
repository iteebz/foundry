"""Tests for tokenizers."""

from foundry.data.tokenize import BPETokenizer, CharTokenizer


def test_char_tokenizer_fit():
    """CharTokenizer learns vocabulary from text."""
    tok = CharTokenizer()
    tok.fit("hello world")
    assert tok.vocab_size > 0
    assert "h" in tok.stoi
    assert " " in tok.stoi


def test_char_tokenizer_encode_decode():
    """CharTokenizer round-trips text."""
    tok = CharTokenizer()
    tok.fit("hello world")
    text = "hello"
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    assert decoded == text


def test_bpe_tokenizer_base_vocab():
    """BPETokenizer initializes with 256 byte tokens."""
    tok = BPETokenizer(vocab_size=300)
    assert len(tok.vocab) == 256
    assert tok.vocab[0] == b"\x00"
    assert tok.vocab[255] == b"\xff"


def test_bpe_tokenizer_fit():
    """BPETokenizer learns merges from text."""
    tok = BPETokenizer(vocab_size=300)
    tok.fit("hello world hello")
    assert len(tok.merges) > 0
    assert len(tok.vocab) > 256


def test_bpe_tokenizer_encode():
    """BPETokenizer encodes text to ids."""
    tok = BPETokenizer(vocab_size=300)
    tok.fit("hello world")
    ids = tok.encode("hello")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert len(ids) > 0


def test_bpe_tokenizer_decode():
    """BPETokenizer decodes ids to text."""
    tok = BPETokenizer(vocab_size=300)
    tok.fit("hello world")
    text = "hello"
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    assert decoded == text


def test_bpe_tokenizer_compression():
    """BPETokenizer compresses repeated patterns."""
    tok = BPETokenizer(vocab_size=300)
    text = "aaaaaaa bbbbbbb"
    tok.fit(text)
    ids = tok.encode(text)
    assert len(ids) < len(text.encode("utf-8"))


if __name__ == "__main__":
    test_char_tokenizer_fit()
    test_char_tokenizer_encode_decode()
    test_bpe_tokenizer_base_vocab()
    test_bpe_tokenizer_fit()
    test_bpe_tokenizer_encode()
    test_bpe_tokenizer_decode()
    test_bpe_tokenizer_compression()
    print("\n✓ All tokenizer tests passed")
