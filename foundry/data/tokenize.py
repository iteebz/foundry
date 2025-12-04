import pickle
from pathlib import Path


class BPETokenizer:
    """Byte Pair Encoding tokenizer."""

    def __init__(self, vocab_size: int = 50257):
        self.vocab_size = vocab_size
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {}
        self._build_base_vocab()

    def _build_base_vocab(self):
        """Initialize with 256 byte tokens."""
        for i in range(256):
            self.vocab[i] = bytes([i])

    def _get_stats(self, ids: list[int]) -> dict[tuple[int, int], int]:
        """Count frequency of adjacent pairs."""
        counts: dict[tuple[int, int], int] = {}
        for pair in zip(ids, ids[1:], strict=False):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge(self, ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        """Replace all occurrences of pair with new_id."""
        result = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                result.append(new_id)
                i += 2
            else:
                result.append(ids[i])
                i += 1
        return result

    def fit(self, text: str) -> "BPETokenizer":
        """Train BPE on text corpus."""
        tokens = list(text.encode("utf-8"))
        num_merges = self.vocab_size - 256

        for i in range(num_merges):
            stats = self._get_stats(tokens)
            if not stats:
                break

            pair = max(stats, key=stats.get)
            new_id = 256 + i
            tokens = self._merge(tokens, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

        return self

    def encode(self, text: str) -> list[int]:
        """Encode text to token ids."""
        tokens = list(text.encode("utf-8"))

        while len(tokens) >= 2:
            stats = self._get_stats(tokens)
            pair = max(stats, key=lambda p: self.merges.get(p, float("-inf")))
            if pair not in self.merges:
                break
            tokens = self._merge(tokens, pair, self.merges[pair])

        return tokens

    def decode(self, ids: list[int]) -> str:
        """Decode token ids to text."""
        missing = [i for i in ids if i not in self.vocab]
        if missing:
            import warnings

            warnings.warn(
                f"Skipping {len(set(missing))} unknown tokens: {set(missing)}", stacklevel=2
            )
        tokens = b"".join(self.vocab[i] for i in ids if i in self.vocab)
        return tokens.decode("utf-8", errors="replace")

    def save(self, path: Path):
        """Save tokenizer to disk."""
        meta = {
            "vocab_size": self.vocab_size,
            "merges": self.merges,
            "vocab": self.vocab,
        }
        with open(path, "wb") as f:
            pickle.dump(meta, f)

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        """Load tokenizer from disk."""
        with open(path, "rb") as f:
            meta = pickle.load(f)
        tok = cls(vocab_size=meta["vocab_size"])
        tok.merges = meta["merges"]
        tok.vocab = meta["vocab"]
        return tok
