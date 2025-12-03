from typing import Callable
import pickle
from pathlib import Path


class CharTokenizer:
    def __init__(self, text: str | None = None):
        self.stoi: dict[str, int] = {}
        self.itos: dict[int, str] = {}
        if text:
            self.fit(text)

    def fit(self, text: str) -> 'CharTokenizer':
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        return self

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids: list[int]) -> str:
        return ''.join(self.itos.get(i, '') for i in ids)

    def save(self, path: Path):
        meta = {'vocab_size': self.vocab_size, 'stoi': self.stoi, 'itos': self.itos}
        with open(path, 'wb') as f:
            pickle.dump(meta, f)

    @classmethod
    def load(cls, path: Path) -> 'CharTokenizer':
        with open(path, 'rb') as f:
            meta = pickle.load(f)
        tok = cls()
        tok.stoi = meta['stoi']
        tok.itos = meta['itos']
        return tok


def encode(text: str, tokenizer: CharTokenizer) -> list[int]:
    return tokenizer.encode(text)


def decode(ids: list[int], tokenizer: CharTokenizer) -> str:
    return tokenizer.decode(ids)
