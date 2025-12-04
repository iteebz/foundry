from pathlib import Path

import numpy as np

from .tokenize import CharTokenizer


def pack_to_bin(tokens: list[int], path: Path, dtype=np.uint16):
    arr = np.array(tokens, dtype=dtype)
    arr.tofile(path)
    return len(arr)


def load_bin(path: Path, dtype=np.uint16) -> np.ndarray:
    return np.memmap(path, dtype=dtype, mode="r")


def prepare_dataset(
    text: str,
    out_dir: Path,
    train_split: float = 0.9,
    tokenizer: CharTokenizer | None = None,
) -> dict:
    if tokenizer is None:
        tokenizer = CharTokenizer(text)

    tokens = tokenizer.encode(text)
    n = len(tokens)
    split_idx = int(n * train_split)

    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"
    meta_path = out_dir / "meta.pkl"

    train_count = pack_to_bin(tokens[:split_idx], train_path)
    val_count = pack_to_bin(tokens[split_idx:], val_path)
    tokenizer.save(meta_path)

    return {
        "train_tokens": train_count,
        "val_tokens": val_count,
        "vocab_size": tokenizer.vocab_size,
    }
