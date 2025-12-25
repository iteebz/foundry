"""Prepare datasets for training - downloads and tokenizes to .bin files."""

from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset


def prepare_tinystories(out_dir: Path | str = "data/tinystories", encoding: str = "cl100k_base"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    enc = tiktoken.get_encoding(encoding)

    ds = load_dataset("roneneldan/TinyStories", split="train")
    val_ds = load_dataset("roneneldan/TinyStories", split="validation")

    def tokenize_split(dataset) -> list[int]:
        tokens = []
        for example in dataset:
            tokens.extend(enc.encode_ordinary(example["text"]))
            tokens.append(enc.eot_token)
        return tokens

    print("Tokenizing train...")
    train_tokens = tokenize_split(ds)

    print("Tokenizing val...")
    val_tokens = tokenize_split(val_ds)

    train_arr = np.array(train_tokens, dtype=np.uint32)
    val_arr = np.array(val_tokens, dtype=np.uint32)

    train_arr.tofile(out_dir / "train.bin")
    val_arr.tofile(out_dir / "val.bin")

    return {
        "encoding": encoding,
        "vocab_size": enc.n_vocab,
        "train_tokens": len(train_arr),
        "val_tokens": len(val_arr),
    }


if __name__ == "__main__":
    prepare_tinystories()
