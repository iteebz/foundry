from pathlib import Path

import numpy as np


def pack_to_bin(tokens: list[int], path: Path, dtype=np.uint16):
    arr = np.array(tokens, dtype=dtype)
    arr.tofile(path)
    return len(arr)


def load_bin(path: Path, dtype=np.uint16) -> np.ndarray:
    return np.memmap(path, dtype=dtype, mode="r")
