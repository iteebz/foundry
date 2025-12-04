from pathlib import Path

import numpy as np


def pack_to_bin(tokens: list[int], path: Path, dtype=np.uint16):
    arr = np.array(tokens, dtype=dtype)
    arr.tofile(path)
    return len(arr)


def load_bin(path: Path, dtype=np.uint16) -> np.ndarray:
    return np.memmap(path, dtype=dtype, mode="r")


def pack_sequences(
    sequences: list[list[int]],
    max_length: int,
    eos_token: int = 0,
    pad_token: int = 0,
) -> list[list[int]]:
    """Pack variable-length sequences to minimize padding (greedy bin-packing).

    Args:
        sequences: List of variable-length token sequences
        max_length: Maximum sequence length (block_size)
        eos_token: Token to separate packed sequences
        pad_token: Token for padding

    Returns:
        List of packed sequences, each of length max_length

    Example:
        >>> seqs = [[1,2,3], [4,5], [6,7,8,9]]
        >>> packed = pack_sequences(seqs, max_length=10, eos_token=0)
        >>> # Result: [[1,2,3,0,4,5,0,6,7], [8,9,0,0,0,0,0,0,0]]
    """
    if not sequences:
        return []

    sorted_seqs = sorted(sequences, key=len, reverse=True)

    packed = []
    current_pack = []
    current_length = 0

    for seq in sorted_seqs:
        seq_with_eos = seq + [eos_token]

        if current_length + len(seq_with_eos) <= max_length:
            current_pack.extend(seq_with_eos)
            current_length += len(seq_with_eos)
        else:
            if current_pack:
                current_pack.extend([pad_token] * (max_length - current_length))
                packed.append(list(current_pack))

            current_pack = seq_with_eos[:]
            current_length = len(seq_with_eos)

            if current_length > max_length:
                current_pack = current_pack[:max_length]
                current_length = max_length

    if current_pack:
        current_pack.extend([pad_token] * (max_length - current_length))
        packed.append(list(current_pack))

    return packed


def compute_packing_efficiency(
    sequences: list[list[int]] | None = None,
    max_length: int | None = None,
    packed_sequences: list[list[int]] | None = None,
) -> dict:
    """Compute packing efficiency metrics.

    Args:
        sequences: List of variable-length sequences (will be packed if packed_sequences not provided)
        max_length: Maximum sequence length
        packed_sequences: Pre-packed sequences (if provided, skips packing step)

    Returns:
        Dictionary with efficiency metrics
    """
    if packed_sequences is not None:
        if not packed_sequences:
            return {"efficiency": 0.0, "waste": 0.0, "num_packed": 0}
        packed = packed_sequences
        total_tokens = sum(len(seq) for seq in packed)
        total_capacity = len(packed) * len(packed[0])
    else:
        if not sequences or max_length is None:
            return {"efficiency": 0.0, "waste": 0.0, "num_packed": 0}
        packed = pack_sequences(sequences, max_length)
        total_tokens = sum(len(seq) for seq in sequences)
        total_capacity = len(packed) * max_length

    padding_tokens = total_capacity - total_tokens
    efficiency = total_tokens / total_capacity if total_capacity > 0 else 0.0
    waste = padding_tokens / total_capacity if total_capacity > 0 else 0.0

    return {
        "efficiency": efficiency,
        "waste": waste,
        "num_sequences": len(sequences) if sequences else None,
        "num_packed": len(packed),
        "total_tokens": total_tokens,
        "total_capacity": total_capacity,
        "padding_tokens": padding_tokens,
    }
