"""PyTorch Dataset for token data with auto-detection."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TokenDataset(Dataset):
    """Memory-mapped or streaming token dataset (auto-detects based on size)."""

    def __init__(self, data_path: str | Path, block_size: int, streaming_threshold: int = 1_000_000_000):
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Token data not found: {data_path}")
        if data_path.stat().st_size == 0:
            raise ValueError(f"Empty token file: {data_path}")

        self.data_path = data_path
        self.block_size = block_size
        file_size = data_path.stat().st_size

        if file_size > streaming_threshold:
            self._mode = "streaming"
            self._init_streaming()
        else:
            self._mode = "memmap"
            self._init_memmap()

    def _init_memmap(self):
        """Fast random access via memory mapping."""
        self.data = np.memmap(self.data_path, dtype=np.uint16, mode="r")
        self._length = len(self.data) - self.block_size

        if self._length <= 0:
            raise ValueError(
                f"Dataset too small: {len(self.data)} tokens < block_size {self.block_size}"
            )

    def _init_streaming(self):
        """Streaming for large files (sequential access with buffering)."""
        self.data = np.memmap(self.data_path, dtype=np.uint16, mode="r")
        self._length = len(self.data) - self.block_size
        
        if self._length <= 0:
            raise ValueError(
                f"Dataset too small: {len(self.data)} tokens < block_size {self.block_size}"
            )
        
        self._buffer_size = 10_000_000
        self._buffer = None
        self._buffer_start = 0

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        if self._mode == "memmap":
            return self._getitem_memmap(idx)
        else:
            return self._getitem_streaming(idx)

    def _getitem_memmap(self, idx):
        """O(1) random access."""
        x = torch.from_numpy(self.data[idx : idx + self.block_size].astype(np.int64))
        y = torch.from_numpy(self.data[idx + 1 : idx + 1 + self.block_size].astype(np.int64))
        return x, y

    def _getitem_streaming(self, idx):
        """Buffered sequential access (optimized for DataLoader workers)."""
        if self._buffer is None or idx < self._buffer_start or idx >= self._buffer_start + len(self._buffer) - self.block_size:
            start = max(0, idx - self._buffer_size // 2)
            end = min(len(self.data), start + self._buffer_size)
            self._buffer = self.data[start:end]
            self._buffer_start = start

        buffer_idx = idx - self._buffer_start
        x = torch.from_numpy(self._buffer[buffer_idx : buffer_idx + self.block_size].astype(np.int64))
        y = torch.from_numpy(self._buffer[buffer_idx + 1 : buffer_idx + 1 + self.block_size].astype(np.int64))
        return x, y
