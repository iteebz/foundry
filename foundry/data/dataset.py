"""PyTorch Datasets for token data with auto-detection and mixture sampling."""

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def validate_bin_file(path: Path, expected_dtype: np.dtype | None = None) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Token file not found: {path}")

    if path.stat().st_size == 0:
        raise ValueError(f"Empty token file: {path}")

    if expected_dtype is None:
        expected_dtype = _detect_dtype(path)

    if path.stat().st_size % np.dtype(expected_dtype).itemsize != 0:
        raise ValueError(
            f"File size {path.stat().st_size} not divisible by {expected_dtype} itemsize"
        )

    data = np.memmap(path, dtype=expected_dtype, mode="r")
    file_hash = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            file_hash.update(chunk)

    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "num_tokens": len(data),
        "dtype": str(expected_dtype),
        "sha256": file_hash.hexdigest(),
    }


def _detect_dtype(path: Path) -> np.dtype:
    """Auto-detect dtype: uint16 for GPT-2 (50k vocab), uint32 for cl100k (100k vocab)."""
    file_size = path.stat().st_size
    if file_size == 0:
        return np.dtype(np.uint16)
    if file_size % 4 == 0 and file_size >= 4:
        data = np.memmap(path, dtype=np.uint32, mode="r")
        if len(data) > 0:
            max_token = int(data[: min(100000, len(data))].max())
            if max_token > 65535:
                return np.dtype(np.uint32)
    return np.dtype(np.uint16)


class TokenDataset(Dataset):
    """Memory-mapped or streaming token dataset (auto-detects based on size)."""

    def __init__(
        self,
        data_path: str | Path,
        block_size: int,
        streaming_threshold: int = 1_000_000_000,
        validate: bool = True,
        dtype: np.dtype | None = None,
    ):
        data_path = Path(data_path)

        if validate and data_path.stat().st_size == 0:
            raise ValueError(f"Empty token file: {data_path}")

        self.dtype = dtype if dtype is not None else _detect_dtype(data_path)

        if validate:
            self._validation = validate_bin_file(data_path, expected_dtype=self.dtype)

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
        self.data = np.memmap(self.data_path, dtype=self.dtype, mode="r")
        self._length = len(self.data) - self.block_size

        if self._length <= 0:
            raise ValueError(
                f"Dataset too small: {len(self.data)} tokens < block_size {self.block_size}"
            )

    def _init_streaming(self):
        """Streaming for large files (sequential access with buffering)."""
        self.data = np.memmap(self.data_path, dtype=self.dtype, mode="r")
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
        return self._getitem_streaming(idx)

    def _getitem_memmap(self, idx):
        """O(1) random access with bounds checking."""
        if idx < 0 or idx >= self._length:
            raise IndexError(f"Index {idx} out of bounds (dataset length: {self._length})")
        if idx + self.block_size > len(self.data):
            raise IndexError(
                f"Cannot read {self.block_size} tokens from index {idx} (file has {len(self.data)} tokens)"
            )
        x = torch.from_numpy(self.data[idx : idx + self.block_size].astype(np.int64))
        y = torch.from_numpy(self.data[idx + 1 : idx + 1 + self.block_size].astype(np.int64))
        return x, y

    def _getitem_streaming(self, idx):
        """Buffered sequential access (optimized for DataLoader workers)."""
        if (
            self._buffer is None
            or idx < self._buffer_start
            or idx >= self._buffer_start + len(self._buffer) - self.block_size
        ):
            start = max(0, idx - self._buffer_size // 2)
            end = min(len(self.data), start + self._buffer_size)
            self._buffer = self.data[start:end]
            self._buffer_start = start

        buffer_idx = idx - self._buffer_start
        x = torch.from_numpy(
            self._buffer[buffer_idx : buffer_idx + self.block_size].astype(np.int64)
        )
        y = torch.from_numpy(
            self._buffer[buffer_idx + 1 : buffer_idx + 1 + self.block_size].astype(np.int64)
        )
        return x, y


class MixtureDataset(Dataset):
    """Sample from multiple datasets according to weights (reference-grade)."""

    def __init__(self, datasets: list[Dataset], weights: list[float], seed: int = 1337):
        if len(datasets) != len(weights):
            raise ValueError(f"datasets ({len(datasets)}) and weights ({len(weights)}) must match")
        if not datasets:
            raise ValueError("Must provide at least one dataset")
        if any(w <= 0 for w in weights):
            raise ValueError("All weights must be positive")

        self.datasets = datasets
        self.weights = np.array(weights, dtype=np.float32)
        self.weights /= self.weights.sum()
        self.seed = seed

        self.indices = self._build_mixture_schedule()

    def _build_mixture_schedule(self):
        """Build deterministic sample schedule balancing dataset weights."""
        rng = np.random.RandomState(self.seed)

        sum(len(ds) for ds in self.datasets)
        samples_per_dataset = [
            max(1, int(len(ds) * w)) for ds, w in zip(self.datasets, self.weights, strict=True)
        ]

        indices = []
        oversample_warning = []
        for ds_idx, n_samples in enumerate(samples_per_dataset):
            dataset_size = len(self.datasets[ds_idx])
            if n_samples > dataset_size:
                oversample_warning.append(
                    f"Dataset {ds_idx}: requesting {n_samples} samples from {dataset_size} items (weight={self.weights[ds_idx]:.2%}). Using oversampling with replacement."
                )
                sampled_indices = rng.choice(dataset_size, size=n_samples, replace=True)
            else:
                sampled_indices = rng.choice(dataset_size, size=n_samples, replace=False)

            indices.extend([(ds_idx, int(idx)) for idx in sampled_indices])

        if oversample_warning:
            import warnings

            warnings.warn("\n".join(oversample_warning), stacklevel=2)

        rng.shuffle(indices)
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ds_idx, sample_idx = self.indices[idx]
        return self.datasets[ds_idx][sample_idx]

    def resample(self, seed: int | None = None):
        """Resample mixture schedule (call at epoch boundaries)."""
        if seed is not None:
            self.seed = seed
        else:
            self.seed += 1
        self.indices = self._build_mixture_schedule()
