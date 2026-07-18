"""Tests for resolve_device/resolve_dtype auto-detection branches.

Invariants:
- explicit config value ("cuda", "cpu", "float16", ...) always passes through
  unchanged, hardware state is never consulted.
- "auto" walks a fallback ladder: cuda -> mps -> cpu (device),
  cuda+bf16-support -> bfloat16, else float16 (dtype).
"""

from dataclasses import dataclass

from foundry.train_common import resolve_device, resolve_dtype


@dataclass
class MockTrainingConfig:
    device: str = "auto"
    dtype: str = "auto"


@dataclass
class MockConfig:
    training: MockTrainingConfig = None

    def __post_init__(self):
        if self.training is None:
            self.training = MockTrainingConfig()


def test_resolve_device_explicit_passthrough_ignores_hardware(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    config = MockConfig(MockTrainingConfig(device="cpu"))
    assert resolve_device(config) == "cpu"


def test_resolve_device_auto_prefers_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)
    config = MockConfig(MockTrainingConfig(device="auto"))
    assert resolve_device(config) == "cuda"


def test_resolve_device_auto_falls_back_to_mps_when_no_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)
    config = MockConfig(MockTrainingConfig(device="auto"))
    assert resolve_device(config) == "mps"


def test_resolve_device_auto_falls_back_to_cpu_when_no_accelerator(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)
    config = MockConfig(MockTrainingConfig(device="auto"))
    assert resolve_device(config) == "cpu"


def test_resolve_dtype_explicit_passthrough_ignores_hardware(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.is_bf16_supported", lambda: True)
    config = MockConfig(MockTrainingConfig(dtype="float32"))
    assert resolve_dtype(config) == "float32"


def test_resolve_dtype_auto_picks_bfloat16_when_cuda_supports_it(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.is_bf16_supported", lambda: True)
    config = MockConfig(MockTrainingConfig(dtype="auto"))
    assert resolve_dtype(config) == "bfloat16"


def test_resolve_dtype_auto_falls_back_to_float16_without_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    config = MockConfig(MockTrainingConfig(dtype="auto"))
    assert resolve_dtype(config) == "float16"


def test_resolve_dtype_auto_falls_back_to_float16_when_cuda_lacks_bf16(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.is_bf16_supported", lambda: False)
    config = MockConfig(MockTrainingConfig(dtype="auto"))
    assert resolve_dtype(config) == "float16"
