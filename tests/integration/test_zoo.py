"""Tests for model zoo."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from foundry.model import GPT, GPTConfig
from foundry.zoo import MODEL_CONFIGS, export_checkpoint, hf_to_foundry_config, load_pretrained


def test_hf_to_foundry_config():
    """HuggingFace config converts to Foundry GPTConfig."""
    hf_config = MODEL_CONFIGS["llama3-8b"]
    config = hf_to_foundry_config(hf_config)

    assert config.vocab_size == 128256
    assert config.n_layer == 32
    assert config.n_head == 32
    assert config.n_kv_head == 8
    assert config.n_embd == 4096
    assert config.block_size == 8192
    assert config.norm_type == "rmsnorm"
    assert config.activation == "swiglu"
    assert config.position_encoding == "rope"


def test_load_pretrained_config_only():
    """Load pretrained model from config (no checkpoint)."""
    with patch("foundry.zoo.GPT") as MockGPT:
        mock_model = MagicMock()
        mock_model.config.vocab_size = 128256
        mock_model.config.n_layer = 16
        mock_model.config.n_head = 32
        mock_model.config.n_kv_head = 8
        mock_model.config.n_embd = 2048
        mock_model.get_num_params.return_value = 1000000
        MockGPT.return_value = mock_model

        model = load_pretrained("llama3-1b", device="cpu")

        assert model.config.vocab_size == 128256
        assert model.config.n_layer == 16
        assert model.config.n_head == 32
        assert model.config.n_kv_head == 8
        assert model.config.n_embd == 2048

        params = model.get_num_params()
        assert params > 0


def test_export_checkpoint():
    """Export model checkpoint."""
    config = GPTConfig(
        vocab_size=1000,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        n_embd=128,
        block_size=256,
        bias=False,
    )
    model = GPT(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "checkpoint.pt"
        export_checkpoint(model, output_path, metadata={"test": True})

        assert output_path.exists()

        checkpoint = torch.load(output_path, map_location="cpu")
        assert "model" in checkpoint
        assert "config" in checkpoint
        assert "metadata" in checkpoint
        assert checkpoint["metadata"]["test"] is True
        assert checkpoint["config"]["vocab_size"] == 1000
        assert checkpoint["config"]["n_layer"] == 2


def test_load_exported_checkpoint():
    """Load model from exported checkpoint."""
    config = GPTConfig(
        vocab_size=1000,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        n_embd=128,
        block_size=256,
        bias=False,
    )
    model1 = GPT(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "test.pt"
        export_checkpoint(model1, checkpoint_path)

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model2 = GPT(config)
        model2.load_state_dict(checkpoint["model"])

        for p1, p2 in zip(model1.parameters(), model2.parameters(), strict=False):
            assert torch.allclose(p1, p2)


def test_all_model_configs():
    """All model configs are valid."""
    for model_name in MODEL_CONFIGS:
        config = hf_to_foundry_config(MODEL_CONFIGS[model_name])
        assert config.vocab_size > 0
        assert config.n_layer > 0
        assert config.n_head > 0
        assert config.n_kv_head > 0
        assert config.n_embd > 0


def test_unknown_model():
    """Unknown model raises ValueError."""
    try:
        load_pretrained("unknown-model")
        raise AssertionError("Should raise ValueError")
    except ValueError as e:
        assert "Unknown model" in str(e)


if __name__ == "__main__":
    test_hf_to_foundry_config()
    test_load_pretrained_config_only()
    test_export_checkpoint()
    test_load_exported_checkpoint()
    test_all_model_configs()
    test_unknown_model()
    print("\n✓ All zoo tests passed")
