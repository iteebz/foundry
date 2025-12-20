import tempfile
from pathlib import Path

import pytest
import torch

from foundry.checkpoint import ResumeState, load_checkpoint, validate_checkpoint
from foundry.model import GPT, GPTConfig


@pytest.fixture
def tiny_model():
    config = GPTConfig(
        block_size=128,
        vocab_size=1024,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        n_embd=64,
        dropout=0.0,
        bias=False,
    )
    return GPT(config)


def test_load_checkpoint_returns_resume_state(tiny_model):
    """load_checkpoint returns ResumeState with iter_num and best_val_loss."""
    model = tiny_model
    optimizer = model.configure_optimizers(0.01, 1e-4, (0.9, 0.95), "cpu")
    original_weight = model.transformer.wte.weight.clone()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        temp_path = f.name

    try:
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": {"test": "config"},
            "iter_num": 42,
            "best_val_loss": 1.5,
        }
        torch.save(checkpoint, temp_path)

        model.transformer.wte.weight.data.zero_()

        resume_state = load_checkpoint(model, optimizer, temp_path)

        assert isinstance(resume_state, ResumeState)
        assert resume_state.iter_num == 42
        assert resume_state.best_val_loss == 1.5
        assert resume_state.config["test"] == "config"
        assert torch.allclose(model.transformer.wte.weight, original_weight)
    finally:
        Path(temp_path).unlink()


def test_load_checkpoint_without_optimizer(tiny_model):
    """load_checkpoint works when optimizer is None."""
    model = tiny_model

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        temp_path = f.name

    try:
        checkpoint = {
            "model": model.state_dict(),
            "config": {},
            "iter_num": 10,
            "best_val_loss": 2.0,
        }
        torch.save(checkpoint, temp_path)

        resume_state = load_checkpoint(model, None, temp_path)

        assert resume_state.iter_num == 10
    finally:
        Path(temp_path).unlink()


def test_load_checkpoint_missing_iter_num_raises(tiny_model):
    """load_checkpoint raises if iter_num missing."""
    model = tiny_model

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        temp_path = f.name

    try:
        checkpoint = {
            "model": model.state_dict(),
            "best_val_loss": 1.0,
        }
        torch.save(checkpoint, temp_path)

        with pytest.raises(ValueError, match="iter_num"):
            load_checkpoint(model, None, temp_path)
    finally:
        Path(temp_path).unlink()


def test_load_checkpoint_missing_best_val_loss_raises(tiny_model):
    """load_checkpoint raises if best_val_loss missing."""
    model = tiny_model

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        temp_path = f.name

    try:
        checkpoint = {
            "model": model.state_dict(),
            "iter_num": 5,
        }
        torch.save(checkpoint, temp_path)

        with pytest.raises(ValueError, match="best_val_loss"):
            load_checkpoint(model, None, temp_path)
    finally:
        Path(temp_path).unlink()


def test_validate_checkpoint_empty_model_raises():
    """validate_checkpoint raises on empty model state."""
    with pytest.raises(ValueError, match="empty"):
        validate_checkpoint({"model": {}})
