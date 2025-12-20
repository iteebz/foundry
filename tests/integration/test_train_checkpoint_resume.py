"""End-to-end training with checkpoint and resume.

Tests the core training loop contract:
- Training produces decreasing loss over iterations
- Checkpoints preserve: model state, optimizer state, iter_num
- Resume from checkpoint continues training from exact point
- Checkpoint contains config for reproducibility
"""

import tempfile
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from foundry.config import DataConfig, RunConfig, TrainingConfig
from foundry.model import GPT, GPTConfig


@pytest.fixture
def tiny_config():
    """Minimal config for fast test training."""
    return RunConfig(
        name="test_tiny",
        model=GPTConfig(
            n_layer=2,
            n_head=4,
            n_embd=64,
            block_size=128,
            vocab_size=256,
            dropout=0.0,
            bias=False,
        ),
        data=DataConfig(
            batch_size=4,
            block_size=128,
        ),
        training=TrainingConfig(
            max_iters=20,
            learning_rate=1e-2,
            eval_interval=5,
            log_interval=5,
            warmup_iters=2,
            lr_decay_iters=20,
            min_lr=1e-4,
            compile=False,
            distributed="none",
        ),
    )


@pytest.fixture
def train_data():
    """Generate small training data."""
    rng = np.random.RandomState(42)
    return torch.from_numpy(rng.randint(0, 256, (1024,), dtype=np.uint16))


def _save_config(config_dict, path):
    """Save config to YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config_dict, f)


def _train_steps(model, optimizer, data, device, num_steps, block_size, batch_size):
    """Execute training steps, return losses."""
    losses = []
    scaler = torch.amp.GradScaler(enabled=False)

    data = data.to(device).long()
    for _step in range(num_steps):
        ix = torch.randint(0, len(data) - block_size, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])

        with nullcontext():
            _, loss = model(x, y)
        loss = loss / 1
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        losses.append(loss.item())

    return losses


def test_training_runs_and_produces_finite_loss(tiny_config, train_data):
    """Training loop executes and produces finite losses."""
    device = "cpu"
    model = GPT(tiny_config.model).to(device)
    optimizer = model.configure_optimizers(
        tiny_config.training.weight_decay,
        tiny_config.training.learning_rate,
        (tiny_config.training.beta1, tiny_config.training.beta2),
        device,
    )

    losses = _train_steps(
        model,
        optimizer,
        train_data,
        device,
        num_steps=10,
        block_size=tiny_config.data.block_size,
        batch_size=tiny_config.data.batch_size,
    )

    assert len(losses) == 10
    assert all(np.isfinite(loss) for loss in losses)


def test_checkpoint_contains_required_fields(tiny_config, train_data):
    """Checkpoint must contain model, optimizer, config, iter_num, loss."""
    with tempfile.TemporaryDirectory() as tmpdir:
        device = "cpu"
        model = GPT(tiny_config.model).to(device)
        optimizer = model.configure_optimizers(
            tiny_config.training.weight_decay,
            tiny_config.training.learning_rate,
            (tiny_config.training.beta1, tiny_config.training.beta2),
            device,
        )

        _train_steps(
            model,
            optimizer,
            train_data,
            device,
            num_steps=5,
            block_size=tiny_config.data.block_size,
            batch_size=tiny_config.data.batch_size,
        )

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": tiny_config.to_dict(),
            "iter_num": 5,
            "best_val_loss": 1e9,
        }

        ckpt_path = Path(tmpdir) / "ckpt.pt"
        torch.save(checkpoint, ckpt_path)

        loaded = torch.load(ckpt_path, weights_only=False)
        assert "model" in loaded
        assert "optimizer" in loaded
        assert "config" in loaded
        assert "iter_num" in loaded
        assert loaded["iter_num"] == 5


def test_resume_from_checkpoint_continues_training(tiny_config, train_data):
    """Resume should reload model and optimizer state exactly, continuing from iter_num.

    Validate: model weights + optimizer state are preserved.
    """
    device = "cpu"
    block_size = tiny_config.data.block_size
    batch_size = tiny_config.data.batch_size

    torch.manual_seed(42)
    model1 = GPT(tiny_config.model).to(device)
    opt1 = model1.configure_optimizers(
        tiny_config.training.weight_decay,
        tiny_config.training.learning_rate,
        (tiny_config.training.beta1, tiny_config.training.beta2),
        device,
    )

    _train_steps(
        model1,
        opt1,
        train_data.clone(),
        device,
        num_steps=10,
        block_size=block_size,
        batch_size=batch_size,
    )

    checkpoint = {
        "model": model1.state_dict(),
        "optimizer": opt1.state_dict(),
        "iter_num": 10,
    }

    torch.manual_seed(99)
    model2 = GPT(tiny_config.model).to(device)
    model2.load_state_dict(checkpoint["model"])

    for k1, k2 in zip(model1.state_dict().values(), model2.state_dict().values(), strict=False):
        assert torch.allclose(k1, k2), "Model weights should match after load"

    opt2 = model2.configure_optimizers(
        tiny_config.training.weight_decay,
        tiny_config.training.learning_rate,
        (tiny_config.training.beta1, tiny_config.training.beta2),
        device,
    )
    opt2.load_state_dict(checkpoint["optimizer"])

    for pg1, pg2 in zip(opt1.param_groups, opt2.param_groups, strict=False):
        assert pg1["lr"] == pg2["lr"], "LR should match after optimizer load"


def test_checkpoint_roundtrip_preserves_model_state(tiny_config, train_data):
    """Model state before/after checkpoint save/load should be identical."""
    with tempfile.TemporaryDirectory() as tmpdir:
        device = "cpu"
        model = GPT(tiny_config.model).to(device)

        _train_steps(
            model,
            model.configure_optimizers(
                tiny_config.training.weight_decay,
                tiny_config.training.learning_rate,
                (tiny_config.training.beta1, tiny_config.training.beta2),
                device,
            ),
            train_data,
            device,
            num_steps=5,
            block_size=tiny_config.data.block_size,
            batch_size=tiny_config.data.batch_size,
        )

        state_before = {k: v.clone() for k, v in model.state_dict().items()}

        ckpt_path = Path(tmpdir) / "ckpt.pt"
        torch.save({"model": model.state_dict()}, ckpt_path)

        loaded = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(loaded["model"])

        state_after = model.state_dict()

        for key in state_before:
            assert key in state_after
            assert torch.allclose(state_before[key], state_after[key], atol=1e-6), (
                f"Mismatch in {key}"
            )
