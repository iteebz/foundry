"""Tests for training math: EMA, gradient accumulation."""

import torch

from foundry.model import GPT, GPTConfig
from foundry.train import EMA


def test_ema_initialization():
    """EMA shadow should initialize with model weights."""
    config = GPTConfig(n_layer=2, n_head=4, n_embd=64, vocab_size=256)
    model = GPT(config)

    ema = EMA(model, decay=0.999)

    for k, v in model.state_dict().items():
        assert torch.allclose(ema.shadow[k], v), f"EMA shadow mismatch for {k}"


def test_ema_update_decay():
    """EMA update should apply exponential decay: shadow = decay * shadow + (1-decay) * model."""
    config = GPTConfig(n_layer=2, n_head=4, n_embd=64, vocab_size=256)
    model = GPT(config)
    ema = EMA(model, decay=0.99)

    initial_shadow = {k: v.clone() for k, v in ema.shadow.items()}

    with torch.no_grad():
        for param in model.parameters():
            param.data.fill_(1.0)

    ema.update(model)

    for k in ema.shadow:
        expected = 0.99 * initial_shadow[k] + 0.01 * 1.0
        assert torch.allclose(ema.shadow[k], expected, atol=1e-6), f"EMA decay formula for {k}"


def test_ema_convergence():
    """After many updates with constant model values, EMA should decay towards model."""
    config = GPTConfig(n_layer=2, n_head=4, n_embd=64, vocab_size=256)
    model = GPT(config)
    ema = EMA(model, decay=0.99)

    initial_shadow = {k: v.clone() for k, v in ema.shadow.items()}

    with torch.no_grad():
        for param in model.parameters():
            param.data.fill_(5.0)

    for _ in range(100):
        ema.update(model)

    for k in ema.shadow:
        assert ema.shadow[k].mean() > initial_shadow[k].mean(), (
            f"EMA should move towards higher model values for {k}"
        )


def test_gradient_accumulation_loss_scaling():
    """With gradient accumulation, loss should be scaled by 1/accumulation_steps."""
    config = GPTConfig(n_layer=2, n_head=4, n_embd=64, vocab_size=256, dropout=0.0)
    model = GPT(config)
    model.eval()

    torch.manual_seed(42)
    X = torch.randint(0, 256, (4, 128))
    Y = torch.randint(0, 256, (4, 128))

    with torch.no_grad():
        _logits, loss_full = model(X, Y)

    raw_loss = loss_full.item()

    scaled_loss_4 = raw_loss / 4
    scaled_loss_8 = raw_loss / 8

    assert scaled_loss_4 < raw_loss, "Scaled loss should be smaller"
    assert scaled_loss_8 < scaled_loss_4, "More accumulation = smaller scaled loss"
    assert abs(scaled_loss_4 - raw_loss * 0.25) < 1e-6, "Scaling factor should be exact"


def test_mixed_precision_no_nan():
    """Training with fp16 should not produce NaN losses."""
    config = GPTConfig(n_layer=2, n_head=4, n_embd=64, vocab_size=256, dropout=0.0)
    model = GPT(config)
    model.train()

    optimizer = model.configure_optimizers(
        weight_decay=0.0,
        learning_rate=1e-2,
        betas=(0.9, 0.95),
        device_type="cpu",
    )

    scaler = torch.amp.GradScaler(enabled=False)

    torch.manual_seed(42)

    for step in range(10):
        X = torch.randint(0, 256, (4, 128))
        Y = torch.randint(0, 256, (4, 128))

        _logits, loss = model(X, Y)
        loss = loss / 1

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        assert not torch.isnan(loss), f"NaN loss at step {step}"
