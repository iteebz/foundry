"""Evaluation harness for model assessment."""

import numpy as np
import torch


def evaluate(model, get_batch, max_iters=100, device="cuda", ctx=None):
    """Evaluate model on validation set.

    Args:
        model: Model to evaluate
        get_batch: Function(split) -> (X, Y) tensors
        max_iters: Number of batches to evaluate
        device: Device to run on
        ctx: Context manager for autocast

    Returns:
        dict with 'loss' and 'perplexity' keys
    """
    if ctx is None:
        ctx = torch.no_grad()

    model.eval()
    losses = []

    with torch.no_grad():
        for _ in range(max_iters):
            X, Y = get_batch("val")
            with ctx:
                _, loss = model(X, Y)
            losses.append(loss.item())

    model.train()

    mean_loss = np.mean(losses)
    return {"loss": mean_loss, "perplexity": np.exp(mean_loss)}
