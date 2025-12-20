"""Evaluation harness for model assessment."""

import numpy as np
import torch


def evaluate(model, batch_iter, max_iters=100, device="cuda", ctx=None):
    """Evaluate model on validation set.

    Args:
        model: Model to evaluate
        batch_iter: Iterable yielding (X, Y) tensor pairs
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
        for i, (X, Y) in enumerate(batch_iter):
            if i >= max_iters:
                break
            with ctx:
                _, loss = model(X, Y)
            losses.append(loss.item())

    model.train()

    mean_loss = np.mean(losses) if losses else float("inf")
    return {"loss": mean_loss, "perplexity": np.exp(mean_loss)}
