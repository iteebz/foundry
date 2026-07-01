"""CLI entrypoint for mutation generation."""

from typing import Annotated

import typer

from . import (
    mutate_activation,
    mutate_adam_betas,
    mutate_attention,
    mutate_batch_size,
    mutate_conversation_format,
    mutate_depth,
    mutate_grad_clip,
    mutate_lora_alpha,
    mutate_lora_dropout,
    mutate_lora_rank,
    mutate_loss,
    mutate_lr,
    mutate_mla,
    mutate_moe,
    mutate_norm,
    mutate_position_encoding,
    mutate_sliding_window,
    mutate_sparse_attention,
    mutate_warmup,
    mutate_weight_decay,
    mutate_width,
    save_mutation,
)

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def attention(variant: Annotated[str, typer.Argument(help="gqa_2kv|gqa_1kv|mha")]):
    """Mutate attention mechanism."""
    path = save_mutation(mutate_attention(variant))
    typer.echo(f"\nGenerated: {path}")
    typer.echo(f"Run with: python -m foundry.train {path}")


@app.command()
def depth(layers: Annotated[int, typer.Argument(help="Number of transformer layers")]):
    """Mutate model depth."""
    path = save_mutation(mutate_depth(layers))
    typer.echo(f"\nGenerated: {path}")
    typer.echo(f"Run with: python -m foundry.train {path}")


@app.command()
def width(dim: Annotated[int, typer.Argument(help="Embedding dimension")]):
    """Mutate model width."""
    path = save_mutation(mutate_width(dim))
    typer.echo(f"\nGenerated: {path}")
    typer.echo(f"Run with: python -m foundry.train {path}")


@app.command()
def lr(rate: Annotated[float, typer.Argument(help="Learning rate")]):
    """Mutate learning rate."""
    path = save_mutation(mutate_lr(rate))
    typer.echo(f"\nGenerated: {path}")
    typer.echo(f"Run with: python -m foundry.train {path}")


@app.command()
def norm(variant: Annotated[str, typer.Argument(help="rmsnorm|layernorm")]):
    """Mutate normalization layer."""
    path = save_mutation(mutate_norm(variant))
    typer.echo(f"\nGenerated: {path}")
    typer.echo(f"Run with: python -m foundry.train {path}")


@app.command()
def activation(variant: Annotated[str, typer.Argument(help="swiglu|gelu|glu")]):
    """Mutate activation function."""
    path = save_mutation(mutate_activation(variant))
    typer.echo(f"\nGenerated: {path}")
    typer.echo(f"Run with: python -m foundry.train {path}")


@app.command()
def position(variant: Annotated[str, typer.Argument(help="rope|alibi")]):
    """Mutate position encoding."""
    path = save_mutation(mutate_position_encoding(variant))
    typer.echo(f"\nGenerated: {path}")
    typer.echo(f"Run with: python -m foundry.train {path}")


@app.command()
def loss(variant: Annotated[str, typer.Argument(help="cross_entropy|focal|label_smoothing|dpo")]):
    """Mutate loss function."""
    path = save_mutation(mutate_loss(variant))
    typer.echo(f"\nGenerated: {path}")
    typer.echo(f"Run with: python -m foundry.train {path}")


@app.command()
def batch_size(size: Annotated[int, typer.Argument(help="Batch size")]):
    """Mutate batch size."""
    path = save_mutation(mutate_batch_size(size))
    typer.echo(f"\nGenerated: {path}")
    typer.echo(f"Run with: python -m foundry.train {path}")


@app.command()
def warmup(iters: Annotated[int, typer.Argument(help="Warmup iterations")]):
    """Mutate warmup schedule."""
    path = save_mutation(mutate_warmup(iters))
    typer.echo(f"\nGenerated: {path}")
    typer.echo(f"Run with: python -m foundry.train {path}")


@app.command()
def grad_clip(max_norm: Annotated[float, typer.Argument(help="Max gradient norm")]):
    """Mutate gradient clipping."""
    path = save_mutation(mutate_grad_clip(max_norm))
    typer.echo(f"\nGenerated: {path}")
    typer.echo(f"Run with: python -m foundry.train {path}")


@app.command()
def weight_decay(value: Annotated[float, typer.Argument(help="Weight decay coefficient")]):
    """Mutate weight decay."""
    path = save_mutation(mutate_weight_decay(value))
    typer.echo(f"\nGenerated: {path}")
    typer.echo(f"Run with: python -m foundry.train {path}")


@app.command()
def adam_betas(
    beta1: Annotated[float, typer.Argument(help="Adam beta1")],
    beta2: Annotated[float, typer.Argument(help="Adam beta2")],
):
    """Mutate Adam optimizer betas."""
    path = save_mutation(mutate_adam_betas(beta1, beta2))
    typer.echo(f"\nGenerated: {path}")
    typer.echo(f"Run with: python -m foundry.train {path}")


@app.command()
def lora_rank(r: Annotated[int, typer.Argument(help="LoRA rank")]):
    """Mutate LoRA rank."""
    path = save_mutation(mutate_lora_rank(r))
    typer.echo(f"\nGenerated: {path}")
    typer.echo(f"Run with: python -m foundry.train {path}")


@app.command()
def lora_alpha(alpha: Annotated[int, typer.Argument(help="LoRA alpha scaling")]):
    """Mutate LoRA alpha."""
    path = save_mutation(mutate_lora_alpha(alpha))
    typer.echo(f"\nGenerated: {path}")
    typer.echo(f"Run with: python -m foundry.train {path}")


@app.command()
def lora_dropout(p: Annotated[float, typer.Argument(help="LoRA dropout probability")]):
    """Mutate LoRA dropout."""
    path = save_mutation(mutate_lora_dropout(p))
    typer.echo(f"\nGenerated: {path}")
    typer.echo(f"Run with: python -m foundry.train {path}")


@app.command()
def conversation_format(variant: Annotated[str, typer.Argument(help="chatml|llama3|alpaca")]):
    """Mutate conversation format."""
    path = save_mutation(mutate_conversation_format(variant))
    typer.echo(f"\nGenerated: {path}")
    typer.echo(f"Run with: python -m foundry.train {path}")


@app.command()
def mla(latent_dim: Annotated[int | None, typer.Argument(help="Latent dimension")] = None):
    """Mutate to Multi-Latent Attention."""
    path = save_mutation(mutate_mla(latent_dim))
    typer.echo(f"\nGenerated: {path}")
    typer.echo(f"Run with: python -m foundry.train {path}")


@app.command()
def moe(
    n_experts: Annotated[int, typer.Argument(help="Number of experts")] = 8,
    top_k: Annotated[int, typer.Argument(help="Top-k routing")] = 2,
):
    """Mutate to Mixture of Experts."""
    path = save_mutation(mutate_moe(n_experts, top_k))
    typer.echo(f"\nGenerated: {path}")
    typer.echo(f"Run with: python -m foundry.train {path}")


@app.command()
def sliding_window(window_size: Annotated[int, typer.Argument(help="Window size")] = 256):
    """Mutate to sliding window attention."""
    path = save_mutation(mutate_sliding_window(window_size))
    typer.echo(f"\nGenerated: {path}")
    typer.echo(f"Run with: python -m foundry.train {path}")


@app.command()
def sparse_attention(
    block_size: Annotated[int, typer.Argument(help="Block size")] = 64,
    stride: Annotated[int | None, typer.Argument(help="Stride")] = None,
):
    """Mutate to sparse attention."""
    path = save_mutation(mutate_sparse_attention(block_size, stride))
    typer.echo(f"\nGenerated: {path}")
    typer.echo(f"Run with: python -m foundry.train {path}")


if __name__ == "__main__":
    app()
