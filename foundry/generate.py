#!/usr/bin/env python3
"""Sample from a trained model checkpoint."""

import os
import pickle
from typing import Annotated

import torch
import typer

app = typer.Typer(add_completion=False)


def load_checkpoint(ckpt_path: str, device: str = "cpu"):
    """Load model from checkpoint."""
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_args = checkpoint["model_args"]
    checkpoint.get("config", {})

    from foundry.model import GPT, GPTConfig

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    state_dict = checkpoint["model"]
    for prefix in ["_orig_mod.", "transformer."]:
        for k in list(state_dict.keys()):
            if k.startswith(prefix):
                state_dict[k[len(prefix) :]] = state_dict.pop(k)

    for k in list(state_dict.keys()):
        if "attn.c_attn" in k:
            state_dict[k.replace("attn.c_attn", "attn.qkv_proj")] = state_dict.pop(k)
        elif "attn.c_proj" in k:
            state_dict[k.replace("attn.c_proj", "attn.out_proj")] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model, model_args


def load_meta(meta_path: str):
    """Load tokenizer metadata."""
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return meta["stoi"], meta["itos"]


@app.command()
def generate(
    ckpt: Annotated[str, typer.Option(help="Checkpoint path")] = "out/ckpt.pt",
    prompt: Annotated[str, typer.Option(help="Prompt string")] = "\n",
    num_samples: Annotated[int, typer.Option(help="Number of samples")] = 1,
    max_new_tokens: Annotated[int, typer.Option(help="Tokens to generate")] = 500,
    temperature: Annotated[float, typer.Option(help="Sampling temperature")] = 0.8,
    top_k: Annotated[int, typer.Option(help="Top-k sampling")] = 200,
    seed: Annotated[int, typer.Option(help="Random seed")] = 1337,
):
    """Generate text from checkpoint."""
    torch.manual_seed(seed)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model, model_args = load_checkpoint(ckpt, device)

    checkpoint_dir = os.path.dirname(ckpt)
    meta_path = os.path.join(checkpoint_dir, "meta.pkl")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.pkl not found at {meta_path}")

    stoi, itos = load_meta(meta_path)

    def encode(s):
        return [stoi[c] for c in s]

    def decode(tokens):
        return "".join([itos[i] for i in tokens])

    start_ids = encode(prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    with torch.no_grad():
        for _k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            typer.echo(decode(y[0].tolist()))
            typer.echo("---------------")


if __name__ == "__main__":
    app()
