#!/usr/bin/env python3
"""CLI for synthetic data generation."""

import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(add_completion=False)


class GenerationType(StrEnum):
    self_instruct = "self_instruct"
    evol_instruct = "evol_instruct"
    math = "math"


def load_model_and_tokenizer(checkpoint_path: Path, device: str):
    import torch

    from foundry.config import RunConfig
    from foundry.model import GPT

    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    config = RunConfig.from_dict(checkpoint["config"])
    model = GPT(config.model)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    try:
        import tiktoken

        tokenizer = tiktoken.get_encoding("gpt2")
    except ImportError as e:
        raise ImportError("Synthetic generation requires tiktoken: pip install tiktoken") from e

    return model, tokenizer


@app.command()
def generate(
    checkpoint: Annotated[Path, typer.Argument(help="Model checkpoint path")],
    output: Annotated[Path, typer.Argument(help="Output JSONL path")],
    gen_type: Annotated[
        GenerationType, typer.Option("--type", "-t", help="Generation type")
    ] = GenerationType.self_instruct,
    num_samples: Annotated[int, typer.Option("--samples", "-n", help="Number to generate")] = 100,
    seed_file: Annotated[
        Path | None, typer.Option("--seed", "-s", help="Seed examples JSONL")
    ] = None,
    temperature: Annotated[float, typer.Option("--temp", help="Sampling temperature")] = 0.8,
    device: Annotated[str, typer.Option(help="Device")] = "auto",
    difficulty: Annotated[str, typer.Option(help="Math difficulty")] = "medium",
):
    """Generate synthetic training data from a trained model."""
    import torch

    from foundry.data.synthetic import (
        evol_instruct,
        generate_math_problems,
        save_synthetic_dataset,
        self_instruct,
    )

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    typer.echo(f"Loading model from {checkpoint}...")
    model, tokenizer = load_model_and_tokenizer(checkpoint, device)

    seed_tasks = []
    if seed_file and seed_file.exists():
        with seed_file.open() as f:
            seed_tasks = [json.loads(line) for line in f]
        typer.echo(f"Loaded {len(seed_tasks)} seed examples")
    elif gen_type != GenerationType.math:
        seed_tasks = [
            {
                "instruction": "Explain photosynthesis.",
                "response": "Photosynthesis is the process by which plants convert sunlight into energy.",
            },
            {
                "instruction": "What is the capital of France?",
                "response": "Paris is the capital of France.",
            },
            {
                "instruction": "Write a haiku about rain.",
                "response": "Drops fall from gray clouds\nPuddles form on quiet streets\nEarth drinks deeply now",
            },
        ]
        typer.echo("Using default seed examples")

    typer.echo(f"Generating {num_samples} {gen_type.value} examples...")

    if gen_type == GenerationType.self_instruct:
        data = self_instruct(
            model,
            tokenizer,
            seed_tasks,
            num_samples=num_samples,
            temperature=temperature,
        )
    elif gen_type == GenerationType.evol_instruct:
        data = evol_instruct(
            model,
            tokenizer,
            seed_tasks,
            num_iterations=3,
            temperature=temperature,
        )
    elif gen_type == GenerationType.math:
        data = generate_math_problems(
            model,
            tokenizer,
            difficulty=difficulty,
            num_problems=num_samples,
        )
    else:
        raise ValueError(f"Unknown generation type: {gen_type}")

    save_synthetic_dataset(data, output)
    typer.echo(f"Saved {len(data)} examples to {output}")


@app.command()
def from_preferences(
    checkpoint: Annotated[Path, typer.Argument(help="Model checkpoint path")],
    prompts: Annotated[Path, typer.Argument(help="Prompts file (one per line)")],
    output: Annotated[Path, typer.Argument(help="Output JSONL path")],
    samples_per_prompt: Annotated[
        int, typer.Option("--samples", "-n", help="Samples per prompt")
    ] = 4,
    temperature: Annotated[float, typer.Option("--temp", help="Sampling temperature")] = 0.8,
    device: Annotated[str, typer.Option(help="Device")] = "auto",
):
    """Generate preference pairs by sampling multiple responses."""
    import torch

    from foundry.data.preferences import generate_pairs_from_samples, save_preference_dataset

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    typer.echo(f"Loading model from {checkpoint}...")
    model, tokenizer = load_model_and_tokenizer(checkpoint, device)

    with prompts.open() as f:
        prompt_list = [line.strip() for line in f if line.strip()]
    typer.echo(f"Loaded {len(prompt_list)} prompts")

    typer.echo(f"Generating preference pairs ({samples_per_prompt} samples each)...")
    pairs = generate_pairs_from_samples(
        model,
        tokenizer,
        prompt_list,
        num_samples_per_prompt=samples_per_prompt,
        temperature=temperature,
    )

    save_preference_dataset(pairs, str(output))
    typer.echo(f"Saved {len(pairs)} preference pairs to {output}")


if __name__ == "__main__":
    app()
