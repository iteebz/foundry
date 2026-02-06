#!/usr/bin/env python3
"""Foundry CLI: unified entry point for training tools."""

import typer

from foundry.cli import compare, sweep, synthetic
from foundry.cli.lr_finder import lr_find_from_config

app = typer.Typer(
    name="foundry",
    help="Training and experimentation toolkit",
    add_completion=False,
    no_args_is_help=True,
)

app.add_typer(sweep.app, name="sweep", help="Run parallel mutation sweeps")
app.add_typer(synthetic.app, name="synthetic", help="Generate synthetic training data")
app.add_typer(compare.app, name="compare", help="A/B test baseline vs mutation")


@app.command("lr-find")
def lr_find_cmd(
    config: str = typer.Argument(..., help="Path to experiment YAML"),
    plot: str = typer.Option("out/lr_find.png", "--plot", "-o", help="Output plot path"),
):
    """Find optimal learning rate using range test."""
    result = lr_find_from_config(config)
    typer.echo(f"Suggested LR: {result.suggested_lr:.2e}")
    typer.echo(f"Steepest descent: {result.steepest_lr:.2e}")
    typer.echo(f"Min loss: {result.min_loss_lr:.2e}")
    result.plot(plot)
    typer.echo(f"Plot saved: {plot}")


@app.command("train")
def train_cmd(
    config: str = typer.Argument(..., help="Path to experiment YAML"),
):
    """Train a model from config."""
    import subprocess
    import sys

    subprocess.run([sys.executable, "-m", "foundry.train", config])  # noqa: S603


@app.command("eval")
def eval_cmd(
    checkpoint: str = typer.Argument(..., help="Path to checkpoint"),
    task: str = typer.Option("constitution", "--task", "-t", help="Benchmark task"),
    samples: int = typer.Option(100, "--samples", "-n", help="Max samples"),
):
    """Run evaluation benchmarks."""
    import subprocess
    import sys

    subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "foundry.eval",
            checkpoint,
            "--task",
            task,
            "--samples",
            str(samples),
        ]
    )


@app.command("health")
def health_cmd():
    """Check repo health: CI, training sanity, experiments, data."""
    from foundry import health

    health.cli()


def main():
    app()


if __name__ == "__main__":
    main()
