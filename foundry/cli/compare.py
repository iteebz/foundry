#!/usr/bin/env python3
"""Comparison harness for baseline vs mutation A/B testing."""

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(add_completion=False)


def run_experiment(experiment_path: str, out_dir: str) -> dict:
    """Run experiment YAML and extract final metrics."""
    cmd = [sys.executable, "-m", "foundry.train", experiment_path]

    experiment_name = Path(experiment_path).stem
    typer.echo(f"\n{'=' * 60}")
    typer.echo(f"Running experiment: {experiment_name}")
    typer.echo(f"Config: {experiment_path}")
    typer.echo(f"{'=' * 60}\n")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        typer.echo(f"❌ Experiment failed: {experiment_name}")
        typer.echo(result.stderr)
        sys.exit(1)

    val_loss = None
    train_loss = None

    for line in reversed(result.stdout.strip().split("\n")):
        if val_loss is None or train_loss is None:
            match = re.search(r"train loss ([\d.]+).*val loss ([\d.]+)", line)
            if match:
                train_loss = float(match.group(1))
                val_loss = float(match.group(2))
                break

    return {
        "experiment": experiment_name,
        "config_path": experiment_path,
        "val_loss": val_loss,
        "train_loss": train_loss,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


@app.command()
def compare_experiments(
    baseline: Annotated[str, typer.Argument(help="Path to baseline experiment YAML")],
    mutation: Annotated[str, typer.Argument(help="Path to mutation experiment YAML")],
) -> None:
    """Run comparison and report winner."""
    baseline_result = run_experiment(baseline, f"out/compare_{Path(baseline).stem}")
    mutation_result = run_experiment(mutation, f"out/compare_{Path(mutation).stem}")

    typer.echo(f"\n{'=' * 60}")
    typer.echo("COMPARISON RESULTS")
    typer.echo(f"{'=' * 60}\n")

    typer.echo(f"Baseline ({baseline_result['experiment']}):")
    typer.echo(
        f"  Train Loss: {baseline_result['train_loss']:.4f}"
        if baseline_result["train_loss"]
        else "  Train Loss: N/A"
    )
    typer.echo(
        f"  Val Loss:   {baseline_result['val_loss']:.4f}"
        if baseline_result["val_loss"]
        else "  Val Loss: N/A"
    )

    typer.echo(f"\nMutation ({mutation_result['experiment']}):")
    typer.echo(
        f"  Train Loss: {mutation_result['train_loss']:.4f}"
        if mutation_result["train_loss"]
        else "  Train Loss: N/A"
    )
    typer.echo(
        f"  Val Loss:   {mutation_result['val_loss']:.4f}"
        if mutation_result["val_loss"]
        else "  Val Loss: N/A"
    )

    if baseline_result["val_loss"] and mutation_result["val_loss"]:
        improvement = (
            (baseline_result["val_loss"] - mutation_result["val_loss"])
            / baseline_result["val_loss"]
        ) * 100

        typer.echo(f"\n{'=' * 60}")
        if mutation_result["val_loss"] < baseline_result["val_loss"]:
            typer.echo(f"✅ WINNER: {mutation_result['experiment']}")
            typer.echo(f"Improvement: {improvement:.2f}%")
        else:
            typer.echo(f"❌ LOSER: {mutation_result['experiment']}")
            typer.echo(f"Regression: {abs(improvement):.2f}%")
        typer.echo(f"{'=' * 60}\n")

        report = {
            "baseline": baseline_result,
            "mutation": mutation_result,
            "winner": mutation_result["experiment"]
            if mutation_result["val_loss"] < baseline_result["val_loss"]
            else baseline_result["experiment"],
            "improvement_pct": improvement,
        }

        report_path = (
            Path("out")
            / f"compare_{baseline_result['experiment']}_vs_{mutation_result['experiment']}.json"
        )
        report_path.write_text(json.dumps(report, indent=2))
        typer.echo(f"Report written to: {report_path}")
    else:
        typer.echo("\n⚠️  Could not extract validation loss for comparison")


if __name__ == "__main__":
    app()
