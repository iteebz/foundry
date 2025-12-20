#!/usr/bin/env python3
"""Parallel mutation sweep runner.

Generate mutations, train in parallel, rank by validation loss or eval metrics.
"""

import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(add_completion=False)


class EvalTask(str, Enum):
    gsm8k = "gsm8k"
    mmlu = "mmlu"
    humaneval = "humaneval"
    constitution = "constitution"


def run_eval_on_checkpoint(checkpoint_path: Path, eval_task: str) -> float:
    """Run eval task on trained checkpoint."""
    cmd = [
        sys.executable,
        "-c",
        f"""import torch
from foundry.benchmarks.harness import run_benchmark_suite
from foundry.model import GPT
from foundry.data.tokenize import BPETokenizer
from pathlib import Path

checkpoint = torch.load('{checkpoint_path}', weights_only=False)
model = GPT(checkpoint['config'])
model.load_state_dict(checkpoint['model'])

meta_path = Path('{checkpoint_path}').parent / 'meta.pkl'
if meta_path.exists():
    tokenizer = BPETokenizer.load(meta_path)
else:
    tokenizer = BPETokenizer()

results = run_benchmark_suite(model, tokenizer, ['{eval_task}'], max_samples=50)
if '{eval_task}' in results and 'accuracy' in results['{eval_task}']:
    print(results['{eval_task}']['accuracy'])
elif '{eval_task}' in results and 'pass_at_1' in results['{eval_task}']:
    print(results['{eval_task}']['pass_at_1'])
else:
    print(0.0)
""",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # noqa: S603 - cmd is internal python
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:  # noqa: S110 - eval failures return 0.0
        pass

    return 0.0


def generate_mutation(mutation_type: str, variant: str) -> Path:
    """Generate mutation YAML."""
    if mutation_type == "adam_betas":
        beta1, beta2 = variant.split(",")
        cmd = [sys.executable, "-m", "foundry.mutate", mutation_type, beta1, beta2]
    else:
        cmd = [sys.executable, "-m", "foundry.mutate", mutation_type, variant]

    result = subprocess.run(cmd, capture_output=True, text=True)  # noqa: S603 - cmd is internal python module
    if result.returncode != 0:
        raise RuntimeError(f"Failed to generate {mutation_type} {variant}: {result.stderr}")

    for line in result.stdout.split("\n"):
        if line.startswith("Generated: "):
            return Path(line.split(": ")[1])

    raise RuntimeError(f"Could not find generated path in output: {result.stdout}")


def train_mutation(experiment_path: Path, eval_task: str | None = None) -> dict:
    """Train single mutation, return metrics."""
    from foundry.metrics import MetricLogger

    cmd = [sys.executable, "-m", "foundry.train", str(experiment_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)  # noqa: S603 - cmd is internal python module

    if result.returncode != 0:
        return {
            "experiment": experiment_path.stem,
            "status": "failed",
            "error": result.stderr,
        }

    out_dir = Path("out")
    logger = MetricLogger(str(out_dir))
    final_metrics = logger.get_final_metrics()

    if final_metrics and "val_loss" in final_metrics and "train_loss" in final_metrics:
        val_loss = final_metrics["val_loss"]
        train_loss = final_metrics["train_loss"]
    else:
        val_loss = None
        train_loss = None
        for line in reversed(result.stdout.strip().split("\n")):
            if "train loss" in line and "val loss" in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "loss" and i > 0 and parts[i - 1] == "train":
                        train_loss = float(parts[i + 1].rstrip(","))
                    if part == "loss" and i > 0 and parts[i - 1] == "val":
                        val_loss = float(parts[i + 1])
                if val_loss and train_loss:
                    break

    metrics = {
        "experiment": experiment_path.stem,
        "config_path": str(experiment_path),
        "val_loss": val_loss,
        "train_loss": train_loss,
        "status": "success",
    }

    if eval_task:
        checkpoint_path = Path("out") / experiment_path.stem / "ckpt.pt"
        if checkpoint_path.exists():
            eval_score = run_eval_on_checkpoint(checkpoint_path, eval_task)
            metrics[f"{eval_task}_score"] = eval_score

    return metrics


def run_sweep(
    mutation_type: str,
    variants: list[str],
    baseline_path: str,
    jobs: int = 4,
    promote: bool = False,
    eval_task: str | None = None,
) -> None:
    """Run parallel sweep of mutations."""
    typer.echo(f"Generating {len(variants)} {mutation_type} mutations...")

    mutation_paths = []
    for variant in variants:
        try:
            path = generate_mutation(mutation_type, variant)
            mutation_paths.append(path)
            typer.echo(f"  ✓ {path.stem}")
        except Exception as e:
            typer.echo(f"  ✗ Failed to generate {mutation_type} {variant}: {e}")

    typer.echo(f"\nTraining {len(mutation_paths)} mutations in parallel (jobs={jobs})...")

    results = []
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(train_mutation, path, eval_task): path for path in mutation_paths
        }

        for future in as_completed(futures):
            path = futures[future]
            try:
                result = future.result()
                results.append(result)

                if result["status"] == "success":
                    typer.echo(f"  ✓ {result['experiment']}: val_loss={result['val_loss']:.4f}")
                else:
                    typer.echo(
                        f"  ✗ {result['experiment']}: {result.get('error', 'unknown error')}"
                    )
            except Exception as e:
                typer.echo(f"  ✗ {path.stem}: {e}")

    successful = [r for r in results if r["status"] == "success" and r["val_loss"] is not None]

    if not successful:
        typer.echo("\n❌ No successful mutations")
        return

    rank_key = f"{eval_task}_score" if eval_task else "val_loss"
    reverse = eval_task is not None
    successful.sort(key=lambda x: x.get(rank_key, 0.0), reverse=reverse)

    metric_name = f"{eval_task} score" if eval_task else "val_loss"
    typer.echo(f"\n{'=' * 60}")
    typer.echo(f"SWEEP RESULTS (ranked by {metric_name})")
    typer.echo(f"{'=' * 60}\n")

    for i, result in enumerate(successful, 1):
        typer.echo(f"{i}. {result['experiment']}")
        if eval_task and f"{eval_task}_score" in result:
            typer.echo(f"   {eval_task}: {result[f'{eval_task}_score']:.4f}")
        typer.echo(f"   Val Loss: {result['val_loss']:.4f}")
        typer.echo(f"   Train Loss: {result['train_loss']:.4f}")

    winner = successful[0]
    typer.echo(f"\n{'=' * 60}")
    typer.echo(f"🏆 WINNER: {winner['experiment']}")
    if eval_task and f"{eval_task}_score" in winner:
        typer.echo(f"   {eval_task}: {winner[f'{eval_task}_score']:.4f}")
    typer.echo(f"   Val Loss: {winner['val_loss']:.4f}")
    typer.echo(f"{'=' * 60}\n")

    report = {
        "mutation_type": mutation_type,
        "variants": variants,
        "baseline": baseline_path,
        "results": successful,
        "winner": winner,
    }

    report_path = Path("out") / f"sweep_{mutation_type}.json"
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    typer.echo(f"Report: {report_path}")

    if promote:
        import shutil

        winner_path = Path(winner["config_path"])
        baseline_backup = Path(baseline_path).with_suffix(".yaml.backup")

        shutil.copy(baseline_path, baseline_backup)
        shutil.copy(winner_path, baseline_path)

        typer.echo(f"\n🚀 PROMOTED: {winner['experiment']} → {baseline_path}")
        typer.echo(f"   Backup: {baseline_backup}")


@app.command()
def sweep(
    mutation_type: Annotated[str, typer.Argument(help="Mutation type (attention, depth, etc)")],
    variants: Annotated[list[str], typer.Argument(help="Variants to test")],
    baseline: Annotated[str, typer.Option(help="Baseline config")] = "experiments/baseline.yaml",
    jobs: Annotated[int, typer.Option(help="Parallel jobs")] = 4,
    promote: Annotated[bool, typer.Option(help="Auto-promote winner to baseline")] = False,
    eval_task: Annotated[
        EvalTask | None, typer.Option(help="Rank by eval task instead of val_loss")
    ] = None,
):
    """Run parallel sweep of mutations."""
    run_sweep(mutation_type, variants, baseline, jobs, promote, eval_task)


if __name__ == "__main__":
    app()
