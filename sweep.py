#!/usr/bin/env python3
"""Parallel mutation sweep runner.

Generate mutations, train in parallel, rank by validation loss or eval metrics.
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def run_eval_on_checkpoint(checkpoint_path: Path, eval_task: str) -> float:
    """Run eval task on trained checkpoint."""
    cmd = [
        sys.executable,
        "-c",
        f"""import torch
import sys
from pathlib import Path
sys.path.insert(0, 'src')
from benchmarks.harness import run_benchmark_suite
from model import GPT
from data.tokenize import CharTokenizer

checkpoint = torch.load('{checkpoint_path}')
model = GPT(checkpoint['config'])
model.load_state_dict(checkpoint['model'])
tokenizer = CharTokenizer()

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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:
        pass

    return 0.0


def generate_mutation(mutation_type: str, variant: str) -> Path:
    """Generate mutation YAML."""
    if mutation_type == "adam_betas":
        beta1, beta2 = variant.split(",")
        cmd = [sys.executable, "-m", "src.mutate", mutation_type, beta1, beta2]
    else:
        cmd = [sys.executable, "-m", "src.mutate", mutation_type, variant]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to generate {mutation_type} {variant}: {result.stderr}")

    for line in result.stdout.split("\n"):
        if line.startswith("Generated: "):
            return Path(line.split(": ")[1])

    raise RuntimeError(f"Could not find generated path in output: {result.stdout}")


def train_mutation(experiment_path: Path, eval_task: str | None = None) -> dict:
    """Train single mutation, return metrics."""
    cmd = [sys.executable, "src/train.py", str(experiment_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        return {
            "experiment": experiment_path.stem,
            "status": "failed",
            "error": result.stderr,
        }

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


def sweep(
    mutation_type: str,
    variants: list[str],
    baseline_path: str,
    jobs: int = 4,
    promote: bool = False,
    eval_task: str | None = None,
) -> None:
    """Run parallel sweep of mutations."""
    print(f"Generating {len(variants)} {mutation_type} mutations...")

    mutation_paths = []
    for variant in variants:
        try:
            path = generate_mutation(mutation_type, variant)
            mutation_paths.append(path)
            print(f"  ✓ {path.stem}")
        except Exception as e:
            print(f"  ✗ Failed to generate {mutation_type} {variant}: {e}")

    print(f"\nTraining {len(mutation_paths)} mutations in parallel (jobs={jobs})...")

    results = []
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(train_mutation, path, eval_task): path
            for path in mutation_paths
        }

        for future in as_completed(futures):
            path = futures[future]
            try:
                result = future.result()
                results.append(result)

                if result["status"] == "success":
                    print(f"  ✓ {result['experiment']}: val_loss={result['val_loss']:.4f}")
                else:
                    print(f"  ✗ {result['experiment']}: {result.get('error', 'unknown error')}")
            except Exception as e:
                print(f"  ✗ {path.stem}: {e}")

    successful = [r for r in results if r["status"] == "success" and r["val_loss"] is not None]

    if not successful:
        print("\n❌ No successful mutations")
        return

    rank_key = f"{eval_task}_score" if eval_task else "val_loss"
    reverse = eval_task is not None
    successful.sort(key=lambda x: x.get(rank_key, 0.0), reverse=reverse)

    metric_name = f"{eval_task} score" if eval_task else "val_loss"
    print(f"\n{'=' * 60}")
    print(f"SWEEP RESULTS (ranked by {metric_name})")
    print(f"{'=' * 60}\n")

    for i, result in enumerate(successful, 1):
        print(f"{i}. {result['experiment']}")
        if eval_task and f"{eval_task}_score" in result:
            print(f"   {eval_task}: {result[f'{eval_task}_score']:.4f}")
        print(f"   Val Loss: {result['val_loss']:.4f}")
        print(f"   Train Loss: {result['train_loss']:.4f}")

    winner = successful[0]
    print(f"\n{'=' * 60}")
    print(f"🏆 WINNER: {winner['experiment']}")
    if eval_task and f"{eval_task}_score" in winner:
        print(f"   {eval_task}: {winner[f'{eval_task}_score']:.4f}")
    print(f"   Val Loss: {winner['val_loss']:.4f}")
    print(f"{'=' * 60}\n")

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
    print(f"Report: {report_path}")

    if promote:
        import shutil

        winner_path = Path(winner["config_path"])
        baseline_backup = Path(baseline_path).with_suffix(".yaml.backup")

        shutil.copy(baseline_path, baseline_backup)
        shutil.copy(winner_path, baseline_path)

        print(f"\n🚀 PROMOTED: {winner['experiment']} → {baseline_path}")
        print(f"   Backup: {baseline_backup}")


def main():
    parser = argparse.ArgumentParser(description="Parallel mutation sweep")
    parser.add_argument("mutation_type", help="Mutation type (attention, depth, etc)")
    parser.add_argument("variants", nargs="+", help="Variants to test")
    parser.add_argument("--baseline", default="experiments/baseline.yaml", help="Baseline config")
    parser.add_argument("--jobs", type=int, default=4, help="Parallel jobs")
    parser.add_argument("--promote", action="store_true", help="Auto-promote winner to baseline")
    parser.add_argument(
        "--eval-task",
        choices=["gsm8k", "mmlu", "humaneval", "constitution"],
        help="Rank by eval task instead of val_loss",
    )

    args = parser.parse_args()
    sweep(
        args.mutation_type,
        args.variants,
        args.baseline,
        args.jobs,
        args.promote,
        args.eval_task,
    )


if __name__ == "__main__":
    main()
