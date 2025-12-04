#!/usr/bin/env python3
"""Comparison harness for baseline vs mutation A/B testing.

Runs two experiment configs, compares results.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def run_experiment(experiment_path: str, out_dir: str) -> dict:
    """Run experiment YAML and extract final metrics."""
    cmd = [sys.executable, "src/train.py", experiment_path]

    experiment_name = Path(experiment_path).stem
    print(f"\n{'=' * 60}")
    print(f"Running experiment: {experiment_name}")
    print(f"Config: {experiment_path}")
    print(f"{'=' * 60}\n")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ Experiment failed: {experiment_name}")
        print(result.stderr)
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


def compare(baseline_path: str, mutation_path: str) -> None:
    """Run comparison and report winner."""
    baseline_result = run_experiment(baseline_path, f"out/compare_{Path(baseline_path).stem}")
    mutation_result = run_experiment(mutation_path, f"out/compare_{Path(mutation_path).stem}")

    print(f"\n{'=' * 60}")
    print("COMPARISON RESULTS")
    print(f"{'=' * 60}\n")

    print(f"Baseline ({baseline_result['experiment']}):")
    print(
        f"  Train Loss: {baseline_result['train_loss']:.4f}"
        if baseline_result["train_loss"]
        else "  Train Loss: N/A"
    )
    print(
        f"  Val Loss:   {baseline_result['val_loss']:.4f}"
        if baseline_result["val_loss"]
        else "  Val Loss: N/A"
    )

    print(f"\nMutation ({mutation_result['experiment']}):")
    print(
        f"  Train Loss: {mutation_result['train_loss']:.4f}"
        if mutation_result["train_loss"]
        else "  Train Loss: N/A"
    )
    print(
        f"  Val Loss:   {mutation_result['val_loss']:.4f}"
        if mutation_result["val_loss"]
        else "  Val Loss: N/A"
    )

    if baseline_result["val_loss"] and mutation_result["val_loss"]:
        improvement = (
            (baseline_result["val_loss"] - mutation_result["val_loss"])
            / baseline_result["val_loss"]
        ) * 100

        print(f"\n{'=' * 60}")
        if mutation_result["val_loss"] < baseline_result["val_loss"]:
            print(f"✅ WINNER: {mutation_result['experiment']}")
            print(f"Improvement: {improvement:.2f}%")
        else:
            print(f"❌ LOSER: {mutation_result['experiment']}")
            print(f"Regression: {abs(improvement):.2f}%")
        print(f"{'=' * 60}\n")

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
        print(f"Report written to: {report_path}")
    else:
        print("\n⚠️  Could not extract validation loss for comparison")


def main():
    parser = argparse.ArgumentParser(description="Compare two experiment configs")
    parser.add_argument("baseline", help="Path to baseline experiment YAML")
    parser.add_argument("mutation", help="Path to mutation experiment YAML")

    args = parser.parse_args()
    compare(args.baseline, args.mutation)


if __name__ == "__main__":
    main()
