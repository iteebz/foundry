#!/usr/bin/env python3
"""Comparison harness for baseline vs mutation A/B testing.

Runs two models (baseline + mutation) on same dataset, logs loss curves, computes winner.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_training(model: str, dataset: str, iters: int, out_dir: str, extra_args: list[str]) -> dict:
    """Run training and extract final metrics."""
    cmd = [
        sys.executable, "src/train.py",
        f"--model={model}",
        f"--dataset={dataset}",
        f"--max_iters={iters}",
        f"--out_dir={out_dir}",
        "--eval_interval=500",
        "--always_save_checkpoint=False",
        "--compile=False",
        *extra_args
    ]
    
    print(f"\n{'='*60}")
    print(f"Training {model} on {dataset} for {iters} iterations...")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Training failed for {model}")
        print(result.stderr)
        sys.exit(1)
    
    # Parse final validation loss from output
    lines = result.stdout.strip().split('\n')
    val_loss = None
    train_loss = None
    
    for line in reversed(lines):
        if 'val loss' in line and val_loss is None:
            parts = line.split()
            for i, part in enumerate(parts):
                if part == 'loss' and i > 0:
                    try:
                        val_loss = float(parts[i-1])
                        break
                    except ValueError:
                        continue
        if 'train loss' in line and train_loss is None:
            parts = line.split()
            for i, part in enumerate(parts):
                if 'loss' in part and i > 0:
                    try:
                        train_loss = float(parts[i+1].rstrip(':,'))
                        break
                    except (ValueError, IndexError):
                        continue
    
    return {
        "model": model,
        "val_loss": val_loss,
        "train_loss": train_loss,
        "stdout": result.stdout,
        "stderr": result.stderr
    }


def compare(baseline: str, mutation: str, dataset: str, iters: int) -> None:
    """Run comparison and report winner."""
    baseline_result = run_training(baseline, dataset, iters, f"out/{baseline}", [])
    mutation_result = run_training(mutation, dataset, iters, f"out/{mutation}", [])
    
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}\n")
    
    print(f"Baseline ({baseline}):")
    print(f"  Train Loss: {baseline_result['train_loss']:.4f}" if baseline_result['train_loss'] else "  Train Loss: N/A")
    print(f"  Val Loss:   {baseline_result['val_loss']:.4f}" if baseline_result['val_loss'] else "  Val Loss: N/A")
    
    print(f"\nMutation ({mutation}):")
    print(f"  Train Loss: {mutation_result['train_loss']:.4f}" if mutation_result['train_loss'] else "  Train Loss: N/A")
    print(f"  Val Loss:   {mutation_result['val_loss']:.4f}" if mutation_result['val_loss'] else "  Val Loss: N/A")
    
    if baseline_result['val_loss'] and mutation_result['val_loss']:
        improvement = ((baseline_result['val_loss'] - mutation_result['val_loss']) / baseline_result['val_loss']) * 100
        
        print(f"\n{'='*60}")
        if mutation_result['val_loss'] < baseline_result['val_loss']:
            print(f"✅ WINNER: {mutation}")
            print(f"Improvement: {improvement:.2f}%")
        else:
            print(f"❌ LOSER: {mutation}")
            print(f"Regression: {abs(improvement):.2f}%")
        print(f"{'='*60}\n")
        
        # Write report
        report = {
            "baseline": baseline_result,
            "mutation": mutation_result,
            "winner": mutation if mutation_result['val_loss'] < baseline_result['val_loss'] else baseline,
            "improvement_pct": improvement
        }
        
        report_path = Path("out") / f"compare_{baseline}_vs_{mutation}.json"
        report_path.write_text(json.dumps(report, indent=2))
        print(f"Report written to: {report_path}")
    else:
        print("\n⚠️  Could not extract validation loss for comparison")


def main():
    parser = argparse.ArgumentParser(description="Compare two model architectures")
    parser.add_argument("--baseline", default="v1", help="Baseline model (default: v1)")
    parser.add_argument("--mutation", default="v2", help="Mutation model (default: v2)")
    parser.add_argument("--dataset", default="shakespeare_char", help="Dataset (default: shakespeare_char)")
    parser.add_argument("--iters", type=int, default=5000, help="Training iterations (default: 5000)")
    
    args = parser.parse_args()
    compare(args.baseline, args.mutation, args.dataset, args.iters)


if __name__ == "__main__":
    main()
