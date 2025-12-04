#!/usr/bin/env python3
"""Parallel mutation sweep runner.

Generate mutations, train in parallel, rank by validation loss.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


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
    
    for line in result.stdout.split('\n'):
        if line.startswith("Generated: "):
            return Path(line.split(": ")[1])
    
    raise RuntimeError(f"Could not find generated path in output: {result.stdout}")


def train_mutation(experiment_path: Path) -> dict:
    """Train single mutation, return metrics."""
    cmd = [sys.executable, "src/train.py", str(experiment_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        return {
            "experiment": experiment_path.stem,
            "status": "failed",
            "error": result.stderr
        }
    
    val_loss = None
    train_loss = None
    
    for line in reversed(result.stdout.strip().split('\n')):
        if "train loss" in line and "val loss" in line:
            parts = line.split()
            for i, part in enumerate(parts):
                if part == "loss" and i > 0 and parts[i-1] == "train":
                    train_loss = float(parts[i+1].rstrip(','))
                if part == "loss" and i > 0 and parts[i-1] == "val":
                    val_loss = float(parts[i+1])
            if val_loss and train_loss:
                break
    
    return {
        "experiment": experiment_path.stem,
        "config_path": str(experiment_path),
        "val_loss": val_loss,
        "train_loss": train_loss,
        "status": "success"
    }


def sweep(mutation_type: str, variants: list[str], baseline_path: str, jobs: int = 4) -> None:
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
        futures = {executor.submit(train_mutation, path): path for path in mutation_paths}
        
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
    
    successful.sort(key=lambda x: x["val_loss"])
    
    print(f"\n{'='*60}")
    print("SWEEP RESULTS (ranked by val_loss)")
    print(f"{'='*60}\n")
    
    for i, result in enumerate(successful, 1):
        print(f"{i}. {result['experiment']}")
        print(f"   Val Loss: {result['val_loss']:.4f}")
        print(f"   Train Loss: {result['train_loss']:.4f}")
    
    winner = successful[0]
    print(f"\n{'='*60}")
    print(f"🏆 WINNER: {winner['experiment']}")
    print(f"   Val Loss: {winner['val_loss']:.4f}")
    print(f"{'='*60}\n")
    
    report = {
        "mutation_type": mutation_type,
        "variants": variants,
        "baseline": baseline_path,
        "results": successful,
        "winner": winner
    }
    
    report_path = Path("out") / f"sweep_{mutation_type}.json"
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Parallel mutation sweep")
    parser.add_argument("mutation_type", help="Mutation type (attention, depth, etc)")
    parser.add_argument("variants", nargs="+", help="Variants to test")
    parser.add_argument("--baseline", default="experiments/baseline.yaml", help="Baseline config")
    parser.add_argument("--jobs", type=int, default=4, help="Parallel jobs")
    
    args = parser.parse_args()
    sweep(args.mutation_type, args.variants, args.baseline, args.jobs)


if __name__ == "__main__":
    main()
