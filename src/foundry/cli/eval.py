"""CLI for running benchmark evaluation suites."""

from typing import Annotated

import typer

app = typer.Typer(
    help="Evaluate model checkpoints against benchmarks", add_completion=False, no_args_is_help=True
)

VALID_TASKS = ["gsm8k", "mmlu", "humaneval", "constitution"]


@app.command()
def run(
    checkpoint: Annotated[str, typer.Argument(help="Path to model checkpoint directory")],
    tasks: Annotated[
        str, typer.Option("--tasks", "-t", help=f"Comma-separated tasks: {','.join(VALID_TASKS)}")
    ] = "gsm8k,mmlu,humaneval",
    max_samples: Annotated[
        int, typer.Option("--max-samples", "-n", help="Max samples per task")
    ] = 100,
    output: Annotated[
        str | None, typer.Option("--output", "-o", help="Save results to JSON path")
    ] = None,
    dataset_dir: Annotated[
        str | None, typer.Option("--dataset-dir", help="Local dataset dir (skips HuggingFace)")
    ] = None,
    device: Annotated[str, typer.Option("--device", help="Device: cpu|cuda|mps")] = "cpu",
):
    """Run benchmark suite against a checkpoint."""
    from foundry.benchmarks import run_benchmark_suite
    from foundry.checkpoint import load_checkpoint

    task_list = [t.strip() for t in tasks.split(",")]
    unknown = [t for t in task_list if t not in VALID_TASKS]
    if unknown:
        typer.echo(f"Unknown tasks: {unknown}. Valid: {VALID_TASKS}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Loading checkpoint: {checkpoint}")
    model, tokenizer = load_checkpoint(checkpoint, device=device)

    typer.echo(f"Running tasks: {task_list} (max_samples={max_samples})")
    use_hf = dataset_dir is None
    results = run_benchmark_suite(
        model=model,
        tokenizer=tokenizer,
        tasks=task_list,
        dataset_dir=dataset_dir,
        max_samples=max_samples,
        device=device,
        use_hf=use_hf,
    )

    agg = results.get("aggregate", {})
    typer.echo("\n── results ──")
    for task in task_list:
        r = results.get(task, {})
        if "error" in r:
            typer.echo(f"  {task}: ERROR — {r['error']}")
        else:
            score = r.get("accuracy") or r.get("pass_at_1") or r.get("preference_accuracy")
            typer.echo(f"  {task}: {score:.3f}" if score is not None else f"  {task}: {r}")
    typer.echo(
        f"  aggregate mean: {agg.get('mean_score', 0):.3f} ({agg.get('num_tasks', 0)} tasks)"
    )

    if output:
        from foundry.benchmarks.harness import save_eval_results

        save_eval_results(results, output)
        typer.echo(f"\nSaved: {output}")
