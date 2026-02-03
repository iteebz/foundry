# Methodology

## Research Question

**What architectural and training decisions maximize capability per training token?**

Sample efficiency is the limiting factor for small-scale transformer research. A 10% improvement in tokens-to-capability means 10% less compute to reach the same performance. Foundry answers this by systematic ablation: change one thing, measure everything.

## Experimental Design

### Controlled Ablation

Every experiment isolates a single variable. The mutation system enforces this:

```
baseline.yaml → mutate attention mla → mla.yaml (one delta)
```

Mutations are typed: architecture (17 types), training (8 types), data (2 types). Each mutation modifies exactly one aspect of the baseline while preserving all other settings.

### Fixed Seeds, Fixed Data

All experiments share:
- `seed: 1337` for reproducibility
- Same data order via deterministic dataloading
- Identical eval conditions (samples, temperature, device)

This eliminates variance from randomness, isolating architecture as the only free variable.

### Provenance Tracking

Every mutation records lineage:

```yaml
_metadata:
  parent_config: "experiments/baseline.yaml"
  mutation_type: "attention"
  variant: "mla"
  generation: 1
  timestamp: "2026-02-03T12:00:00"
```

Lineage enables multi-hop comparisons: if MLA beats GQA and GQA beats MHA, transitive inference is justified by shared ancestry.

## Metrics

### Primary: Capability Tasks

Loss is a proxy. Capabilities are the target.

| Task | Metric | Measures |
|------|--------|----------|
| GSM8K | accuracy | Math reasoning |
| MMLU | accuracy | Factual knowledge |
| HumanEval | pass@1 | Code generation |
| Constitution | preference_accuracy | Alignment |

Aggregate score: arithmetic mean across tasks. Simple, interpretable, no hyperparameters.

### Secondary: Training Dynamics

- **Val loss**: Per-iteration validation loss (overfitting detection)
- **Perplexity**: `exp(loss)` for interpretability
- **Train/val gap**: Generalization health

Loss curves matter for understanding *why* an architecture works, not for ranking.

### Ranking Logic

Sweeps rank by capability task when specified:

```bash
python -m foundry.cli.sweep attention mla gqa --eval-task gsm8k --promote
```

Without eval task, rank by val_loss. Higher capability score wins; lower loss wins.

## Baselines

### Reference Architecture

The baseline is the current champion:

```yaml
model_args:
  attention_type: "gqa"      # 2 KV heads
  norm_type: "rmsnorm"
  activation: "swiglu"
  position_encoding: "rope"
  n_layer: 6
  n_head: 6
  n_embd: 384
```

This represents consensus best practices (Llama-style) scaled to research-viable size. The baseline evolves as mutations win and get promoted.

### Scale Considerations

Baseline is deliberately small (~10M params). This choice trades ecological validity for iteration speed:

- **Pro**: Full sweep in minutes, not days
- **Con**: Some effects don't transfer to scale (e.g., MoE benefits may require >1B)

Foundry is designed for *finding candidates*, not proving scaling behavior. Candidates that win at small scale are hypotheses for larger-scale verification.

## Validity Conditions

### What Makes a Fair Comparison

1. **Same compute budget**: Identical `max_iters` and `batch_size`
2. **Same data exposure**: Identical dataset and order
3. **Same evaluation**: Same eval samples, same extraction logic
4. **Isolated change**: One mutation per config

Violations invalidate comparison. The mutation system prevents most violations by construction.

### What Foundry Cannot Answer

- **Scaling laws**: Behavior at 10M doesn't prove behavior at 70B
- **Data efficiency trade-offs**: Same data, so data quality effects are fixed
- **Training stability at scale**: Small models don't hit the same loss spikes
- **Inference optimization**: Foundry measures training efficiency only

These require larger-scale experiments outside Foundry's scope.

## Iteration Protocol

### Autonomous Loop

```
mutate → train → evaluate → promote → repeat
```

The `--promote` flag enables fully autonomous evolution: winner replaces baseline, next sweep uses new baseline as parent. No human in the loop until compute runs out.

### Convergence

The loop terminates when:
1. No mutation beats baseline (local optimum)
2. Improvement falls below threshold (diminishing returns)
3. Compute budget exhausted

In practice, 3-5 generations of sweeps typically exhaust easy wins.

## Statistical Considerations

### Single Runs

Current design uses single runs (no replication). This is a deliberate compute-efficiency trade-off:

- **Risk**: Variance obscures true effect
- **Mitigation**: Fixed seeds reduce variance; large effects are robust

For high-stakes comparisons, multiple seeds are recommended:

```bash
for seed in 1337 42 7; do
  python -m foundry.train experiments/mla.yaml --seed $seed
done
```

### Effect Size

Small differences (<1% accuracy) are noise. Report effects worth acting on:

- **Meaningful**: 5%+ accuracy delta
- **Marginal**: 2-5% (worth noting, not decisive)
- **Noise**: <2% (rerun with more seeds before claiming)

## Extending the Benchmark Suite

New eval tasks implement a simple interface:

```python
def evaluate_task(
    model, tokenizer, dataset_path, max_samples, device
) -> dict[str, Any]:
    # Must return dict with score key (e.g., "accuracy", "pass_at_1")
    return {"accuracy": correct / total, "correct": correct, "total": total}
```

Register in `harness.py`:

```python
task_map = {
    "gsm8k": ("gsm8k_test.jsonl", evaluate_gsm8k),
    "new_task": ("new_task.jsonl", evaluate_new_task),  # add here
}
```

Aggregate scoring auto-includes any task that returns `accuracy` or `pass_at_1`.
