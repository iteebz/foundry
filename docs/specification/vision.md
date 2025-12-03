# Foundry

Foundation model training infrastructure. Agent-modifiable, reference-grade.

## Why

HuggingFace abstracts control away. We want maximum surface area for agents to evolve training architecture itself—not just use it.

The edge isn't in PyTorch code. It's in:
- Novel architectures agents discover through mutation
- Training dynamics agents optimize through iteration
- Data pipelines agents can instrument and modify
- Eval loops agents can tighten

DeepSeek found MLA, MoE routing gains, training efficiency tricks. Those came from control over infrastructure. We want that surface area.

## Scope

### Training
- Raw PyTorch, no abstraction layers
- nanoGPT as pedagogical base (~300 lines, agent-parseable)
- Small model validation (CPU minutes, GPU seconds)
- Scale promising discoveries when GPU available

### Finetuning
- LoRA/QLoRA for efficient adaptation (16GB GPU viable)
- Full finetune path when resources permit
- Constitution/behavior injection into base models

### Architecture Evolution
- Attention patterns (MLA, GQA, sliding window, novel)
- Position encodings (RoPE, ALiBi, learned, novel)
- Activation functions, normalization schemes
- MoE routing, sparse attention, efficiency tricks
- Loss functions, curriculum strategies

### Data Pipelines
- Conversation → token formatting
- Quality filtering, deduplication
- Curriculum ordering
- Agent-generated synthetic data integration

### Eval Harness
- Measure what's actually learned
- Fast iteration feedback
- Benchmark comparisons
- Custom eval for constitution/behavior

### Model Zoo
- Accept publicly trained models (llama, mistral, qwen, phi)
- Train from scratch when warranted
- Export trained models for inference

## Constraints

- Reference-grade only. No scaffolding.
- Agent-modifiable. Every component agents can read/mutate.
- Small-model validation first. GPU access is not assumed.
- nanoGPT as ground truth. Karpathy's clarity as baseline.

## The Loop

```
agent proposes architecture mod → train small → eval → keep/discard → iterate
```

RSI applied to ML infrastructure. The flywheel pattern transfers.
