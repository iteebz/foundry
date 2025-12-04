# Foundry

Agent-modifiable nanoGPT training infrastructure.

## Quick Start

```bash
# Install dependencies
poetry install

# Train baseline
python src/train.py experiments/baseline.yaml

# Generate mutations
python -m src.mutate attention gqa_2kv
python -m src.mutate depth 8
python -m src.mutate width 512

# Compare baseline vs mutation
python compare.py experiments/baseline.yaml experiments/attn_gqa_2kv.yaml
```

## Mutation Framework

Experiments are YAML configs that specify model architecture and training parameters:

```yaml
name: "baseline"

training:
  max_iters: 5000
  learning_rate: 6e-4
  dataset: "shakespeare_char"

model_args:
  n_layer: 6
  n_head: 6
  n_kv_head: 2
  n_embd: 384
```

Flow: `experiments/*.yaml` → `train.py`

Usage: `python src/train.py experiments/baseline.yaml`

## Architecture

```
foundry/
├── experiments/
│   ├── baseline.yaml
│   └── modern.yaml
├── src/
│   ├── model.py        # GPT (RoPE+GQA+RMSNorm+SwiGLU)
│   ├── model_factory.py
│   ├── train.py
│   └── modules/
│       ├── rope.py
│       ├── gqa.py
│       ├── rmsnorm.py
│       └── swiglu.py
└── tests/
```

### Mutation Engine

Generate experiment configs programmatically:

```bash
# Attention variants
python -m src.mutate attention gqa_2kv  # 2 KV heads
python -m src.mutate attention gqa_1kv  # 1 KV head (near-MQA)
python -m src.mutate attention mha      # Multi-head (baseline)

# Architecture scaling
python -m src.mutate depth 8            # 8 layers
python -m src.mutate width 512          # 512 embedding dim

# Training hyperparameters
python -m src.mutate lr 3e-4            # Learning rate
```

Mutations saved to `experiments/*.yaml`, ready for training.

### Mutation Surfaces

1. **Attention** - GQA, MHA (implemented), MLA, sliding window, sparse
2. **Depth/Width** - Layer/embedding scaling (implemented)
3. **Hyperparameters** - LR, batch size, warmup (LR implemented)
4. **Position Encoding** - RoPE (current), ALiBi, learned
5. **Activations** - SwiGLU (current), GELU, GLU variants
6. **Normalization** - RMSNorm (current), LayerNorm, QKNorm
7. **Loss** - CrossEntropy (current), focal, label smoothing
8. **Data Pipeline** - Curriculum, filtering, packing
