# Foundry

Agent-modifiable nanoGPT training infrastructure.

## Quick Start

```bash
# Install dependencies
poetry install

# Train baseline model
python src/train.py experiments/baseline.yaml

# Compare baseline vs modern
python compare.py experiments/baseline.yaml experiments/modern.yaml
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

### Mutation Surfaces

1. **Attention** - MHA→GQA, MLA, sliding window, sparse
2. **Position Encoding** - RoPE, ALiBi, learned hybrids
3. **Activations** - GELU→SwiGLU, GLU variants
4. **Normalization** - LayerNorm→RMSNorm, QKNorm
5. **Loss** - CrossEntropy, focal, label smoothing
6. **LR Schedule** - Cosine, cyclic, adaptive
7. **Data Pipeline** - Curriculum, filtering, packing
