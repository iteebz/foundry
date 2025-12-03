# Foundry

Agent-modifiable nanoGPT training infrastructure.

## Quick Start

```bash
# Install dependencies
poetry install

# Train baseline model
python src/train.py --experiment=experiments/baseline.yaml

# Compare baseline vs mutation
python compare.py v1 v2 shakespeare_char --iters=5000
```

## Mutation Framework

Experiments are YAML configs that specify model architecture and training parameters:

```yaml
name: "baseline_v1"
base_model: "v1"  # or "v2"

training:
  max_iters: 5000
  learning_rate: 6e-4
  dataset: "shakespeare_char"

model_args:
  n_layer: 6
  n_head: 6
  n_embd: 384
```

Flow: `experiments/*.yaml` → `model_factory.py` → `train.py`

## Architecture

```
foundry/
├── docs/experiments/oplot-foundry.md
├── experiments/
│   ├── baseline.yaml   # v1 config
│   └── modern.yaml     # v2 config
├── src/
│   ├── model.py        # nanoGPT base (330 lines)
│   ├── model_v2.py     # modern modules integrated
│   ├── model_factory.py # YAML → model loader
│   ├── train.py        # nanoGPT training (--experiment flag)
│   └── modules/
│       ├── rope.py     # RoPE (52 lines)
│       ├── gqa.py      # GQA (44 lines)
│       ├── rmsnorm.py  # RMSNorm (20 lines)
│       └── swiglu.py   # SwiGLU (23 lines)
└── tests/
    ├── test_modules.py
    └── test_integration.py  # 6 tests passing
```

### Mutation Surfaces

1. **Attention** - MHA→GQA, MLA, sliding window, sparse
2. **Position Encoding** - RoPE, ALiBi, learned hybrids
3. **Activations** - GELU→SwiGLU, GLU variants
4. **Normalization** - LayerNorm→RMSNorm, QKNorm
5. **Loss** - CrossEntropy, focal, label smoothing
6. **LR Schedule** - Cosine, cyclic, adaptive
7. **Data Pipeline** - Curriculum, filtering, packing
