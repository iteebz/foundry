# Architecture

Foundry is agent-modifiable ML training infrastructure. 1586 LOC. Zero ceremony.

## Core Loop

```
mutate → train → compare → promote → iterate
```

Agents propose mutations, train in parallel, promote winners autonomously.

## Components

### Training (`src/train.py` - 293 LOC)
- Single GPU + DDP support
- YAML config driven
- EMA, gradient clipping, cosine LR decay
- Outputs: checkpoints, final eval metrics

### Model (`src/model.py` - 235 LOC)
- GPT architecture with modular components
- Configurable: attention, norm, activation, position encoding, loss
- 11 swappable modules in `src/modules/`

### Mutation Engine (`src/mutate.py` - 391 LOC)
- 14 mutation types
- Generates YAML configs from baseline
- CLI: `python -m src.mutate <type> <variant>`

### Sweep Runner (`sweep.py` - 161 LOC)
- Parallel training with ProcessPoolExecutor
- Ranks mutations by validation loss
- `--promote`: auto-replace baseline with winner
- Enables autonomous iteration

### Comparison (`compare.py` - 109 LOC)
- A/B testing harness
- Extracts metrics, reports winner
- JSON output for agent consumption

## Mutation Coverage

- **Attention**: GQA (1kv, 2kv), MHA
- **Architecture**: Depth, width scaling
- **Normalization**: RMSNorm, LayerNorm, QKNorm
- **Activation**: SwiGLU, GELU, GLU
- **Position Encoding**: RoPE, ALiBi
- **Loss**: CrossEntropy, Focal, LabelSmoothing
- **Training**: LR, batch size, warmup, grad clip
- **Optimizer**: Weight decay, Adam betas
- **Data**: Filtering, dedupe

## Data Pipeline (`src/data/`)

- `tokenize.py`: CharTokenizer (minimal, serializable)
- `pack.py`: Binary packing for train/val splits
- `filter.py`: Dedupe, length filtering

## Design Principles

1. **Reference grade only** - No scaffolding, no ceremony
2. **Agent-parseable** - All files <400 LOC, clear structure
3. **Composable** - Mutations stack, configs override
4. **Autonomous** - Sweep runner closes the loop

## Testing

59 tests covering:
- All mutation types
- Model components
- Data pipeline
- Integration (checkpoint, generate)

All tests pass. No flakes.

## Usage

```bash
# Single mutation
python -m src.mutate attention gqa_2kv
python src/train.py experiments/attn_gqa_2kv.yaml

# Autonomous sweep
python sweep.py norm rmsnorm layernorm --promote --jobs 8

# Iterate indefinitely
while true; do
  python sweep.py activation swiglu gelu glu --promote
  python sweep.py lr 3e-4 6e-4 1e-3 --promote
done
```

## What Makes This Different

**nanoGPT**: Human-readable training code  
**Foundry**: Agent-iterable mutation infrastructure

Karpathy optimized for pedagogy. We optimized for recursion.
