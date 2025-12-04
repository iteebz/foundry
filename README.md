# Foundry

Agent-modifiable nanoGPT training infrastructure.

## Quick Start

```bash
# Install dependencies
poetry install

# Train baseline
python src/train.py experiments/baseline.yaml

# Generate single mutation
python -m src.mutate attention gqa_2kv

# Compare baseline vs mutation
python compare.py experiments/baseline.yaml experiments/attn_gqa_2kv.yaml

# Run parallel sweep (autonomous iteration)
python sweep.py attention gqa_2kv gqa_1kv mha --promote
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
│   └── baseline.yaml
├── src/
│   ├── model.py
│   ├── model_factory.py
│   ├── train.py
│   ├── mutate.py
│   └── modules/
│       ├── rope.py, alibi.py
│       ├── gqa.py
│       ├── rmsnorm.py, layernorm.py, qknorm.py
│       └── swiglu.py, gelu.py, glu.py
└── tests/
```

### Mutation Engine

Generate experiment configs programmatically:

```bash
# Attention
python -m src.mutate attention gqa_2kv
python -m src.mutate attention mha

# Architecture
python -m src.mutate depth 8
python -m src.mutate width 512

# Normalization
python -m src.mutate norm layernorm
python -m src.mutate norm rmsnorm

# Activation
python -m src.mutate activation gelu
python -m src.mutate activation glu

# Position encoding
python -m src.mutate position alibi
python -m src.mutate position rope

# Hyperparameters
python -m src.mutate lr 3e-4
```

Mutations saved to `experiments/*.yaml`, ready for training.

## Autonomous Iteration

The sweep runner enables fully autonomous architecture search:

```bash
# Generate + train + rank mutations in parallel
python sweep.py norm rmsnorm layernorm --jobs 8 --promote

# Auto-promotes winner to baseline.yaml
# Next sweep builds on the winner automatically
python sweep.py activation swiglu gelu glu --promote

# Iterate indefinitely
python sweep.py lr 3e-4 6e-4 1e-3 --promote
```

Each `--promote` replaces baseline with the winning mutation. The loop self-improves without human intervention.

## Model Zoo

Load pretrained models as starting points:

```bash
# Load model config (llama3, mistral, qwen2)
python -m src.zoo llama3-1b

# Use in training (after downloading checkpoint)
# Add to experiment YAML:
# init_from: "pretrained"
# checkpoint: "path/to/llama3.pt"
```

Available configs: llama3-8b, llama3-1b, mistral-7b, qwen2-7b

## LoRA Finetuning

Low-rank adaptation for efficient finetuning:

```bash
# Generate LoRA config
python -m src.mutate lora_rank 16

# Train with LoRA (freezes base, trains adapters)
python src/train.py experiments/lora_r16.yaml

# Sweep LoRA hyperparameters
python sweep.py lora_rank 4 8 16 32 64 --promote
```

LoRA reduces trainable params by 90-99%, enabling finetuning on limited hardware.

## Conversation Data

Format chat/instruction data for training:

```bash
# Generate conversation format mutation
python -m src.mutate conversation_format chatml

# Formats: chatml, llama3, alpaca
```

Supports ShareGPT and OpenAI message formats. See `src/data/conversation.py`.

### Mutation Surfaces

**Implemented:**
1. **Attention** - GQA (2kv, 1kv), MHA
2. **Architecture** - Depth/width scaling
3. **Normalization** - RMSNorm, LayerNorm, QKNorm
4. **Activation** - SwiGLU, GELU, GLU
5. **Position Encoding** - RoPE, ALiBi
6. **Loss** - CrossEntropy, Focal, LabelSmoothing
7. **Training** - LR, batch size, warmup, grad clip
8. **Optimizer** - Weight decay, Adam betas
9. **Data** - Filtering, dedupe, conversation formats
10. **LoRA** - Rank, alpha, dropout (finetuning)
11. **Model Zoo** - Load pretrained (llama3, mistral, qwen2)

**Future:**
- Advanced attention (MLA, sliding window, sparse)
- Curriculum learning
- Multi-GPU DDP validation
