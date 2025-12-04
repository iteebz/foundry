# Architecture

Foundry is autonomous ML training infrastructure. 4,100 LOC. Zero ceremony.

## Core Loop

```
mutate → train → evaluate → promote → repeat
```

Agents propose mutations, train in parallel, rank by capability, promote winners autonomously.

## Components

### Model (`foundry/model.py` - 361 LOC)
- GPT architecture with swappable components
- Config validation prevents invalid combinations
- Modular: attention, norm, activation, position, loss
- 26 variants across `foundry/modules/`

### Training (`foundry/train.py` - 357 LOC)
- Single GPU + DDP support
- YAML config driven
- EMA, gradient clipping, cosine LR decay
- Outputs: checkpoints, eval metrics

### Mutation Engine (`foundry/mutate/` - 580 LOC)
- 26 mutation types
- Generates YAML configs from baseline
- CLI: `python -m foundry.mutate <type> <variant>`
- Mutations: architecture, training, data

### Sweep Runner (`foundry/cli/sweep.py` - 234 LOC)
- Parallel training with ProcessPoolExecutor
- Ranks by validation loss OR eval task
- `--promote`: auto-replace baseline with winner
- Enables autonomous iteration

### Eval Harness (`foundry/benchmarks/` - 367 LOC)
- GSM8K (math reasoning)
- MMLU (knowledge)
- HumanEval (code generation)
- Constitution (alignment via preference pairs)

### Tokenizers (`foundry/data/tokenize.py` - 139 LOC)
- CharTokenizer: toy datasets
- BPETokenizer: production (byte-pair encoding)
- Both save/load from disk

### Model Zoo (`foundry/zoo.py` - 161 LOC)
- Load pretrained configs (llama3, mistral, qwen2)
- Convert HuggingFace → Foundry format
- Export checkpoints

## Mutation Coverage

**Architecture:**
- Attention: GQA (1kv, 2kv), MHA, MLA, MoE, sliding window, sparse
- Depth/width scaling
- Norm: RMSNorm, LayerNorm
- Activation: SwiGLU, GELU, GLU
- Position: RoPE, ALiBi
- Loss: CrossEntropy, Focal, LabelSmoothing, DPO

**Training:**
- LR, batch size, warmup, grad clip
- Weight decay, Adam betas
- LoRA rank/alpha/dropout

**Data:**
- Curriculum learning (length-based, perplexity-based)
- Conversation formats (ChatML, Llama3, Alpaca)
- Constitution injection (preference pairs)
- Filtering (length, dedupe)

## Design Principles

1. **Reference grade** - All files <450 LOC, modules <75 LOC
2. **Agent-parseable** - Zero ceremony, clear structure
3. **Composable** - Mutations stack, configs override
4. **Autonomous** - Sweep runner closes the loop
5. **Beautiful code reads like English** - Semantic naming, helpers over conditionals

## Testing

181 tests covering:
- All 26 mutation types
- Model components (MLA, MoE, sparse attention)
- Data pipeline (tokenizers, curriculum, conversation)
- Eval harness (GSM8K, MMLU, HumanEval, constitution)
- Config validation
- Integration (checkpoint, generate, sweep)

All tests pass. No flakes.

## Usage

```bash
# Single mutation
python -m foundry.mutate attention mla
python -m foundry.train experiments/attn_mla.yaml

# Autonomous sweep (capability-ranked)
python -m foundry.cli.sweep attention mla gqa_2kv \\
  --eval-task gsm8k --promote --jobs 4

# Iterate indefinitely
while true; do
  python -m foundry.cli.sweep norm rmsnorm layernorm --promote
  python -m foundry.cli.sweep activation swiglu gelu glu --promote
  python -m foundry.cli.sweep lr 3e-4 6e-4 1e-3 --promote
done
```

## What Makes This Different

**nanoGPT** - Human-readable training code  
**Foundry** - Agent-iterable mutation infrastructure

Karpathy optimized for pedagogy. We optimized for recursion.

The edge isn't in PyTorch code. It's in:
- Novel architectures agents discover through mutation
- Training dynamics agents optimize through iteration
- Data pipelines agents instrument and modify
- Eval loops agents tighten

DeepSeek found MLA, MoE routing gains, training efficiency tricks through infrastructure control. Foundry gives agents that same surface area.
