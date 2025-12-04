# Architecture

## Core Loop

```
mutate → train → evaluate → promote → repeat
```

Mutation generator creates config variants. Sweep runner trains in parallel, ranks by eval task, promotes winners.

## Components

### Model (`foundry/model.py`)
- GPT architecture with swappable components
- Config validation prevents invalid combinations
- Modular: attention, norm, activation, position, loss
- Gradient checkpointing for memory efficiency

### Training (`foundry/train.py`)
- Auto-distributed: Single GPU / DDP / FSDP
- YAML config driven
- EMA, gradient clipping, cosine LR decay
- Checkpoint resume, eval metrics

### Distributed (`foundry/distributed.py`)
- Auto-detection from env vars (torchrun)
- Strategy: auto/ddp/fsdp/none
- DDP for <1B params, FSDP for ≥1B
- Zero overhead on single GPU

### Mutation Engine (`foundry/mutate/`)
- 21 mutation types
- Generates YAML configs from baseline
- CLI: `python -m foundry.mutate <type> <variant>`
- Architecture, training, data mutations

### Sweep Runner (`foundry/cli/sweep.py`)
- Parallel training with ProcessPoolExecutor
- Ranks by validation loss OR eval task
- `--promote`: auto-replace baseline with winner
- Enables autonomous iteration

### Eval Harness (`foundry/benchmarks/`)
- GSM8K (math reasoning)
- MMLU (knowledge)
- HumanEval (code generation)
- Constitution (alignment via preference pairs)

### Data Pipeline (`foundry/data/`)
- Tokenizers: BPE, char-level
- Curriculum: order by length/perplexity
- Synthetic: self-instruct generation
- Pack: greedy bin-packing
- Filter: dedupe, length
- Conversation formats: ChatML, Llama3, Alpaca
- Constitution injection
- Preference pairs (DPO)

### Model Zoo (`foundry/zoo.py`)
- Load pretrained configs (llama3, mistral, qwen2)
- Convert HuggingFace → Foundry format
- Export checkpoints

### CLI Tools (`foundry/cli/`)
- sweep: parallel mutation training
- compare: A/B test baseline vs mutation
- lr_finder: learning rate range test

## Mutations

21 mutation types. 17 architecture modules.

**Architecture:**
- Attention: GQA (1kv, 2kv), MHA, MLA, MoE, sliding window, sparse
- Depth/width scaling
- Norm: RMSNorm, LayerNorm, QKNorm
- Activation: SwiGLU, GELU, GLU
- Position: RoPE, ALiBi
- Loss: CrossEntropy, Focal, LabelSmoothing, DPO

**Training:**
- LR, batch size, warmup, grad clip
- Weight decay, Adam betas
- LoRA rank/alpha/dropout

**Data:**
- Conversation formats: ChatML, Llama3, Alpaca

Modules: `foundry/modules/{alibi,dpo_loss,focal_loss,gelu,glu,gqa,label_smoothing,layernorm,mla,moe,qknorm,rmsnorm,rope,sliding_window,sparse_attention,swiglu}.py`

## Design Principles

1. **Reference grade** - All files <450 LOC, modules <75 LOC
2. **Agent-parseable** - Zero ceremony, clear structure
3. **Composable** - Mutations stack, configs override
4. **Autonomous** - Sweep runner closes the loop
5. **Beautiful code reads like English** - Semantic naming, helpers over conditionals

## Testing

Comprehensive test coverage:
- All 21 mutation types
- Model components (MLA, MoE, sparse attention)
- Data pipeline (tokenizers, curriculum, synthetic, pack, filter)
- Eval harness (GSM8K, MMLU, HumanEval, constitution)
- Config validation
- Integration (checkpoint, generate, sweep)

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

