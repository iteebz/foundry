# Foundry

Autonomous ML training infrastructure. 4,100 LOC. Zero ceremony.

## Quick Start

```bash
# Train baseline
python -m foundry.train experiments/baseline.yaml

# Generate mutation
python -m foundry.mutate attention mla

# Autonomous sweep
python -m foundry.cli.sweep attention mla gqa_2kv --eval-task gsm8k --promote
```

## The Loop

```
mutate → train → evaluate → promote → repeat
```

Agents propose mutations, train in parallel, rank by capability, promote winners autonomously.

## Mutations

26 mutation types across architecture, training, data:

**Architecture:**
- Attention: GQA, MLA, MoE, sliding window, sparse
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
- Curriculum learning (length, perplexity)
- Conversation formats (ChatML, Llama3, Alpaca)
- Constitution injection (preference pairs)
- Filtering, dedupe

## Eval Harness

Rank mutations by capability, not loss:

```bash
python -m foundry.cli.sweep norm rmsnorm layernorm \\
  --eval-task gsm8k --promote --jobs 4
```

Tasks: GSM8K (math), MMLU (knowledge), HumanEval (code), Constitution (alignment)

## Model Zoo

Start from pretrained:

```python
from foundry.zoo import load_pretrained

model = load_pretrained("llama3-1b", device="cuda")
```

Configs: llama3-8b, llama3-1b, mistral-7b, qwen2-7b

## Distributed Training

Auto-detects and configures distributed training:

**Single GPU:**
```bash
python -m foundry.train experiments/baseline.yaml
```

**Multi-GPU (auto-selects DDP or FSDP based on model size):**
```bash
torchrun --nproc_per_node=4 -m foundry.train experiments/baseline.yaml
```

**Manual override:**
```yaml
# experiments/baseline.yaml
training:
  distributed: "ddp"  # or "fsdp", "auto", "none"
  fsdp_min_params: 1000000000  # Use FSDP for models >1B params
```

**Auto-selection logic:**
- 1 GPU/CPU → No wrapping (zero overhead)
- Multi-GPU + <1B params → DDP
- Multi-GPU + ≥1B params → FSDP
- Multi-CPU → DDP with gloo (dev/testing only, slow)

## LoRA Finetuning

90-99% param reduction:

```bash
python -m foundry.mutate lora_rank 16
python -m foundry.train experiments/lora_r16.yaml
```

## Tokenizers

- **CharTokenizer** - Character-level (legacy, toy datasets)
- **BPETokenizer** - Production (byte-pair encoding, configurable vocab_size)

```python
from foundry.data.tokenize import BPETokenizer

tok = BPETokenizer(vocab_size=50257)
tok.fit(corpus)
ids = tok.encode("hello world")
```

## Structure

```
foundry/
├── foundry/          # Package
│   ├── model.py      # GPT (361 LOC)
│   ├── train.py      # Training loop (357 LOC)
│   ├── modules/      # Swappable components
│   ├── data/         # Tokenize, curriculum, pack
│   ├── mutate/       # Config generation
│   ├── benchmarks/   # Eval tasks
│   └── cli/          # sweep, compare, lr_finder
├── tests/            # 181 tests
└── experiments/      # YAML configs
```

## Design

1. **Reference grade** - All files <450 LOC, modules <75 LOC
2. **Agent-parseable** - Zero ceremony, clear structure
3. **Composable** - Mutations stack, configs override
4. **Autonomous** - Sweep runner closes the loop

See [docs/architecture.md](docs/architecture.md) for details.

## What Makes This Different

**nanoGPT** - Human-readable training code  
**Foundry** - Agent-iterable mutation infrastructure

Karpathy optimized for pedagogy. We optimized for recursion.
