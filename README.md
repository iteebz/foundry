# Foundry

Mutation-based ML training framework. Built on nanoGPT in 21 hours.

**Dec 3–4, 2025.** One human steering one AI agent. 46 commits. Zero ML training experience from the human. The [commit log](../../commits/main) is timestamped.

**Jan–Feb 2026.** 7 autonomous agents returned to the repo independently — type compliance, benchmarks, test coverage, health scoring. No human steering. Visible in `git log --format="%an"`.

## What it does

```
mutate → train → rank → promote → repeat
```

21 mutation types across architecture (GQA/MQA/MHA, MLA, MoE, depth, width, norm, activation, position encoding, loss, sliding window, sparse attention), training (LR, batch size, warmup, grad clip, weight decay, Adam betas, LoRA), and data (conversation format, filtering).

## What's real

- **Training loop.** Cosine LR, AMP, gradient accumulation, EMA, checkpoint resume, wandb. CPU/MPS/CUDA.
- **Sweep runner.** Parallel training, ranking, auto-promote winner.
- **Mutation engine.** All 21 types generate valid configs. Tested.
- **Data pipeline.** BPE tokenizer, memory-mapped datasets, 8-filter quality pipeline, curriculum learning.

## What's not

- **Never trained at scale.** No GPUs were available. The loop runs, mechanics verified, no loss curves.
- **Eval harness untested at scale.** GSM8K, MMLU, HumanEval implementations exist, never evaluated a real checkpoint.
- **Model zoo is config-only.** No HuggingFace weight download.

## Structure

```
foundry/
├── model.py          # GPT with swappable components (383 lines)
├── train.py          # Training loop (501 lines)
├── mutate/           # 21 mutation generators
├── modules/          # 16 architecture components
├── data/             # Tokenize, filter, curriculum, mixture
├── benchmarks/       # GSM8K, MMLU, HumanEval
├── cli/              # sweep, compare
├── distributed.py    # DDP/FSDP auto-selection
├── lora.py           # LoRA adapters
└── config.py         # RunConfig with freeze/validate
```

~6,300 lines of source. ~5,500 lines of tests.

The interesting part isn't this repo. It's what built it.

## License

Apache 2.0
