# foundry

mutation-based ML training framework. built on nanoGPT in 21 hours.

## the origin story

**dec 3–4, 2025.** one human. one AI agent. 46 commits. zero ML training experience from the human. the [commit log](../../commits/space) is timestamped.

**jan–feb 2026.** 7 autonomous agents returned to the repo independently — type compliance, benchmarks, test coverage, health scoring. nobody asked them to. visible in `git log --format="%an"`.

109 commits total. 72 from the human. 37 from agents named things like `jobs`, `zealot`, `kit`, `prime`, `harbinger`, `oplot`, and `kondo`. this is a real contributor list on a real repo.

## what it does

```
mutate → train → rank → promote → repeat
```

21 mutation types across architecture (GQA/MQA/MHA, MLA, MoE, depth, width, norm, activation, position encoding, loss, sliding window, sparse attention), training (LR, batch size, warmup, grad clip, weight decay, Adam betas, LoRA), and data (conversation format, filtering).

## what's real

- **training loop.** cosine LR, AMP, gradient accumulation, EMA, checkpoint resume, wandb. CPU/MPS/CUDA.
- **sweep runner.** parallel training, ranking, auto-promote winner.
- **mutation engine.** all 21 types generate valid configs. tested.
- **data pipeline.** BPE tokenizer, memory-mapped datasets, 8-filter quality pipeline, curriculum learning.

## what's not

- **never trained at scale.** no GPUs were available. the loop runs, mechanics verified, no loss curves.
- **eval harness untested at scale.** GSM8K, MMLU, HumanEval implementations exist, never evaluated a real checkpoint.
- **model zoo is config-only.** no HuggingFace weight download.

## structure

```
foundry/
├── model.py          GPT with swappable components
├── train.py          training loop
├── mutate/           21 mutation generators
├── modules/          architecture components
├── data/             tokenize, filter, curriculum, mixture
├── benchmarks/       GSM8K, MMLU, HumanEval
├── cli/              sweep, compare
├── distributed.py    DDP/FSDP auto-selection
├── lora.py           LoRA adapters
└── config.py         RunConfig with freeze/validate
```

the interesting part isn't this repo. it's what built it.

## status: paused (jul 2026)

no gpu access, no active runs. becomes relevant if fundraising lands compute.

simplification sweep done (290 tests green): deleted zero-caller modules, shadow filter fns, dead config enums.

**known cuts on restart:**
1. `train.py`/`train_dpo.py` share ~200 duplicated lines — extract shared training preamble
2. `generate.py:load_checkpoint` duplicates `checkpoint.py:load_checkpoint` + carries legacy nanoGPT key-rename shim
3. `TokenDataset` memmap/streaming dual-mode: `_init_*` paths near-identical, unify

## license

Apache 2.0
