# oplot-foundry

Mission doc for foundry ML training infrastructure.

## Goal

Agent-modifiable training infrastructure. nanoGPT base, reference-grade, maximum mutation surface.

## Status

| Phase | Status | Channel |
|-------|--------|---------|
| Scoping | done | foundry-scope |
| Repo Init | done | foundry-init |
| Cherry-pick Modules | done | foundry-modules |
| Integration Test | done | foundry-test |
| Data Prep | done | foundry-data |
| Training Validation | pending | foundry-train |
| Architecture Mutations | pending | foundry-mutate |

## Decision Log

- 2024-12-03: nanoGPT > HF Llama (10:1 complexity, 100% mutable)
- 2024-12-03: Cherry-pick RoPE/GQA/RMSNorm/SwiGLU as standalone modules
- 2024-12-03: CWD injection fix applied to spawn context (prevents agent directory confusion)
- 2024-12-03: Shakespeare char-level for training validation (minimal, CPU-friendly)

## Architecture

```
foundry/
├── docs/specification/vision.md  # done
├── src/
│   ├── model.py      # nanoGPT base (330 lines)
│   ├── model_v2.py   # modern modules integrated
│   ├── train.py      # nanoGPT training (336 lines)
│   └── modules/
│       ├── rope.py   # RoPE (52 lines)
│       ├── gqa.py    # GQA (44 lines)
│       ├── rmsnorm.py # RMSNorm (20 lines)
│       └── swiglu.py # SwiGLU (23 lines)
└── tests/
    ├── test_modules.py
    └── test_integration.py  # 6 tests passing
```

## Mutation Surfaces (from zealot-1 analysis)

1. **Attention** - MHA→GQA, MLA, sliding window, sparse
2. **Position Encoding** - RoPE, ALiBi, learned hybrids
3. **Activations** - GELU→SwiGLU, GLU variants
4. **Normalization** - LayerNorm→RMSNorm, QKNorm
5. **Loss** - CrossEntropy, focal, label smoothing
6. **LR Schedule** - Cosine, cyclic, adaptive
7. **Data Pipeline** - Curriculum, filtering, packing

## Commit Style

TITLE ONLY. SHORT. Examples:
- `add nanoGPT model.py`
- `add RoPE module`
- `fix attention mask`

## Blockers

None yet.
