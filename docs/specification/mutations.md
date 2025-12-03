# Mutation Surface

Agent-modifiable architecture components. Reference implementations in `src/modules/`.

## Attention

**Current:**
- v1: Multi-Head Attention (MHA) — standard transformer
- v2: Grouped Query Attention (GQA) — `n_kv_head < n_head`, shared KV across groups

**Mutation space:**
- MHA → GQA: reduce KV cache (config: `n_kv_head`)
- GQA → Multi-Latent Attention (MLA): latent compression of KV
- Sparse patterns: sliding window, block-sparse, strided
- Flash attention: enabled by default via `F.scaled_dot_product_attention`

**Implementation:** `src/modules/gqa.py`

**Config surface:**
```yaml
model_args:
  n_head: 6        # query heads
  n_kv_head: 2     # kv heads (GQA only, v2)
```

## Position Encoding

**Current:**
- v1: Learned absolute embeddings (`wpe`)
- v2: Rotary Position Embedding (RoPE) — relative, no learned params

**Mutation space:**
- Learned → RoPE: better length extrapolation
- RoPE → ALiBi: bias-based, even simpler
- Hybrid: RoPE + learned bias
- NoPE experiments: rely on attention patterns alone

**Implementation:** `src/modules/rope.py`

**Config surface:**
```yaml
base_model: "v1"   # learned position embeddings
base_model: "v2"   # RoPE (no config needed)
```

## Normalization

**Current:**
- v1: LayerNorm — standard transformer
- v2: RMSNorm — simpler, no mean centering

**Mutation space:**
- LayerNorm → RMSNorm: 10-15% faster, equivalent quality
- RMSNorm → QKNorm: normalize queries/keys directly
- Pre/post norm placement: foundry uses pre-norm

**Implementation:** `src/modules/rmsnorm.py`

**Config surface:**
No config—hardcoded in model architecture (v1 vs v2).

## Activation

**Current:**
- v1: GELU — GPT-2 standard
- v2: SwiGLU — gated linear unit with Swish

**Mutation space:**
- GELU → SwiGLU: better expressiveness, 1.5x params in FFN
- SwiGLU variants: GLU, ReGLU, GeGLU
- Hidden dim ratio: v2 uses `(8*dim/3)`, v1 uses `4*dim`

**Implementation:** `src/modules/swiglu.py`

**Config surface:**
No config—hardcoded in model architecture (v1 vs v2).

## MLP Architecture

**Current:**
- v1: `Linear(d, 4d) → GELU → Linear(4d, d)`
- v2: `SwiGLU(d, 8d/3)` — gated, larger hidden

**Mutation space:**
- FFN ratio: 4x, 8x/3, 2.67x (MoE style)
- Gating: add gating to v1, remove from v2
- MoE: replace dense FFN with sparse experts

**Implementation:** `src/model.py` (v1 MLP class), `src/modules/swiglu.py` (v2)

**Config surface:**
None exposed—requires code mutation.

## Model Scaling

**Config surface:**
```yaml
model_args:
  n_layer: 6       # transformer blocks
  n_head: 6        # attention heads
  n_embd: 384      # embedding dimension
  block_size: 1024 # context length
  dropout: 0.0     # regularization
  bias: false      # bias in Linear/LayerNorm
```

## Training Dynamics

**Config surface:**
```yaml
training:
  max_iters: 5000
  learning_rate: 6e-4
  batch_size: 12
  gradient_accumulation_steps: 40
  weight_decay: 1e-1
  beta1: 0.9
  beta2: 0.95
  grad_clip: 1.0
  warmup_iters: 2000
  lr_decay_iters: 5000
  min_lr: 6e-5
  eval_interval: 500
```

## Mutation Protocol

1. **Propose:** Identify component to mutate (attention, norm, activation)
2. **Implement:** Add new module to `src/modules/` or modify existing
3. **Integrate:** Wire into model architecture (v1 or v2 or new v3)
4. **Config:** Create YAML experiment file in `experiments/`
5. **Train:** `python src/train.py experiments/your_mutation.yaml`
6. **Eval:** Compare loss/perplexity against baseline
7. **Decide:** Keep if improvement, discard if regression

## Testing

All modules must pass:
```bash
just ci  # runs pytest on src/modules/
```

Reference test structure: `tests/test_modules.py`

Required assertions:
- Shape preservation
- No NaN outputs
- Gradient flow (if training-specific)

## Future Mutation Surfaces

Not yet implemented:

- **MoE routing:** sparse experts, top-k gating
- **Sparse attention:** local/global patterns, kernel fusion
- **KV cache optimization:** compression, quantization
- **Loss functions:** focal loss, label smoothing, curriculum
- **Data curriculum:** ordering strategies, difficulty scheduling
- **Adaptive training:** learning rate schedules, batch size annealing
