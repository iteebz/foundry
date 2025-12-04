#!/bin/bash
# Quick CPU smoke test for distributed logic (dev only, not for actual training)

echo "Testing single-process CPU training..."
python -m foundry.train experiments/baseline.yaml \
    --max_iters 2 \
    --eval_interval 1 \
    --compile false

echo ""
echo "Testing multi-process CPU training (2 workers)..."
torchrun --nproc_per_node=2 -m foundry.train experiments/baseline.yaml \
    --max_iters 2 \
    --eval_interval 1 \
    --compile false

echo ""
echo "✅ If you see this, distributed logic works on CPU"
echo "⚠️  For real training, use GPU (50-100x faster)"
