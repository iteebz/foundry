# Foundry automation

# Run all tests
test:
    pytest tests/

# Quick validation run (both models, 100 iters) - requires CUDA
train-smoke:
    @echo "==> Smoke test: v1 baseline"
    python src/train.py --model=v1 --dataset=shakespeare_char --max_iters=100 --out_dir=out/smoke_v1 --compile=False --always_save_checkpoint=False --eval_interval=100
    @echo "==> Smoke test: v2 modern"
    python src/train.py --model=v2 --dataset=shakespeare_char --max_iters=100 --out_dir=out/smoke_v2 --compile=False --always_save_checkpoint=False --eval_interval=100

# Baseline vs v2 full comparison run - requires CUDA
compare:
    python compare.py --baseline=v1 --mutation=v2 --dataset=shakespeare_char --iters=5000

# CI: test only (train-smoke requires GPU)
ci: test
    @echo "✅ CI passed (tests only - training requires GPU)"
