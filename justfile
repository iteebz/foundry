# Foundry automation

# Run all tests
test:
    pytest tests/

# Quick validation run - requires CUDA
train-smoke:
    @echo "==> Smoke test: baseline"
    python src/train.py experiments/baseline.yaml

# Baseline vs modern full comparison run - requires CUDA
compare:
    python compare.py experiments/baseline.yaml experiments/modern.yaml

# CI: test only (train-smoke requires GPU)
ci: test
    @echo "✅ CI passed (tests only - training requires GPU)"
