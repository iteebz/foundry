default:
    @just --list

clean:
    @echo "Cleaning foundry..."
    @rm -rf .pytest_cache .ruff_cache __pycache__ .venv out/smoke_* out/*.pt
    @find . -type d -name "__pycache__" -exec rm -rf {} +
    @find . -type d -name ".pytest_cache" -exec rm -rf {} +

install:
    @poetry lock
    @poetry install

ci:
    @poetry run ruff format .
    @poetry run ruff check . --fix --unsafe-fixes
    @poetry run pytest tests/ -q

test:
    @poetry run pytest tests/

cov:
    @poetry run pytest --cov=src tests/

format:
    @poetry run ruff format .

lint:
    @poetry run ruff check .

fix:
    @poetry run ruff check . --fix --unsafe-fixes

train-smoke:
    @echo "==> Smoke test: baseline"
    poetry run python src/train.py experiments/baseline.yaml

compare:
    poetry run python compare.py experiments/baseline.yaml experiments/modern.yaml

commits:
    @git --no-pager log --pretty=format:"%h | %ar | %s"
