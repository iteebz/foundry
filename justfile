default:
    @just --list

clean:
    @echo "Cleaning foundry..."
    @rm -rf .pytest_cache .ruff_cache __pycache__ .venv out/smoke_* out/*.pt
    @find . -type d -name "__pycache__" -exec rm -rf {} +
    @find . -type d -name ".pytest_cache" -exec rm -rf {} +

install:
    @uv sync --all-extras

ci:
    @uv run ruff format .
    @uv run ruff check . --fix --unsafe-fixes
    @uv run pytest tests/ -q

test:
    @uv run pytest tests/

cov:
    @uv run pytest --cov=src tests/

format:
    @uv run ruff format .

lint:
    @uv run ruff check .

fix:
    @uv run ruff check . --fix --unsafe-fixes

train-smoke:
    @echo "==> Smoke test: baseline"
    uv run python src/train.py experiments/baseline.yaml

compare:
    uv run python compare.py experiments/baseline.yaml experiments/modern.yaml

commits:
    @git --no-pager log --pretty=format:"%h | %ar | %s"
