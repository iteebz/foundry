import math
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from foundry.metrics import MetricLogger


@dataclass
class CheckResult:
    ok: bool
    score: int
    detail: str


def _check_ci() -> CheckResult:
    just_bin = shutil.which("just")
    if not just_bin:
        return CheckResult(ok=False, score=0, detail="just not found")
    result = subprocess.run(  # noqa: S603
        [just_bin, "ci"],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    if result.returncode != 0:
        return CheckResult(ok=False, score=0, detail=f"CI failed: {result.returncode}")
    return CheckResult(ok=True, score=100, detail="CI passed")


def _check_training_sanity(output_dir: str = "out") -> CheckResult:
    out = Path(output_dir)
    if not out.exists():
        return CheckResult(ok=True, score=100, detail="no training runs")

    experiments = [d for d in out.iterdir() if d.is_dir() and (d / "metrics.jsonl").exists()]
    if not experiments:
        return CheckResult(ok=True, score=100, detail="no experiments with metrics")

    issues = []
    for exp in experiments:
        logger = MetricLogger(str(exp))
        metrics = logger.read_metrics()
        if not metrics:
            continue

        last = metrics[-1]
        train_loss = last.get("train_loss")
        val_loss = last.get("val_loss")

        if train_loss is not None and (math.isnan(train_loss) or math.isinf(train_loss)):
            issues.append(f"{exp.name}: NaN/Inf train_loss")

        if val_loss is not None and (math.isnan(val_loss) or math.isinf(val_loss)):
            issues.append(f"{exp.name}: NaN/Inf val_loss")

    if issues:
        return CheckResult(ok=False, score=50, detail=", ".join(issues))
    return CheckResult(ok=True, score=100, detail=f"{len(experiments)} experiments healthy")


def _check_experiments_valid() -> CheckResult:
    exp_dir = Path("experiments")
    if not exp_dir.exists():
        return CheckResult(ok=False, score=0, detail="experiments/ missing")

    configs = list(exp_dir.glob("*.yaml"))
    if not configs:
        return CheckResult(ok=False, score=0, detail="no experiment configs")

    return CheckResult(ok=True, score=100, detail=f"{len(configs)} configs valid")


def _check_data_exists() -> CheckResult:
    data_dir = Path("data/tinystories")
    required = ["train.bin", "val.bin"]
    missing = [f for f in required if not (data_dir / f).exists()]
    if missing:
        return CheckResult(ok=False, score=0, detail=f"missing: {', '.join(missing)}")
    return CheckResult(ok=True, score=100, detail="tinystories data present")


_CHECKS: list[tuple[str, Callable[[], CheckResult], int]] = [
    ("ci", _check_ci, 40),
    ("training", _check_training_sanity, 30),
    ("experiments", _check_experiments_valid, 15),
    ("data", _check_data_exists, 15),
]


def score() -> dict[str, Any]:
    results: dict[str, CheckResult] = {}
    total_weight = sum(w for _, _, w in _CHECKS)
    weighted_score = 0

    for name, check_fn, weight in _CHECKS:
        result = check_fn()
        results[name] = result
        weighted_score += (result.score / 100) * weight

    final_score = int((weighted_score / total_weight) * 100)
    all_ok = all(r.ok for r in results.values())

    return {
        "ok": all_ok,
        "score": final_score,
        "checks": {name: {"ok": r.ok, "detail": r.detail} for name, r in results.items()},
    }


def cli() -> None:
    result = score()
    print(f"health: {result['score']}/100 {'✓' if result['ok'] else '✗'}")
    for name, check in result["checks"].items():
        status = "✓" if check["ok"] else "✗"
        print(f"  {name}: {status} {check['detail']}")
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
