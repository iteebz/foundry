"""Structured metric logging for training runs."""

import json
from pathlib import Path


class MetricLogger:
    """JSON Lines metric logger for structured training metrics."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.jsonl"

    def log(self, metrics: dict) -> None:
        """Append metrics to JSON Lines file."""
        with self.metrics_path.open("a") as f:
            f.write(json.dumps(metrics) + "\n")

    def read_metrics(self) -> list[dict]:
        """Read all metrics from file."""
        if not self.metrics_path.exists():
            return []

        return [
            json.loads(line) for line in self.metrics_path.read_text().splitlines() if line.strip()
        ]

    def get_final_metrics(self) -> dict | None:
        """Get the last logged metrics."""
        metrics = self.read_metrics()
        return metrics[-1] if metrics else None
