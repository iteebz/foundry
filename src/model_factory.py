"""Model factory for loading models from experiment YAML configs."""

import contextlib
from pathlib import Path

import yaml


def load_experiment_config(experiment_file: str) -> dict:
    """Load experiment configuration from YAML file."""
    config_path = Path(experiment_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {experiment_file}")

    with open(config_path) as f:
        return yaml.safe_load(f)



def get_model_from_experiment(experiment_file: str) -> dict:
    """Extract model args from experiment config."""
    config = load_experiment_config(experiment_file)
    return config.get("model_args", {})


def get_training_overrides(experiment_file: str) -> dict:
    """Extract training parameter overrides from experiment config."""
    config = load_experiment_config(experiment_file)
    training_config = config.get("training", {})
    model_args = config.get("model_args", {})

    overrides = {}
    overrides.update(training_config)
    overrides.update(model_args)

    for key, val in overrides.items():
        if isinstance(val, str):
            with contextlib.suppress(ValueError):
                overrides[key] = float(val)

    return overrides
