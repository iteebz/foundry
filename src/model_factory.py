"""Model factory for loading models from experiment YAML configs."""
import yaml
from pathlib import Path


def load_experiment_config(experiment_file: str) -> dict:
    """Load experiment configuration from YAML file."""
    config_path = Path(experiment_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {experiment_file}")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    return config


def get_model_from_experiment(experiment_file: str) -> tuple[str, dict]:
    """Extract model version and args from experiment config.
    
    Returns:
        tuple: (model_version, model_args) where model_version is 'v1' or 'v2'
    """
    config = load_experiment_config(experiment_file)
    
    base_model = config.get("base_model", "v1")
    if base_model not in ["v1", "v2"]:
        raise ValueError(f"Invalid base_model: {base_model}. Must be 'v1' or 'v2'")
    
    model_args = config.get("model_args", {})
    
    return base_model, model_args
