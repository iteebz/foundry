"""Unified configuration for training runs."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from foundry.model import GPTConfig


@dataclass
class DataConfig:
    dataset: str = "shakespeare_char"
    batch_size: int = 64
    block_size: int = 256


@dataclass
class TrainingConfig:
    out_dir: str = "out"
    eval_interval: int = 500
    log_interval: int = 1
    eval_iters: int = 200
    eval_only: bool = False
    always_save_checkpoint: bool = True
    init_from: str = "scratch"
    gradient_accumulation_steps: int = 1
    learning_rate: float = 6e-4
    max_iters: int = 5000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    use_ema: bool = True
    ema_decay: float = 0.9999
    decay_lr: bool = True
    warmup_iters: int = 100
    lr_decay_iters: int = 5000
    min_lr: float = 6e-5
    device: str = "auto"
    dtype: str = "auto"
    compile: bool = True
    compile_mode: str = "default"
    gradient_checkpointing: bool = False
    distributed: str = "auto"
    fsdp_min_params: int = 1_000_000_000


@dataclass
class LoRAConfig:
    enabled: bool = False
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "foundry"
    run_name: str = "run"


@dataclass
class RunConfig:
    name: str
    model: GPTConfig
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> "RunConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)

        name = raw.get("name", path.stem)

        model_args = raw.get("model_args", {})
        model_config = GPTConfig(**model_args)

        training_args = raw.get("training", {})
        training_config = TrainingConfig(**training_args)

        data_args = raw.get("data", {})
        if "dataset" in training_args:
            data_args.setdefault("dataset", training_args["dataset"])
        if "batch_size" in training_args:
            data_args.setdefault("batch_size", training_args["batch_size"])
        if "block_size" in model_args:
            data_args.setdefault("block_size", model_args["block_size"])
        data_config = DataConfig(**data_args)

        lora_args = raw.get("lora", {})
        lora_config = LoRAConfig(**lora_args)

        wandb_args = raw.get("wandb", {})
        if "wandb_log" in training_args:
            wandb_args.setdefault("enabled", training_args["wandb_log"])
        if "wandb_project" in training_args:
            wandb_args.setdefault("project", training_args["wandb_project"])
        if "wandb_run_name" in training_args:
            wandb_args.setdefault("run_name", training_args["wandb_run_name"])
        wandb_config = WandbConfig(**wandb_args)

        metadata = raw.get("_metadata", {})

        return cls(
            name=name,
            model=model_config,
            data=data_config,
            training=training_config,
            lora=lora_config,
            wandb=wandb_config,
            metadata=metadata,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "model_args": {
                "block_size": self.model.block_size,
                "vocab_size": self.model.vocab_size,
                "n_layer": self.model.n_layer,
                "n_head": self.model.n_head,
                "n_kv_head": self.model.n_kv_head,
                "n_embd": self.model.n_embd,
                "dropout": self.model.dropout,
                "bias": self.model.bias,
                "norm_type": self.model.norm_type,
                "activation": self.model.activation,
                "position_encoding": self.model.position_encoding,
                "loss_type": self.model.loss_type,
                "attention_type": self.model.attention_type,
                "mla_latent_dim": self.model.mla_latent_dim,
                "mlp_type": self.model.mlp_type,
                "moe_n_experts": self.model.moe_n_experts,
                "moe_top_k": self.model.moe_top_k,
                "sliding_window_size": self.model.sliding_window_size,
                "sparse_block_size": self.model.sparse_block_size,
                "sparse_stride": self.model.sparse_stride,
            },
            "training": {
                "max_iters": self.training.max_iters,
                "learning_rate": self.training.learning_rate,
                "weight_decay": self.training.weight_decay,
                "beta1": self.training.beta1,
                "beta2": self.training.beta2,
                "grad_clip": self.training.grad_clip,
                "warmup_iters": self.training.warmup_iters,
                "lr_decay_iters": self.training.lr_decay_iters,
                "min_lr": self.training.min_lr,
                "gradient_checkpointing": self.training.gradient_checkpointing,
            },
            "data": {
                "dataset": self.data.dataset,
                "batch_size": self.data.batch_size,
                "block_size": self.data.block_size,
            },
            "_metadata": self.metadata,
        }
