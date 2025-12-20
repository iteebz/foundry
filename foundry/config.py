"""Unified configuration for training runs."""

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from foundry.model import GPTConfig


@dataclass
class DataSource:
    path: str
    weight: float = 1.0


@dataclass
class DataConfig:
    dataset: str = "openwebtext"
    batch_size: int = 64
    block_size: int = 256
    sources: list[DataSource] = field(default_factory=list)


@dataclass
class TrainingConfig:
    out_dir: str = "out"
    seed: int | None = None
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
    log_gns: bool = False
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


class ConfigFrozenError(Exception):
    """Raised when attempting to modify a frozen config."""


@dataclass
class RunConfig:
    name: str
    model: GPTConfig
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    metadata: dict = field(default_factory=dict)
    _frozen: bool = field(default=False, repr=False)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False) and name != "_frozen":
            raise ConfigFrozenError(f"Cannot modify frozen config: {name}")
        super().__setattr__(name, value)

    def freeze(self) -> "RunConfig":
        """Freeze config to prevent further modifications."""
        object.__setattr__(self, "_frozen", True)
        return self

    def validate(self) -> None:
        """Validate config before freezing. Raises ValueError on invalid config."""
        if self.training.seed is None:
            raise ValueError("training.seed must be explicitly set for reproducibility")

    @classmethod
    def from_yaml(cls, path: Path) -> "RunConfig":
        """Load configuration from YAML file."""
        with path.open() as f:
            raw = yaml.safe_load(f)

        name = raw.get("name", path.stem)

        model_args = raw.get("model_args", {})
        model_config = GPTConfig(**model_args)

        training_args = raw.get("training", {})
        training_config = TrainingConfig(**training_args)

        data_args = raw.get("data", {})

        sources_raw = data_args.pop("sources", [])
        sources = [DataSource(**s) for s in sources_raw] if sources_raw else []
        data_config = DataConfig(**data_args, sources=sources)

        lora_args = raw.get("lora", {})
        lora_config = LoRAConfig(**lora_args)

        wandb_args = raw.get("wandb", {})
        wandb_config = WandbConfig(**wandb_args)

        metadata = raw.get("_metadata", {})

        run_id = raw.get("run_id", uuid.uuid4().hex[:8])

        config = cls(
            name=name,
            run_id=run_id,
            model=model_config,
            data=data_config,
            training=training_config,
            lora=lora_config,
            wandb=wandb_config,
            metadata=metadata,
        )
        config.validate()
        return config.freeze()

    @classmethod
    def from_dict(cls, raw: dict) -> "RunConfig":
        """Load configuration from dictionary (e.g., checkpoint)."""
        model_config = GPTConfig(**raw.get("model_args", {}))

        training_config = TrainingConfig(**raw.get("training", {}))

        data_args = raw.get("data", {})
        sources_raw = data_args.pop("sources", [])
        sources = [DataSource(**s) for s in sources_raw] if sources_raw else []
        data_config = DataConfig(**data_args, sources=sources)

        lora_config = LoRAConfig(**raw.get("lora", {}))
        wandb_config = WandbConfig(**raw.get("wandb", {}))

        return cls(
            name=raw.get("name", "unknown"),
            run_id=raw.get("run_id", uuid.uuid4().hex[:8]),
            model=model_config,
            data=data_config,
            training=training_config,
            lora=lora_config,
            wandb=wandb_config,
            metadata=raw.get("_metadata", {}),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "run_id": self.run_id,
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
                "gradient_checkpointing": self.model.gradient_checkpointing,
            },
            "training": {
                "out_dir": self.training.out_dir,
                "seed": self.training.seed,
                "eval_interval": self.training.eval_interval,
                "log_interval": self.training.log_interval,
                "eval_iters": self.training.eval_iters,
                "eval_only": self.training.eval_only,
                "always_save_checkpoint": self.training.always_save_checkpoint,
                "init_from": self.training.init_from,
                "gradient_accumulation_steps": self.training.gradient_accumulation_steps,
                "learning_rate": self.training.learning_rate,
                "max_iters": self.training.max_iters,
                "weight_decay": self.training.weight_decay,
                "beta1": self.training.beta1,
                "beta2": self.training.beta2,
                "grad_clip": self.training.grad_clip,
                "log_gns": self.training.log_gns,
                "use_ema": self.training.use_ema,
                "ema_decay": self.training.ema_decay,
                "decay_lr": self.training.decay_lr,
                "warmup_iters": self.training.warmup_iters,
                "lr_decay_iters": self.training.lr_decay_iters,
                "min_lr": self.training.min_lr,
                "device": self.training.device,
                "dtype": self.training.dtype,
                "compile": self.training.compile,
                "compile_mode": self.training.compile_mode,
                "gradient_checkpointing": self.training.gradient_checkpointing,
                "distributed": self.training.distributed,
                "fsdp_min_params": self.training.fsdp_min_params,
            },
            "data": {
                "dataset": self.data.dataset,
                "batch_size": self.data.batch_size,
                "block_size": self.data.block_size,
                "sources": [{"path": s.path, "weight": s.weight} for s in self.data.sources],
            },
            "lora": {
                "enabled": self.lora.enabled,
                "r": self.lora.r,
                "lora_alpha": self.lora.lora_alpha,
                "lora_dropout": self.lora.lora_dropout,
            },
            "wandb": {
                "enabled": self.wandb.enabled,
                "project": self.wandb.project,
                "run_name": self.wandb.run_name,
            },
            "_metadata": self.metadata,
        }
