"""Intelligent distributed training configuration."""

import os

import torch
import torch.nn as nn


def get_world_info() -> tuple[int, int, int, bool]:
    """Get distributed world information.

    Returns:
        (rank, local_rank, world_size, is_distributed)
    """
    rank = int(os.environ.get("RANK", -1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = rank != -1

    return rank, local_rank, world_size, is_distributed


def should_use_fsdp(model: nn.Module, min_params: int = 1_000_000_000) -> bool:
    """Determine if FSDP should be used based on model size.

    Args:
        model: Model to check
        min_params: Minimum parameter count to enable FSDP (default: 1B)

    Returns:
        True if model is large enough to benefit from FSDP
    """
    total_params = sum(p.numel() for p in model.parameters())
    return total_params >= min_params


def wrap_model_distributed(
    model: nn.Module,
    strategy: str = "auto",
    fsdp_min_params: int = 1_000_000_000,
) -> tuple[nn.Module, bool, bool]:
    """Intelligently wrap model for distributed training.

    Auto-selection logic:
    - Single GPU/CPU → No wrapping
    - Multi-GPU + small model (<1B) → DDP
    - Multi-GPU + large model (≥1B) → FSDP
    - Multi-CPU → DDP with gloo backend (dev/testing only)

    Args:
        model: Model to wrap
        strategy: "auto", "ddp", "fsdp", or "none"
        fsdp_min_params: Minimum params to use FSDP in auto mode

    Returns:
        (wrapped_model, is_ddp, is_fsdp)
    """
    rank, local_rank, world_size, is_distributed = get_world_info()

    if not is_distributed or world_size == 1:
        return model, False, False

    if strategy == "none":
        return model, False, False

    if not torch.cuda.is_available():
        if rank == 0:
            print("WARNING: Multi-process training on CPU (dev/testing only, very slow)")
            print("         Using gloo backend with DDP")
        from torch.nn.parallel import DistributedDataParallel as DDP

        model = DDP(model)
        return model, True, False

    torch.cuda.set_device(local_rank)
    model = model.to(f"cuda:{local_rank}")

    if strategy == "auto":
        use_fsdp = should_use_fsdp(model, fsdp_min_params)
        strategy = "fsdp" if use_fsdp else "ddp"

    if strategy == "ddp":
        from torch.nn.parallel import DistributedDataParallel as DDP

        model = DDP(model, device_ids=[local_rank])
        return model, True, False

    if strategy == "fsdp":
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

        auto_wrap_policy = size_based_auto_wrap_policy(
            min_num_params=100_000_000  # Wrap layers with >100M params
        )

        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            device_id=local_rank,
            limit_all_gathers=True,
        )
        return model, False, True

    raise ValueError(f"Unknown strategy: {strategy}. Use 'auto', 'ddp', 'fsdp', or 'none'")


def init_distributed(backend: str = "nccl") -> tuple[bool, int, int]:
    """Initialize distributed process group if needed.

    Args:
        backend: Distributed backend ('nccl' for GPU, 'gloo' for CPU)

    Returns:
        (master_process, rank, world_size)
    """
    rank, local_rank, world_size, is_distributed = get_world_info()

    if not is_distributed:
        return True, 0, 1

    if not torch.distributed.is_initialized():
        if not torch.cuda.is_available() and backend == "nccl":
            backend = "gloo"
            if rank == 0:
                print("Switching to gloo backend (CPU-only mode)")

        torch.distributed.init_process_group(backend=backend)

    master_process = rank == 0
    return master_process, rank, world_size


def cleanup_distributed():
    """Clean up distributed process group."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def print_distributed_info(model: nn.Module, is_ddp: bool, is_fsdp: bool):
    """Print distributed training configuration."""
    rank, local_rank, world_size, is_distributed = get_world_info()

    if not is_distributed:
        print("Training mode: Single GPU")
        return

    total_params = sum(p.numel() for p in model.parameters()) / 1e6

    if is_fsdp:
        strategy = "FSDP (Fully Sharded Data Parallel)"
    elif is_ddp:
        strategy = "DDP (Distributed Data Parallel)"
    else:
        strategy = "None"

    if rank == 0:
        print(f"Distributed training: {strategy}")
        print(f"  World size: {world_size}")
        print(f"  Model params: {total_params:.1f}M")
        print(
            f"  Backend: {torch.distributed.get_backend() if torch.distributed.is_initialized() else 'N/A'}"
        )
