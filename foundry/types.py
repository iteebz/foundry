"""Type protocols for foundry."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import torch


class ModelProtocol(Protocol):
    """Protocol for model objects."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]: ...

    def eval(self) -> Any: ...

    def train(self, mode: bool = True) -> Any: ...

    def to(self, device: str | torch.device) -> ModelProtocol: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(
        self, state_dict: dict[str, Any], strict: bool = True, assign: bool = False
    ) -> Any: ...

    def parameters(self) -> Any: ...

    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor: ...


class OptimizerProtocol(Protocol):
    """Protocol for optimizer objects."""

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state_dict: dict[str, Any]) -> Any: ...


class TokenizerProtocol(Protocol):
    """Protocol for tokenizer objects."""

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...
