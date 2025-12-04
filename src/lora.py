"""LoRA (Low-Rank Adaptation) implementation."""

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Linear layer with LoRA adapter."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        merge_weights: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        self.merge_weights = merge_weights

        self.linear = nn.Linear(in_features, out_features, bias=bias)

        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(in_features, r))
            self.lora_B = nn.Parameter(torch.zeros(r, out_features))
            self.scaling = self.lora_alpha / self.r

            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.linear(x)

        if self.r > 0:
            lora_out = (self.lora_dropout(x) @ self.lora_A @ self.lora_B) * self.scaling
            result = result + lora_out

        return result

    def merge(self):
        """Merge LoRA weights into base linear layer."""
        if self.r > 0 and not self.merge_weights:
            self.linear.weight.data += (self.lora_A @ self.lora_B).T * self.scaling
            self.merge_weights = True

    def unmerge(self):
        """Unmerge LoRA weights from base linear layer."""
        if self.r > 0 and self.merge_weights:
            self.linear.weight.data -= (self.lora_A @ self.lora_B).T * self.scaling
            self.merge_weights = False


def mark_only_lora_as_trainable(model: nn.Module) -> None:
    """Freeze all parameters except LoRA adapters."""
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True


def apply_lora_to_model(
    model: nn.Module,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    target_modules: list[str] = None,
) -> nn.Module:
    """Replace linear layers with LoRA adapters.

    Args:
        model: Model to apply LoRA to
        r: LoRA rank
        lora_alpha: LoRA scaling factor
        lora_dropout: Dropout probability
        target_modules: List of module name patterns to target (e.g., ['q_proj', 'v_proj'])
                       If None, targets all linear layers
    """
    if target_modules is None:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "fc",
            "proj",
            "w1",
            "w2",
        ]

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            should_apply = any(target in name for target in target_modules)

            if should_apply:
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]

                parent = model
                if parent_name:
                    for part in parent_name.split("."):
                        parent = getattr(parent, part)

                lora_layer = LoRALinear(
                    module.in_features,
                    module.out_features,
                    r=r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias=module.bias is not None,
                )

                lora_layer.linear.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    lora_layer.linear.bias.data = module.bias.data.clone()

                setattr(parent, child_name, lora_layer)

    mark_only_lora_as_trainable(model)
    return model


def get_lora_params(model: nn.Module) -> dict:
    """Get statistics about LoRA parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_pct": 100 * trainable_params / total_params if total_params > 0 else 0,
    }
