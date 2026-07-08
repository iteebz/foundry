from __future__ import annotations

import inspect
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from foundry.modules.alibi import ALiBi
from foundry.modules.focal_loss import FocalLoss
from foundry.modules.gelu import GELU
from foundry.modules.glu import GLU
from foundry.modules.gqa import GroupedQueryAttention
from foundry.modules.label_smoothing import LabelSmoothingCrossEntropy
from foundry.modules.layernorm import LayerNorm
from foundry.modules.mla import MultiLatentAttention
from foundry.modules.moe import MoELayer
from foundry.modules.rmsnorm import RMSNorm
from foundry.modules.rope import RotaryEmbedding, apply_rotary_emb
from foundry.modules.sliding_window import SlidingWindowMask
from foundry.modules.sparse_attention import SparseAttentionMask
from foundry.modules.swiglu import SwiGLU


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 4
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    norm_type: str = "rmsnorm"
    activation: str = "swiglu"
    position_encoding: str = "rope"
    loss_type: str = "cross_entropy"
    attention_type: str = "gqa"
    mla_latent_dim: int | None = None
    mlp_type: str = "standard"
    moe_n_experts: int = 8
    moe_top_k: int = 2
    sliding_window_size: int | None = None
    sparse_block_size: int | None = None
    sparse_stride: int | None = None
    gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")

        if self.n_kv_head > self.n_head:
            raise ValueError(f"n_kv_head ({self.n_kv_head}) cannot exceed n_head ({self.n_head})")

        if self.n_head % self.n_kv_head != 0:
            raise ValueError(
                f"n_head ({self.n_head}) must be divisible by n_kv_head ({self.n_kv_head})"
            )

        head_dim = self.n_embd // self.n_head
        if self.position_encoding == "rope" and head_dim % 2 != 0:
            raise ValueError(
                f"RoPE requires even head_dim, got {head_dim} (n_embd={self.n_embd}, n_head={self.n_head})"
            )

        if self.attention_type == "mla" and self.mla_latent_dim is None:
            raise ValueError("mla_latent_dim required when attention_type='mla'")

        if self.sparse_block_size is not None and self.sparse_stride is None:
            self.sparse_stride = self.sparse_block_size

        if self.mlp_type == "moe" and self.moe_top_k > self.moe_n_experts:
            raise ValueError(
                f"moe_top_k ({self.moe_top_k}) cannot exceed moe_n_experts ({self.moe_n_experts})"
            )

        if self.norm_type not in ["rmsnorm", "layernorm"]:
            raise ValueError(f"Unknown norm_type: {self.norm_type}")

        if self.position_encoding not in ["rope", "alibi"]:
            raise ValueError(f"Unknown position_encoding: {self.position_encoding}")

        if self.activation not in ["swiglu", "gelu", "glu"]:
            raise ValueError(f"Unknown activation: {self.activation}")

        if self.loss_type not in ["cross_entropy", "focal", "label_smoothing"]:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        if self.attention_type not in ["gqa", "mla"]:
            raise ValueError(f"Unknown attention_type: {self.attention_type}")

        if self.mlp_type not in ["standard", "moe"]:
            raise ValueError(f"Unknown mlp_type: {self.mlp_type}")


def _build_norm(config: GPTConfig) -> LayerNorm | RMSNorm:
    if config.norm_type == "layernorm":
        return LayerNorm(config.n_embd, bias=config.bias)
    return RMSNorm(config.n_embd)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.use_rope = config.position_encoding == "rope"
        self.use_alibi = config.position_encoding == "alibi"
        self.use_mla = config.attention_type == "mla"
        self.use_sliding_window = config.sliding_window_size is not None
        self.use_sparse = config.sparse_block_size is not None

        self.mla: MultiLatentAttention | None = None
        self.gqa: GroupedQueryAttention | None = None
        self.rope: RotaryEmbedding | None = None
        self.alibi: ALiBi | None = None
        self.sliding_window: SlidingWindowMask | None = None
        self.sparse: SparseAttentionMask | None = None

        if self.use_mla:
            self.mla = MultiLatentAttention(
                config.n_embd,
                config.n_head,
                latent_dim=config.mla_latent_dim,
                bias=config.bias,
                dropout=config.dropout,
                block_size=config.block_size,
            )
        else:
            self.gqa = GroupedQueryAttention(
                config.n_embd,
                config.n_head,
                config.n_kv_head,
                bias=config.bias,
                dropout=config.dropout,
            )

            if self.use_rope:
                self.rope = RotaryEmbedding(self.head_dim, max_seq_len=config.block_size)
            elif self.use_alibi:
                self.alibi = ALiBi(config.n_head, max_seq_len=config.block_size)

            if self.use_sliding_window and config.sliding_window_size is not None:
                self.sliding_window = SlidingWindowMask(
                    config.sliding_window_size, max_seq_len=config.block_size
                )

            if self.use_sparse and config.sparse_block_size is not None:
                self.sparse = SparseAttentionMask(
                    block_size=config.sparse_block_size,
                    stride=config.sparse_stride or config.sparse_block_size,
                    max_seq_len=config.block_size,
                )

    def _apply_mask(self, position_bias: torch.Tensor | None, mask: torch.Tensor) -> torch.Tensor:
        return position_bias * mask if position_bias is not None else mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mla is not None:
            return self.mla(x)

        assert self.gqa is not None
        B, T, C = x.size()

        q = self.gqa.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.gqa.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.gqa.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        position_bias: torch.Tensor | None = None
        if self.rope is not None:
            cos, sin = self.rope(x, T)
            cos = cos.unsqueeze(0).unsqueeze(1)
            sin = sin.unsqueeze(0).unsqueeze(1)
            q, k = apply_rotary_emb(q, k, cos, sin)
        elif self.alibi is not None:
            position_bias = self.alibi(T)

        k = k.repeat_interleave(self.gqa.n_rep, dim=1)
        v = v.repeat_interleave(self.gqa.n_rep, dim=1)

        is_causal = True
        if self.sliding_window is not None:
            position_bias = self._apply_mask(position_bias, self.sliding_window(T))
            is_causal = False
        elif self.sparse is not None:
            position_bias = self._apply_mask(position_bias, self.sparse(T))
            is_causal = False

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=position_bias,
            dropout_p=self.gqa.attn_dropout.p if self.training else 0.0,
            is_causal=is_causal,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.gqa.resid_dropout(self.gqa.o_proj(y))


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.ln_1 = _build_norm(config)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = _build_norm(config)

        self.mlp: MoELayer | SwiGLU | GELU | GLU
        if config.mlp_type == "moe":
            self.mlp = MoELayer(
                config.n_embd,
                n_experts=config.moe_n_experts,
                top_k=config.moe_top_k,
                bias=config.bias,
                dropout=config.dropout,
            )
        else:
            activation_map: dict[str, type[SwiGLU] | type[GELU] | type[GLU]] = {
                "swiglu": SwiGLU,
                "gelu": GELU,
                "glu": GLU,
            }
            act_cls = activation_map.get(config.activation, SwiGLU)
            self.mlp = act_cls(config.n_embd, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint as ckpt

            def attn_fn(t: torch.Tensor) -> torch.Tensor:
                return self.attn(self.ln_1(t))

            def mlp_fn(t: torch.Tensor) -> torch.Tensor:
                return self.mlp(self.ln_2(t))

            attn_out = ckpt(attn_fn, x, use_reentrant=False)
            mlp_out = ckpt(mlp_fn, x + attn_out, use_reentrant=False)
            return x + attn_out + mlp_out
        attn_out = x + self.attn(self.ln_1(x))
        return attn_out + self.mlp(self.ln_2(attn_out))


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        loss_map: dict[str, FocalLoss | LabelSmoothingCrossEntropy | None] = {
            "cross_entropy": None,
            "focal": FocalLoss(),
            "label_smoothing": LabelSmoothingCrossEntropy(),
        }
        self.loss_fn = loss_map.get(config.loss_type)

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = _build_norm(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith(("w2.weight", "o_proj.weight")):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def get_num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wte.weight.numel()
        return n

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)  # pyright: ignore[reportUnnecessaryComparison]
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _b, t = idx.size()
        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        )

        x = self.drop(self.wte(idx))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            if self.loss_fn is None:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
                )
            else:
                loss = self.loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
    ) -> torch.optim.AdamW:
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = {"fused": True} if use_fused else {}
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = (
                idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            )
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
