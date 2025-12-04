import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from modules.rmsnorm import RMSNorm
from modules.layernorm import LayerNorm
from modules.rope import RotaryEmbedding, apply_rotary_emb
from modules.alibi import ALiBi
from modules.swiglu import SwiGLU
from modules.gelu import GELU
from modules.glu import GLU
from modules.gqa import GroupedQueryAttention
from modules.mla import MultiLatentAttention
from modules.moe import MoELayer
from modules.sliding_window import SlidingWindowMask
from modules.focal_loss import FocalLoss
from modules.label_smoothing import LabelSmoothingCrossEntropy

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.use_rope = config.position_encoding == 'rope'
        self.use_alibi = config.position_encoding == 'alibi'
        self.use_mla = config.attention_type == 'mla'
        self.use_sliding_window = getattr(config, 'sliding_window_size', None) is not None
        
        if self.use_mla:
            self.mla = MultiLatentAttention(
                config.n_embd,
                config.n_head,
                latent_dim=getattr(config, 'mla_latent_dim', config.n_embd // 2),
                bias=config.bias,
                dropout=config.dropout,
                block_size=config.block_size
            )
        else:
            self.gqa = GroupedQueryAttention(
                config.n_embd,
                config.n_head,
                config.n_kv_head,
                bias=config.bias,
                dropout=config.dropout
            )
            
            if self.use_rope:
                self.rope = RotaryEmbedding(self.head_dim, max_seq_len=config.block_size)
            elif self.use_alibi:
                self.alibi = ALiBi(config.n_head, max_seq_len=config.block_size)
            
            if self.use_sliding_window:
                self.sliding_window = SlidingWindowMask(
                    config.sliding_window_size,
                    max_seq_len=config.block_size
                )

    def forward(self, x):
        if self.use_mla:
            return self.mla(x)
        
        B, T, C = x.size()
        
        q = self.gqa.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.gqa.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.gqa.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        
        attn_bias = None
        if self.use_rope:
            cos, sin = self.rope(x, T)
            cos = cos.unsqueeze(0).unsqueeze(1)
            sin = sin.unsqueeze(0).unsqueeze(1)
            q, k = apply_rotary_emb(q, k, cos, sin)
        elif self.use_alibi:
            attn_bias = self.alibi(T)
        
        k = k.repeat_interleave(self.gqa.n_rep, dim=1)
        v = v.repeat_interleave(self.gqa.n_rep, dim=1)
        
        if self.use_sliding_window:
            sw_mask = self.sliding_window(T)
            if attn_bias is not None:
                attn_bias = attn_bias * sw_mask
            else:
                attn_bias = sw_mask
            is_causal = False
        else:
            is_causal = True
        
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,
            dropout_p=self.gqa.attn_dropout.p if self.training else 0.0,
            is_causal=is_causal
        )
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.gqa.resid_dropout(self.gqa.o_proj(y))
        return y

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        norm_cls = LayerNorm if config.norm_type == 'layernorm' else RMSNorm
        self.ln_1 = norm_cls(config.n_embd, bias=config.bias) if config.norm_type == 'layernorm' else norm_cls(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = norm_cls(config.n_embd, bias=config.bias) if config.norm_type == 'layernorm' else norm_cls(config.n_embd)
        
        if config.mlp_type == 'moe':
            self.mlp = MoELayer(
                config.n_embd,
                n_experts=getattr(config, 'moe_n_experts', 8),
                top_k=getattr(config, 'moe_top_k', 2),
                bias=config.bias,
                dropout=config.dropout
            )
        else:
            activation_map = {
                'swiglu': SwiGLU,
                'gelu': GELU,
                'glu': GLU,
            }
            act_cls = activation_map.get(config.activation, SwiGLU)
            self.mlp = act_cls(config.n_embd, bias=config.bias)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

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
    norm_type: str = 'rmsnorm'
    activation: str = 'swiglu'
    position_encoding: str = 'rope'
    loss_type: str = 'cross_entropy'
    attention_type: str = 'gqa'
    mla_latent_dim: int = None
    mlp_type: str = 'standard'
    moe_n_experts: int = 8
    moe_top_k: int = 2
    sliding_window_size: int = None

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        
        loss_map = {
            'cross_entropy': None,
            'focal': FocalLoss(),
            'label_smoothing': LabelSmoothingCrossEntropy(),
        }
        self.loss_fn = loss_map.get(config.loss_type)

        norm_cls = LayerNorm if config.norm_type == 'layernorm' else RMSNorm
        final_norm = norm_cls(config.n_embd, bias=config.bias) if config.norm_type == 'layernorm' else norm_cls(config.n_embd)
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = final_norm,
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('w2.weight') or pn.endswith('o_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        tok_emb = self.transformer.wte(idx)
        x = self.transformer.drop(tok_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            if self.loss_fn is None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            else:
                loss = self.loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        for block in self.transformer.h:
            if hasattr(block.attn, 'rope'):
                block.attn.rope.max_seq_len = block_size
            elif hasattr(block.attn, 'alibi'):
                block.attn.alibi.max_seq_len = block_size

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        raise NotImplementedError("model_v2 doesn't support pretrained loading (no learned pos embeddings)")

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt)
        flops_promised = 312e12
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
