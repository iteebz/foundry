import torch

from foundry.modules.gqa import GroupedQueryAttention
from foundry.modules.rmsnorm import RMSNorm
from foundry.modules.rope import RotaryEmbedding, apply_rotary_emb
from foundry.modules.swiglu import SwiGLU


def test_rmsnorm():
    batch, seq, dim = 2, 10, 64
    x = torch.randn(batch, seq, dim)
    norm = RMSNorm(dim)
    out = norm(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()
    print("✓ RMSNorm")


def test_rope():
    batch, seq, n_head, head_dim = 2, 10, 8, 64
    x = torch.randn(batch, seq, n_head * head_dim)
    rope = RotaryEmbedding(head_dim)
    cos, sin = rope(x, seq)
    assert cos.shape == (seq, head_dim)
    assert sin.shape == (seq, head_dim)
    q = torch.randn(batch, n_head, seq, head_dim)
    k = torch.randn(batch, n_head, seq, head_dim)
    cos_expanded = cos.unsqueeze(0).unsqueeze(1)
    sin_expanded = sin.unsqueeze(0).unsqueeze(1)
    q_rot, k_rot = apply_rotary_emb(q, k, cos_expanded, sin_expanded)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    print("✓ RoPE")


def test_swiglu():
    batch, seq, dim = 2, 10, 64
    x = torch.randn(batch, seq, dim)
    mlp = SwiGLU(dim)
    out = mlp(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()
    print("✓ SwiGLU")


def test_gqa():
    batch, seq, n_embd = 2, 10, 64
    n_head, n_kv_head = 8, 2
    x = torch.randn(batch, seq, n_embd)
    attn = GroupedQueryAttention(n_embd, n_head, n_kv_head)
    out = attn(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()
    print("✓ GQA")


if __name__ == "__main__":
    test_rmsnorm()
    test_rope()
    test_swiglu()
    test_gqa()
    print("\nAll module tests passed")
