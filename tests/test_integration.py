import sys
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from model_v2 import GPT, GPTConfig

def test_forward_pass():
    config = GPTConfig(
        block_size=128,
        vocab_size=1024,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        n_embd=64,
        dropout=0.0,
        bias=False
    )
    
    model = GPT(config)
    model.eval()
    
    batch_size = 2
    seq_len = 32
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        logits, loss = model(idx)
    
    assert logits.shape == (batch_size, 1, config.vocab_size)
    assert loss is None
    assert not torch.isnan(logits).any()
    print(f"✓ Forward pass (no targets): {logits.shape}")
    
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    with torch.no_grad():
        logits, loss = model(idx, targets)
    
    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    assert loss is not None
    assert not torch.isnan(loss)
    print(f"✓ Forward pass (with targets): loss={loss.item():.4f}")

def test_generate():
    config = GPTConfig(
        block_size=128,
        vocab_size=1024,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        n_embd=64,
        dropout=0.0,
        bias=False
    )
    
    model = GPT(config)
    model.eval()
    
    idx = torch.randint(0, config.vocab_size, (1, 10))
    generated = model.generate(idx, max_new_tokens=20)
    
    assert generated.shape == (1, 30)
    print(f"✓ Generation: {idx.shape} -> {generated.shape}")

if __name__ == '__main__':
    print("Testing nanoGPT v2 with modern modules:")
    print("- RMSNorm (replaces LayerNorm)")
    print("- RoPE (replaces learned positional embeddings)")
    print("- SwiGLU (replaces GELU MLP)")
    print("- GQA (replaces MHA)")
    print()
    test_forward_pass()
    test_generate()
    print("\nAll integration tests passed")
