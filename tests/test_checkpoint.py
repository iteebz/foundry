import sys
import torch
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model import GPT, GPTConfig
from checkpoint import save_checkpoint, load_checkpoint


def test_save_load_checkpoint():
    """Test checkpoint save/load roundtrip."""
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
    optimizer = model.configure_optimizers(0.01, 1e-4, (0.9, 0.95), 'cpu')
    
    original_weight = model.transformer.wte.weight.clone()
    
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        temp_path = f.name
    
    try:
        save_checkpoint(model, optimizer, {'test': 'config'}, temp_path)
        
        model.transformer.wte.weight.data.zero_()
        assert not torch.allclose(model.transformer.wte.weight, original_weight)
        
        loaded_config = load_checkpoint(model, optimizer, temp_path)
        
        assert torch.allclose(model.transformer.wte.weight, original_weight)
        assert loaded_config['test'] == 'config'
        print("✓ Checkpoint save/load")
    finally:
        Path(temp_path).unlink()


def test_load_checkpoint_no_optimizer():
    """Test checkpoint load without optimizer."""
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
    optimizer = model.configure_optimizers(0.01, 1e-4, (0.9, 0.95), 'cpu')
    
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        temp_path = f.name
    
    try:
        save_checkpoint(model, optimizer, {}, temp_path)
        load_checkpoint(model, None, temp_path)
        print("✓ Checkpoint load without optimizer")
    finally:
        Path(temp_path).unlink()


if __name__ == '__main__':
    test_save_load_checkpoint()
    test_load_checkpoint_no_optimizer()
    print("\nAll checkpoint tests passed")
