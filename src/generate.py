#!/usr/bin/env python3
"""Sample from a trained model checkpoint."""
import os
import pickle
import argparse
import torch

def load_checkpoint(ckpt_path: str, device: str = 'cpu'):
    """Load model from checkpoint."""
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_args = checkpoint['model_args']
    config = checkpoint.get('config', {})
    
    model_type = config.get('model', 'v1')
    if model_type == 'v2' or 'n_kv_head' in model_args:
        from model_v2 import GPT, GPTConfig
    else:
        from model import GPT, GPTConfig
    
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    
    state_dict = checkpoint['model']
    for prefix in ['_orig_mod.', 'transformer.']:
        for k in list(state_dict.keys()):
            if k.startswith(prefix):
                state_dict[k[len(prefix):]] = state_dict.pop(k)
    
    for k in list(state_dict.keys()):
        if 'attn.c_attn' in k:
            state_dict[k.replace('attn.c_attn', 'attn.qkv_proj')] = state_dict.pop(k)
        elif 'attn.c_proj' in k:
            state_dict[k.replace('attn.c_proj', 'attn.out_proj')] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model, model_args

def load_meta(meta_path: str):
    """Load tokenizer metadata."""
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    return meta['stoi'], meta['itos']

def generate(
    ckpt_path: str,
    prompt: str = '\n',
    num_samples: int = 1,
    max_new_tokens: int = 500,
    temperature: float = 0.8,
    top_k: int = 200,
    seed: int = 1337,
):
    """Generate text from checkpoint."""
    torch.manual_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model, model_args = load_checkpoint(ckpt_path, device)
    
    meta_path = os.path.join(os.path.dirname(ckpt_path), '..', 'data', 'shakespeare_char', 'meta.pkl')
    if not os.path.exists(meta_path):
        meta_path = 'data/shakespeare_char/meta.pkl'
    
    stoi, itos = load_meta(meta_path)
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    start_ids = encode(prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
    
    with torch.no_grad():
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate text from checkpoint')
    parser.add_argument('--ckpt', type=str, default='out/ckpt.pt', help='checkpoint path')
    parser.add_argument('--prompt', type=str, default='\n', help='prompt string')
    parser.add_argument('--num_samples', type=int, default=1, help='number of samples')
    parser.add_argument('--max_new_tokens', type=int, default=500, help='tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='sampling temperature')
    parser.add_argument('--top_k', type=int, default=200, help='top-k sampling')
    parser.add_argument('--seed', type=int, default=1337, help='random seed')
    
    args = parser.parse_args()
    generate(
        args.ckpt,
        args.prompt,
        args.num_samples,
        args.max_new_tokens,
        args.temperature,
        args.top_k,
        args.seed
    )
