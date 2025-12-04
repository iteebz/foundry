"""CLI entrypoint for mutation generation."""

import sys

from . import (
    mutate_activation,
    mutate_adam_betas,
    mutate_attention,
    mutate_batch_size,
    mutate_conversation_format,
    mutate_depth,
    mutate_grad_clip,
    mutate_lora_alpha,
    mutate_lora_dropout,
    mutate_lora_rank,
    mutate_loss,
    mutate_lr,
    mutate_mla,
    mutate_moe,
    mutate_norm,
    mutate_position_encoding,
    mutate_sliding_window,
    mutate_sparse_attention,
    mutate_warmup,
    mutate_weight_decay,
    mutate_width,
    save_mutation,
)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.mutate <type> [variant]")
        print("Types: attention, depth, width, lr, norm, activation, position, loss,")
        print("       batch_size, warmup, grad_clip, weight_decay, adam_betas,")
        print("       lora_rank, lora_alpha, lora_dropout, conversation_format,")
        print("       mla, moe, sliding_window, sparse_attention")
        print("\nExamples:")
        print("  python -m src.mutate attention gqa_2kv")
        print("  python -m src.mutate mla 192")
        print("  python -m src.mutate moe 8 2")
        print("  python -m src.mutate sliding_window 512")
        print("  python -m src.mutate sparse_attention 64 128")
        sys.exit(1)

    mutation_type = sys.argv[1]

    if mutation_type == "attention":
        config = mutate_attention(sys.argv[2])
    elif mutation_type == "depth":
        config = mutate_depth(int(sys.argv[2]))
    elif mutation_type == "width":
        config = mutate_width(int(sys.argv[2]))
    elif mutation_type == "lr":
        config = mutate_lr(float(sys.argv[2]))
    elif mutation_type == "norm":
        config = mutate_norm(sys.argv[2])
    elif mutation_type == "activation":
        config = mutate_activation(sys.argv[2])
    elif mutation_type == "position":
        config = mutate_position_encoding(sys.argv[2])
    elif mutation_type == "loss":
        config = mutate_loss(sys.argv[2])
    elif mutation_type == "batch_size":
        config = mutate_batch_size(int(sys.argv[2]))
    elif mutation_type == "warmup":
        config = mutate_warmup(int(sys.argv[2]))
    elif mutation_type == "grad_clip":
        config = mutate_grad_clip(float(sys.argv[2]))
    elif mutation_type == "weight_decay":
        config = mutate_weight_decay(float(sys.argv[2]))
    elif mutation_type == "adam_betas":
        if len(sys.argv) < 4:
            print("adam_betas requires two arguments: beta1 beta2")
            sys.exit(1)
        config = mutate_adam_betas(float(sys.argv[2]), float(sys.argv[3]))
    elif mutation_type == "lora_rank":
        config = mutate_lora_rank(int(sys.argv[2]))
    elif mutation_type == "lora_alpha":
        config = mutate_lora_alpha(int(sys.argv[2]))
    elif mutation_type == "lora_dropout":
        config = mutate_lora_dropout(float(sys.argv[2]))
    elif mutation_type == "conversation_format":
        config = mutate_conversation_format(sys.argv[2])
    elif mutation_type == "mla":
        latent_dim = int(sys.argv[2]) if len(sys.argv) > 2 else None
        config = mutate_mla(latent_dim)
    elif mutation_type == "moe":
        n_experts = int(sys.argv[2]) if len(sys.argv) > 2 else 8
        top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 2
        config = mutate_moe(n_experts, top_k)
    elif mutation_type == "sliding_window":
        window_size = int(sys.argv[2]) if len(sys.argv) > 2 else 256
        config = mutate_sliding_window(window_size)
    elif mutation_type == "sparse_attention":
        block_size = int(sys.argv[2]) if len(sys.argv) > 2 else 64
        stride = int(sys.argv[3]) if len(sys.argv) > 3 else None
        config = mutate_sparse_attention(block_size, stride)
    else:
        print(f"Unknown mutation type: {mutation_type}")
        sys.exit(1)

    path = save_mutation(config)
    print(f"\nGenerated: {path}")
    print(f"Run with: python src/train.py {path}")
