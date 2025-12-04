"""Tests for GPTConfig validation."""

import pytest

from foundry.model import GPTConfig


def test_valid_config():
    """Valid config passes validation."""
    config = GPTConfig(n_embd=768, n_head=12, n_kv_head=4)
    assert config.n_embd == 768


def test_n_embd_not_divisible_by_n_head():
    """n_embd must be divisible by n_head."""
    with pytest.raises(ValueError, match="n_embd.*divisible.*n_head"):
        GPTConfig(n_embd=100, n_head=12)


def test_n_kv_head_exceeds_n_head():
    """n_kv_head cannot exceed n_head."""
    with pytest.raises(ValueError, match="n_kv_head.*cannot exceed.*n_head"):
        GPTConfig(n_head=4, n_kv_head=8)


def test_mla_without_latent_dim():
    """MLA requires mla_latent_dim."""
    with pytest.raises(ValueError, match="mla_latent_dim required"):
        GPTConfig(attention_type="mla", mla_latent_dim=None)


def test_mla_with_latent_dim():
    """MLA with latent_dim is valid."""
    config = GPTConfig(attention_type="mla", mla_latent_dim=384)
    assert config.mla_latent_dim == 384


def test_moe_top_k_exceeds_n_experts():
    """MoE top_k cannot exceed n_experts."""
    with pytest.raises(ValueError, match="moe_top_k.*cannot exceed.*moe_n_experts"):
        GPTConfig(mlp_type="moe", moe_n_experts=4, moe_top_k=8)


def test_invalid_norm_type():
    """Unknown norm_type raises error."""
    with pytest.raises(ValueError, match="Unknown norm_type"):
        GPTConfig(norm_type="invalid")


def test_invalid_position_encoding():
    """Unknown position_encoding raises error."""
    with pytest.raises(ValueError, match="Unknown position_encoding"):
        GPTConfig(position_encoding="invalid")


def test_invalid_activation():
    """Unknown activation raises error."""
    with pytest.raises(ValueError, match="Unknown activation"):
        GPTConfig(activation="invalid")


def test_invalid_loss_type():
    """Unknown loss_type raises error."""
    with pytest.raises(ValueError, match="Unknown loss_type"):
        GPTConfig(loss_type="invalid")


def test_invalid_attention_type():
    """Unknown attention_type raises error."""
    with pytest.raises(ValueError, match="Unknown attention_type"):
        GPTConfig(attention_type="invalid")


def test_invalid_mlp_type():
    """Unknown mlp_type raises error."""
    with pytest.raises(ValueError, match="Unknown mlp_type"):
        GPTConfig(mlp_type="invalid")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
