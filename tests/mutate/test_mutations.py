"""Parametric tests for mutation generators.

Tests two invariants:
1. Mutations produce valid configs (correct keys, types)
2. Mutations roundtrip through YAML
"""

import tempfile

import pytest
import yaml

from foundry.mutate import (
    load_baseline,
    mutate_activation,
    mutate_adam_betas,
    mutate_attention,
    mutate_batch_size,
    mutate_constitution,
    mutate_curriculum,
    mutate_depth,
    mutate_dpo,
    mutate_grad_clip,
    mutate_lora_alpha,
    mutate_lora_dropout,
    mutate_lora_rank,
    mutate_loss,
    mutate_lr,
    mutate_norm,
    mutate_position_encoding,
    mutate_warmup,
    mutate_weight_decay,
    mutate_width,
    save_mutation,
)


def test_load_baseline():
    config = load_baseline()
    assert "name" in config
    assert "training" in config
    assert "model_args" in config


MUTATION_CASES = [
    (mutate_attention, ("gqa_2kv",), {"name": "attn_gqa_2kv", "model_args": {"n_kv_head": int}}),
    (mutate_attention, ("gqa_1kv",), {"name": "attn_gqa_1kv", "model_args": {"n_kv_head": int}}),
    (mutate_attention, ("mha",), {"name": "attn_mha", "model_args": {"n_kv_head": int}}),
    (mutate_depth, (8,), {"name": "depth_8L", "model_args": {"n_layer": 8}}),
    (mutate_width, (512,), {"name": "width_512d", "model_args": {"n_embd": 512}}),
    (mutate_lr, (3e-4,), {"name": "lr_3e-04", "training": {"learning_rate": 3e-4}}),
    (
        mutate_norm,
        ("layernorm",),
        {"name": "norm_layernorm", "model_args": {"norm_type": "layernorm"}},
    ),
    (mutate_norm, ("rmsnorm",), {"name": "norm_rmsnorm", "model_args": {"norm_type": "rmsnorm"}}),
    (mutate_activation, ("gelu",), {"name": "act_gelu", "model_args": {"activation": "gelu"}}),
    (
        mutate_activation,
        ("swiglu",),
        {"name": "act_swiglu", "model_args": {"activation": "swiglu"}},
    ),
    (
        mutate_position_encoding,
        ("rope",),
        {"name": "pos_rope", "model_args": {"position_encoding": "rope"}},
    ),
    (
        mutate_position_encoding,
        ("alibi",),
        {"name": "pos_alibi", "model_args": {"position_encoding": "alibi"}},
    ),
    (mutate_loss, ("focal",), {"name": "loss_focal", "model_args": {"loss_type": "focal"}}),
    (mutate_loss, ("dpo",), {"name": "loss_dpo", "model_args": {"loss_type": "dpo"}}),
    (mutate_weight_decay, (1e-2,), {"name": "wd_1e-02", "training": {"weight_decay": 1e-2}}),
    (mutate_batch_size, (128,), {"name": "batch_128", "training": {"batch_size": 128}}),
    (mutate_warmup, (500,), {"name": "warmup_500", "training": {"warmup_iters": 500}}),
    (mutate_grad_clip, (0.5,), {"name": "gradclip_0.5", "training": {"grad_clip": 0.5}}),
    (mutate_lora_rank, (16,), {"name": "lora_r16", "lora": {"r": 16, "enabled": True}}),
    (
        mutate_lora_alpha,
        (32,),
        {"name": "lora_alpha32", "lora": {"lora_alpha": 32, "enabled": True}},
    ),
    (
        mutate_lora_dropout,
        (0.1,),
        {"name": "lora_dropout0.1", "lora": {"lora_dropout": 0.1, "enabled": True}},
    ),
]


def _check_expected(config, expected):
    for key, val in expected.items():
        if isinstance(val, dict):
            assert key in config
            for subkey, subval in val.items():
                if subval is int:
                    assert isinstance(config[key][subkey], int)
                else:
                    assert config[key][subkey] == subval
        else:
            assert config[key] == val


@pytest.mark.parametrize("mutator,args,expected", MUTATION_CASES)
def test_mutation_produces_valid_config(mutator, args, expected):
    config = mutator(*args)
    _check_expected(config, expected)


@pytest.mark.parametrize("mutator,args,expected", MUTATION_CASES)
def test_mutation_yaml_roundtrip(mutator, args, expected):
    config = mutator(*args)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_mutation(config, tmpdir)
        assert path.exists()
        with open(path) as f:
            loaded = yaml.safe_load(f)
        assert loaded["name"] == config["name"]


def test_mutate_adam_betas():
    config = mutate_adam_betas(0.9, 0.999)
    assert config["name"] == "adam_b1_0.9_b2_0.999"
    assert config["training"]["beta1"] == 0.9
    assert config["training"]["beta2"] == 0.999


def test_mutate_dpo_custom():
    config = mutate_dpo(beta=0.5, label_smoothing=0.1)
    assert config["name"] == "dpo_beta0.5_smooth0.1"
    assert config["model_args"]["dpo_beta"] == 0.5
    assert config["model_args"]["dpo_label_smoothing"] == 0.1


def test_mutate_curriculum():
    config = mutate_curriculum(strategy="length", schedule="linear", num_stages=4)
    assert config["name"] == "curriculum_length_linear_4stages"
    assert config["data"]["curriculum"]["enabled"] is True
    assert config["data"]["curriculum"]["strategy"] == "length"


def test_mutate_constitution():
    config = mutate_constitution("data/constitutions/helpful.jsonl")
    assert config["name"] == "constitution_helpful"
    assert config["data"]["constitution"]["enabled"] is True


INVALID_CASES = [
    (mutate_attention, ("invalid_variant",), "Unknown attention variant"),
    (mutate_norm, ("invalid_norm",), "Unknown norm type"),
    (mutate_activation, ("invalid_act",), "Unknown activation"),
    (mutate_position_encoding, ("invalid_pos",), "Unknown position encoding"),
    (mutate_loss, ("invalid_loss",), "Unknown loss type"),
    (mutate_curriculum, (), {"strategy": "invalid"}, "Unknown strategy"),
    (mutate_curriculum, (), {"schedule": "invalid"}, "Unknown schedule"),
]


@pytest.mark.parametrize("mutator,args,match", [c[:3] for c in INVALID_CASES if len(c) == 3])
def test_invalid_raises(mutator, args, match):
    with pytest.raises(ValueError, match=match):
        mutator(*args)


def test_curriculum_invalid_strategy():
    with pytest.raises(ValueError, match="Unknown strategy"):
        mutate_curriculum(strategy="invalid")


def test_curriculum_invalid_schedule():
    with pytest.raises(ValueError, match="Unknown schedule"):
        mutate_curriculum(schedule="invalid")
