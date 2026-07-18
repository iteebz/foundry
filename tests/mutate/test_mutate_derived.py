"""Pins the DERIVED logic in mutation generators.

test_mutations.py checks each mutator sets its named field, but leaves the
arithmetic and branch logic unguarded: mutate_width derives n_head/n_kv_head,
mutate_attention('mha') mirrors n_head, mutate_mla halves n_embd, and several
name strings are built by conditional concatenation. A mutation to any of
those derivations passes the existing suite. These tests pin the contract.
"""

from foundry.mutate.architecture import (
    mutate_attention,
    mutate_dpo,
    mutate_mla,
    mutate_sparse_attention,
    mutate_width,
)
from foundry.mutate.data import mutate_data_filter


def _base():
    return {"name": "b", "model_args": {"n_head": 6, "n_embd": 384, "n_kv_head": 2}, "training": {}}


def test_width_derives_n_head_from_head_dim_64():
    # n_head = n_embd // 64; n_kv_head = max(1, n_head // 3)
    args = mutate_width(768, _base())["model_args"]
    assert args["n_head"] == 12
    assert args["n_kv_head"] == 4


def test_width_floors_n_head_to_one_below_head_dim():
    # n_embd < 64 would yield n_head 0 without the floor guard
    args = mutate_width(32, _base())["model_args"]
    assert args["n_head"] == 1
    assert args["n_kv_head"] == 1


def test_width_leaves_n_kv_head_absent_when_baseline_lacks_it():
    args = mutate_width(512, {"name": "b", "model_args": {"n_head": 6, "n_embd": 384}})[
        "model_args"
    ]
    assert "n_kv_head" not in args


def test_attention_mha_mirrors_n_head():
    # mha variant sets n_kv_head equal to n_head (full multi-head)
    args = mutate_attention("mha", _base())["model_args"]
    assert args["n_kv_head"] == 6


def test_attention_gqa_1kv_sets_single_kv_head():
    args = mutate_attention("gqa_1kv", _base())["model_args"]
    assert args["n_kv_head"] == 1


def test_mla_default_latent_is_half_n_embd():
    args = mutate_mla(None, _base())["model_args"]
    assert args["mla_latent_dim"] == 192


def test_mla_falls_back_to_384_when_n_embd_missing():
    args = mutate_mla(None, {"name": "b", "model_args": {}})["model_args"]
    assert args["mla_latent_dim"] == 192


def test_mla_explicit_latent_overrides_default():
    args = mutate_mla(64, _base())["model_args"]
    assert args["mla_latent_dim"] == 64


def test_sparse_attention_stride_defaults_to_block_size():
    args = mutate_sparse_attention(128, None, _base())["model_args"]
    assert args["sparse_stride"] == 128
    assert args["sparse_block_size"] == 128


def test_sparse_attention_keeps_explicit_stride():
    args = mutate_sparse_attention(128, 256, _base())["model_args"]
    assert args["sparse_stride"] == 256


def test_dpo_name_omits_smooth_suffix_when_zero():
    config = mutate_dpo(0.1, 0.0, {"name": "b", "model_args": {}})
    assert config["name"] == "dpo_beta0.1"


def test_dpo_name_adds_smooth_suffix_when_positive():
    config = mutate_dpo(0.1, 0.05, {"name": "b", "model_args": {}})
    assert config["name"] == "dpo_beta0.1_smooth0.05"


def test_data_filter_name_only_min_when_no_max_no_dedupe():
    config = mutate_data_filter(10, None, False, {"name": "b", "data": {}})
    assert config["name"] == "filter_min10"
    assert "max_len" not in config["data"]


def test_data_filter_name_appends_max_and_dedupe():
    config = mutate_data_filter(10, 512, True, {"name": "b", "data": {}})
    assert config["name"] == "filter_min10_max512_dedupe"
    assert config["data"]["max_len"] == 512
