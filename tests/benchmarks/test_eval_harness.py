"""Eval harness tests - aggregate scoring and error handling."""

import json
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from foundry.benchmarks.harness import (
    compute_aggregate_metrics,
    run_benchmark_suite,
    save_eval_results,
)
from foundry.benchmarks.tasks import extract_answer


class DummyTokenizer:
    def encode(self, text):
        return [ord(c) % 100 for c in text[:20]]

    def decode(self, tokens):
        return "The answer is 42"


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def eval(self):
        return self

    def forward(self, inputs):
        return torch.randn(inputs.size(0), inputs.size(1), 100)

    def generate(self, inputs, max_new_tokens=100, temperature=0.7):
        return torch.tensor([[42] * 20])


def test_compute_aggregate_metrics():
    results = {
        "gsm8k": {"accuracy": 0.5},
        "mmlu": {"accuracy": 0.6},
        "humaneval": {"pass_at_1": 0.4},
    }
    agg = compute_aggregate_metrics(results)
    assert "mean_score" in agg
    assert "num_tasks" in agg
    assert agg["num_tasks"] == 3
    assert 0.0 <= agg["mean_score"] <= 1.0


def test_compute_aggregate_metrics_empty_results_is_zero_not_nan():
    # no tasks -> division by zero would raise; must floor to 0.0
    agg = compute_aggregate_metrics({})
    assert agg == {"mean_score": 0.0, "num_tasks": 0}


def test_compute_aggregate_metrics_error_only_task_excluded():
    # a task that only produced an error dict has no "accuracy" key -> excluded
    results = {"gsm8k": {"error": "OOM"}}
    agg = compute_aggregate_metrics(results)
    assert agg == {"mean_score": 0.0, "num_tasks": 0}


def test_compute_aggregate_metrics_constitution_task_counted():
    # constitution uses "preference_accuracy", a distinct key from "accuracy"
    results = {"constitution": {"preference_accuracy": 0.8}}
    agg = compute_aggregate_metrics(results)
    assert agg == {"mean_score": 0.8, "num_tasks": 1}


def test_compute_aggregate_metrics_partial_subset_averages_only_present():
    # only gsm8k and constitution ran; mmlu/humaneval absent must not count as 0
    results = {
        "gsm8k": {"accuracy": 1.0},
        "constitution": {"preference_accuracy": 0.0},
    }
    agg = compute_aggregate_metrics(results)
    assert agg["num_tasks"] == 2
    assert agg["mean_score"] == 0.5


def test_compute_aggregate_metrics_unknown_task_key_ignored():
    # "aggregate" or any unrecognized task name must not be picked up
    results = {"gsm8k": {"accuracy": 0.5}, "aggregate": {"mean_score": 0.9}}
    agg = compute_aggregate_metrics(results)
    assert agg["num_tasks"] == 1
    assert agg["mean_score"] == 0.5


def test_save_eval_results():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "results.json"
        results = {"gsm8k": {"accuracy": 0.5}, "aggregate": {"mean_score": 0.5}}
        save_eval_results(results, output_path)
        assert output_path.exists()
        with open(output_path) as f:
            loaded = json.load(f)
        assert loaded["gsm8k"]["accuracy"] == 0.5


def test_unknown_task_error_handling():
    with tempfile.TemporaryDirectory() as tmpdir:
        model = DummyModel()
        tokenizer = DummyTokenizer()
        results = run_benchmark_suite(model, tokenizer, tasks=["unknown_task"], dataset_dir=tmpdir)
        assert "unknown_task" in results
        assert "error" in results["unknown_task"]


def test_extract_answer_math():
    assert extract_answer("The answer is 42", task_type="math") == "42"
    assert extract_answer("#### 1,234", task_type="math") == "1234"


def test_extract_answer_multiple_choice():
    assert extract_answer("The answer is B", task_type="multiple_choice") == "B"


def test_extract_answer_math_hash_takes_last_match():
    # multiple #### markers: the LAST wins (final answer after reasoning steps)
    assert extract_answer("#### 5\n#### 42", task_type="math") == "42"


def test_extract_answer_math_hash_precedes_bare_number():
    # #### marker beats any bare number appearing earlier in the text
    assert extract_answer("step gives 100\n#### 7", task_type="math") == "7"


def test_extract_answer_math_answer_is_fallback():
    # no #### -> "answer is"/"equals" phrase, last match
    assert extract_answer("it equals 7", task_type="math") == "7"
    assert extract_answer("first answer is 3, final answer is 9", task_type="math") == "9"


def test_extract_answer_math_bare_number_last_resort():
    # no marker, no phrase -> last bare number in text
    assert extract_answer("result 3 then 9", task_type="math") == "9"


def test_extract_answer_math_strips_commas_keeps_dots():
    assert extract_answer("#### 1,234,567", task_type="math") == "1234567"
    assert extract_answer("#### 3.14", task_type="math") == "3.14"


def test_extract_answer_math_none_when_no_number():
    assert extract_answer("nothing here", task_type="math") is None


def test_extract_answer_mc_first_match_wins():
    # multiple choice returns the FIRST isolated A-D letter, not the last
    assert extract_answer("C or D", task_type="multiple_choice") == "C"


def test_extract_answer_mc_requires_word_boundary():
    # a letter embedded in a word is not an answer
    assert extract_answer("BANANA", task_type="multiple_choice") is None


def test_extract_answer_mc_uppercases_input():
    # lowercase choice still resolves
    assert extract_answer("the answer is c", task_type="multiple_choice") == "C"


def test_extract_answer_mc_none_when_no_letter():
    assert extract_answer("no letters", task_type="multiple_choice") is None


def test_extract_answer_unknown_task_type_returns_none():
    # only "math" and "multiple_choice" are handled
    assert extract_answer("42", task_type="other") is None
