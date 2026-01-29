"""Tests for scenario-to-preference bridge."""

import tempfile
from pathlib import Path

from foundry.data.constitution_format import load_constitution, validate_constitution
from foundry.data.scenarios import (
    SCENARIOS,
    generate_scenario_pairs,
    generate_variations,
    save_scenario_constitution,
)


def test_scenarios_have_required_fields():
    """All scenarios have prompt, chosen, rejected."""
    for scenario_id, scenario in SCENARIOS.items():
        assert "prompt" in scenario, f"{scenario_id} missing prompt"
        assert "chosen" in scenario, f"{scenario_id} missing chosen"
        assert "rejected" in scenario, f"{scenario_id} missing rejected"
        assert "description" in scenario, f"{scenario_id} missing description"


def test_generate_scenario_pairs():
    """Generate pairs in correct format."""
    pairs = generate_scenario_pairs()

    assert len(pairs) == len(SCENARIOS)
    assert validate_constitution(pairs)

    for pair in pairs:
        assert "metadata" in pair
        assert "scenario_id" in pair["metadata"]
        assert pair["metadata"]["source"] == "space-os/scenario.py"


def test_generate_pairs_without_metadata():
    """Generate pairs without metadata."""
    pairs = generate_scenario_pairs(include_metadata=False)

    assert len(pairs) == len(SCENARIOS)
    assert validate_constitution(pairs)

    for pair in pairs:
        assert "metadata" not in pair


def test_generate_variations():
    """Generate variations of base pairs."""
    base = generate_scenario_pairs()
    varied = generate_variations(base, num_variations=2)

    assert len(varied) > len(base)
    assert validate_constitution(varied)


def test_save_and_load_scenario_constitution():
    """Save and reload scenario constitution."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "constitution.jsonl"

        count = save_scenario_constitution(path, include_variations=False)
        assert count == len(SCENARIOS)
        assert path.exists()

        loaded = load_constitution(path)
        assert len(loaded) == len(SCENARIOS)
        assert validate_constitution(loaded)


def test_save_with_variations():
    """Save with variations produces more pairs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "constitution.jsonl"

        count = save_scenario_constitution(path, include_variations=True, num_variations=2)
        assert count > len(SCENARIOS)

        loaded = load_constitution(path)
        assert validate_constitution(loaded)


def test_chosen_is_longer_than_rejected():
    """Chosen responses are generally more thoughtful (longer)."""
    for scenario_id, scenario in SCENARIOS.items():
        assert len(scenario["chosen"]) > len(scenario["rejected"]), (
            f"{scenario_id}: chosen should be more substantive than rejected"
        )


def test_chosen_responses_are_substantive():
    """Chosen responses have reasonable length."""
    for scenario_id, scenario in SCENARIOS.items():
        assert len(scenario["chosen"]) > 100, f"{scenario_id}: chosen response too short"


if __name__ == "__main__":
    test_scenarios_have_required_fields()
    test_generate_scenario_pairs()
    test_generate_pairs_without_metadata()
    test_generate_variations()
    test_save_and_load_scenario_constitution()
    test_save_with_variations()
    test_chosen_is_longer_than_rejected()
    test_chosen_responses_are_substantive()
    print("\n✓ All scenario tests passed")
