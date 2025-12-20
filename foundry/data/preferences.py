"""Preference pair generation for RLHF/DPO training.

Generate training data from model outputs by comparing responses.
Agents can use this to create alignment datasets from their own models.
"""

import random
from dataclasses import dataclass

import torch


@dataclass
class PreferencePair:
    """A single preference example for DPO training."""

    prompt: str
    chosen: str
    rejected: str
    chosen_score: float = 0.0
    rejected_score: float = 0.0


def generate_pairs_from_samples(
    model,
    tokenizer,
    prompts: list[str],
    num_samples_per_prompt: int = 4,
    temperature: float = 0.8,
    max_new_tokens: int = 512,
    reward_fn=None,
) -> list[PreferencePair]:
    """Generate preference pairs by sampling multiple responses.

    For each prompt, generate N responses and rank them.
    Top response = chosen, worst = rejected.

    Args:
        model: Trained GPT model
        tokenizer: Tokenizer instance
        prompts: List of instruction prompts
        num_samples_per_prompt: Responses to generate per prompt
        temperature: Sampling temperature (>1.0 for diversity)
        max_new_tokens: Max tokens per response
        reward_fn: Optional reward function (prompt, response) -> float
                   If None, uses length as proxy

    Returns:
        List of PreferencePair objects
    """
    if reward_fn is None:

        def reward_fn(p, r):
            return len(r.split())

    model.eval()
    device = next(model.parameters()).device
    pairs = []

    with torch.no_grad():
        for prompt in prompts:
            input_ids = torch.tensor([tokenizer.encode(prompt)]).to(device)

            responses = []
            for _ in range(num_samples_per_prompt):
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=50,
                )
                response = tokenizer.decode(output_ids[0].tolist())
                score = reward_fn(prompt, response)
                responses.append((response, score))

            responses.sort(key=lambda x: x[1], reverse=True)

            chosen = responses[0]
            rejected = responses[-1]

            pairs.append(
                PreferencePair(
                    prompt=prompt,
                    chosen=chosen[0],
                    rejected=rejected[0],
                    chosen_score=chosen[1],
                    rejected_score=rejected[1],
                )
            )

    return pairs


def generate_pairs_from_models(
    model_a,
    model_b,
    tokenizer,
    prompts: list[str],
    reward_fn=None,
    prefer_model: str = "a",
    max_new_tokens: int = 512,
) -> list[PreferencePair]:
    """Generate preference pairs by comparing two models.

    Useful for distillation or A/B testing model variants.

    Args:
        model_a: First model (e.g., baseline)
        model_b: Second model (e.g., larger/better model)
        tokenizer: Shared tokenizer
        prompts: Prompts to compare on
        reward_fn: Optional (prompt, response) -> float scoring function
                   If provided, uses scores to pick chosen/rejected
                   If None, uses prefer_model
        prefer_model: Which model produces "chosen" (if no reward_fn)
        max_new_tokens: Max tokens per response

    Returns:
        Preference pairs with model_a vs model_b
    """
    model_a.eval()
    model_b.eval()
    pairs = []

    device_a = next(model_a.parameters()).device
    device_b = next(model_b.parameters()).device

    with torch.no_grad():
        for prompt in prompts:
            input_ids_a = torch.tensor([tokenizer.encode(prompt)]).to(device_a)
            output_a = model_a.generate(input_ids_a, max_new_tokens=max_new_tokens)
            response_a = tokenizer.decode(output_a[0].tolist())

            input_ids_b = torch.tensor([tokenizer.encode(prompt)]).to(device_b)
            output_b = model_b.generate(input_ids_b, max_new_tokens=max_new_tokens)
            response_b = tokenizer.decode(output_b[0].tolist())

            if reward_fn is not None:
                score_a = reward_fn(prompt, response_a)
                score_b = reward_fn(prompt, response_b)

                if score_a >= score_b:
                    chosen, rejected = response_a, response_b
                    chosen_score, rejected_score = score_a, score_b
                else:
                    chosen, rejected = response_b, response_a
                    chosen_score, rejected_score = score_b, score_a

                pairs.append(
                    PreferencePair(
                        prompt=prompt,
                        chosen=chosen,
                        rejected=rejected,
                        chosen_score=chosen_score,
                        rejected_score=rejected_score,
                    )
                )
            else:
                if prefer_model == "a":
                    chosen, rejected = response_a, response_b
                else:
                    chosen, rejected = response_b, response_a

                pairs.append(PreferencePair(prompt=prompt, chosen=chosen, rejected=rejected))

    return pairs


def generate_constitution_pairs(
    model,
    tokenizer,
    prompts: list[str],
    constitution_principles: list[str],
    max_new_tokens: int = 512,
) -> list[PreferencePair]:
    """Generate preference pairs using constitutional AI principles.

    For each prompt, generate initial response, then critique and revise
    based on constitution. Original = rejected, revised = chosen.

    Args:
        model: Trained GPT model
        tokenizer: Tokenizer instance
        prompts: User prompts
        constitution_principles: List of rules (e.g., "Be helpful and harmless")
        max_new_tokens: Max tokens per response

    Returns:
        Pairs with original vs constitutionally-revised responses
    """
    model.eval()
    device = next(model.parameters()).device
    pairs = []

    critique_template = """Response: {response}

Critique this response based on: {principle}

Critique:"""

    revision_template = """Original response: {response}

Critique: {critique}

Revised response following the critique:"""

    with torch.no_grad():
        for prompt in prompts:
            input_ids = torch.tensor([tokenizer.encode(prompt)]).to(device)
            output = model.generate(input_ids, max_new_tokens=max_new_tokens)
            original_response = tokenizer.decode(output[0].tolist())

            principle = random.choice(constitution_principles)  # noqa: S311 - not crypto

            critique_prompt = critique_template.format(
                response=original_response, principle=principle
            )
            critique_ids = torch.tensor([tokenizer.encode(critique_prompt)]).to(device)
            critique_output = model.generate(critique_ids, max_new_tokens=256)
            critique = tokenizer.decode(critique_output[0].tolist())

            revision_prompt = revision_template.format(
                response=original_response, critique=critique
            )
            revision_ids = torch.tensor([tokenizer.encode(revision_prompt)]).to(device)
            revision_output = model.generate(revision_ids, max_new_tokens=max_new_tokens)
            revised_response = tokenizer.decode(revision_output[0].tolist())

            pairs.append(
                PreferencePair(prompt=prompt, chosen=revised_response, rejected=original_response)
            )

    return pairs


def save_preference_dataset(pairs: list[PreferencePair], output_path: str):
    """Save preference pairs to JSONL for DPO training."""
    import json
    from pathlib import Path

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with Path(output_path).open("w") as f:
        f.writelines(
            json.dumps(
                {
                    "prompt": pair.prompt,
                    "chosen": pair.chosen,
                    "rejected": pair.rejected,
                    "chosen_score": pair.chosen_score,
                    "rejected_score": pair.rejected_score,
                }
            )
            + "\n"
            for pair in pairs
        )
