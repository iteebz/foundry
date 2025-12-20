"""Synthetic data generation for training data augmentation.

Agents can use these to create training data from existing models,
closing the loop: model → synthetic data → retrain → improve.
"""

import json
import random
from pathlib import Path

import torch


def self_instruct(
    model,
    tokenizer,
    seed_tasks: list[dict],
    num_samples: int = 1000,
    temperature: float = 0.8,
    max_new_tokens: int = 512,
) -> list[dict]:
    """Generate instruction-following data from seed tasks.

    Based on Self-Instruct paper (Wang et al., 2023).
    Uses existing model to generate new (instruction, response) pairs.

    Args:
        model: Trained GPT model
        tokenizer: Tokenizer instance
        seed_tasks: List of {"instruction": str, "response": str}
        num_samples: Number of examples to generate
        temperature: Sampling temperature
        max_new_tokens: Max tokens per generation

    Returns:
        List of {"instruction": str, "response": str} dicts
    """
    model.eval()
    device = next(model.parameters()).device
    generated = []

    prompt_template = """Below are instruction-response pairs. Generate a new instruction-response pair following the same style.

Examples:
{examples}

New instruction-response pair:
Instruction:"""

    with torch.no_grad():
        for _ in range(num_samples):
            examples = random.sample(seed_tasks, min(3, len(seed_tasks)))
            examples_str = "\n\n".join(
                f"Instruction: {ex['instruction']}\nResponse: {ex['response']}" for ex in examples
            )

            prompt = prompt_template.format(examples=examples_str)
            input_ids = torch.tensor([tokenizer.encode(prompt)]).to(device)

            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=50,
            )

            generated_text = tokenizer.decode(output_ids[0].tolist())

            try:
                if "Response:" in generated_text:
                    parts = generated_text.split("Response:", 1)
                    instruction = parts[0].replace("Instruction:", "").strip()
                    response = parts[1].strip()
                    generated.append({"instruction": instruction, "response": response})
            except Exception:  # noqa: S112 - skip malformed generations
                continue

    return generated


def evol_instruct(
    model,
    tokenizer,
    base_tasks: list[dict],
    num_iterations: int = 3,
    evolution_types: list[str] | None = None,
    temperature: float = 0.7,
) -> list[dict]:
    """Evolve instructions to increase complexity.

    Based on Evol-Instruct (Xu et al., 2023).
    Iteratively makes instructions more complex via model rewriting.

    Args:
        model: Trained GPT model
        tokenizer: Tokenizer instance
        base_tasks: Starting instruction-response pairs
        num_iterations: Evolution depth
        evolution_types: Types of evolution (default: all)
        temperature: Sampling temperature

    Returns:
        Evolved instruction-response pairs
    """
    if evolution_types is None:
        evolution_types = [
            "add_constraints",
            "deepen",
            "concretize",
            "increase_reasoning",
        ]

    evolution_prompts = {
        "add_constraints": "Rewrite this instruction by adding 2-3 additional constraints or requirements:\n\n{instruction}\n\nRewritten instruction:",
        "deepen": "Make this instruction require deeper reasoning or more steps:\n\n{instruction}\n\nDeepened instruction:",
        "concretize": "Make this instruction more specific with concrete examples:\n\n{instruction}\n\nConcrete instruction:",
        "increase_reasoning": "Rewrite to require multi-step reasoning:\n\n{instruction}\n\nReasoning instruction:",
    }

    model.eval()
    device = next(model.parameters()).device
    evolved = list(base_tasks)

    with torch.no_grad():
        for _ in range(num_iterations):
            new_evolved = []
            for task in random.sample(evolved, min(len(evolved), 100)):
                evo_type = random.choice(evolution_types)  # noqa: S311 - not crypto
                prompt = evolution_prompts[evo_type].format(instruction=task["instruction"])

                input_ids = torch.tensor([tokenizer.encode(prompt)]).to(device)
                output_ids = model.generate(
                    input_ids, max_new_tokens=256, temperature=temperature, top_k=40
                )

                evolved_instruction = tokenizer.decode(output_ids[0].tolist()).strip()
                new_evolved.append(
                    {"instruction": evolved_instruction, "response": task["response"]}
                )

            evolved.extend(new_evolved)

    return evolved


def generate_math_problems(
    model,
    tokenizer,
    difficulty: str = "medium",
    num_problems: int = 1000,
    topics: list[str] | None = None,
) -> list[dict]:
    """Generate synthetic math problems.

    Args:
        model: Trained GPT model
        tokenizer: Tokenizer instance
        difficulty: "easy", "medium", "hard"
        num_problems: Number to generate
        topics: Math topics (default: arithmetic, algebra, geometry)

    Returns:
        List of {"problem": str, "solution": str, "answer": str}
    """
    if topics is None:
        topics = ["arithmetic", "algebra", "geometry", "word_problems"]

    difficulty_prompts = {
        "easy": "single-step, basic",
        "medium": "multi-step, requires planning",
        "hard": "complex, requires advanced reasoning",
    }

    prompt_template = """Generate a {difficulty} {topic} math problem with step-by-step solution.

Problem:"""

    model.eval()
    device = next(model.parameters()).device
    problems = []

    with torch.no_grad():
        for _ in range(num_problems):
            topic = random.choice(topics)  # noqa: S311 - not crypto
            prompt = prompt_template.format(difficulty=difficulty_prompts[difficulty], topic=topic)

            input_ids = torch.tensor([tokenizer.encode(prompt)]).to(device)
            output_ids = model.generate(input_ids, max_new_tokens=512, temperature=0.7)

            generated = tokenizer.decode(output_ids[0].tolist())

            try:
                if "Solution:" in generated and "Answer:" in generated:
                    parts = generated.split("Solution:")
                    problem = parts[0].replace("Problem:", "").strip()
                    solution_answer = parts[1].split("Answer:")
                    solution = solution_answer[0].strip()
                    answer = solution_answer[1].strip()

                    problems.append({"problem": problem, "solution": solution, "answer": answer})
            except Exception:  # noqa: S112 - skip malformed generations
                continue

    return problems


def save_synthetic_dataset(data: list[dict], output_path: Path):
    """Save synthetic data to JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        f.writelines(json.dumps(item) + "\n" for item in data)
