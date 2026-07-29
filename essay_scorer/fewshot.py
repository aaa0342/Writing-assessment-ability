from __future__ import annotations

import itertools
import json
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from .data import index_examples
from .prompts import build_target_json, build_user_prompt
from .schemas import DIMENSIONS, EssayExample, RationaleRecord


def select_balanced_demos(
    examples: Sequence[EssayExample],
    records: Sequence[RationaleRecord],
) -> list[dict[str, str]]:
    """Select low/mid/high demonstrations without using validation examples."""

    examples_by_id = index_examples(examples)
    candidates = [
        record
        for record in records
        if record.passed and record.example_id in examples_by_id
    ]
    if len(candidates) < 3:
        raise ValueError("At least three passed rationale records are required")
    lengths = sorted(len(examples_by_id[record.example_id].essay) for record in candidates)
    q1 = lengths[len(lengths) // 4]
    q3 = lengths[(3 * len(lengths)) // 4]
    median_length = statistics.median(lengths)
    targets = (2.25, 3.25, 4.25)
    chosen: list[RationaleRecord] = []
    used_prompt_nums: set[str] = set()
    for target in targets:
        eligible = []
        for record in candidates:
            example = examples_by_id[record.example_id]
            if record in chosen or example.prompt_num in used_prompt_nums:
                continue
            if not q1 <= len(example.essay) <= q3:
                continue
            mean_score = sum(record.integer_scores[d] for d in DIMENSIONS) / 3
            objective = (
                abs(mean_score - target)
                + 0.15 * abs(len(example.essay) - median_length) / max(median_length, 1)
                - 0.08 * record.critic_mean
            )
            eligible.append((objective, example.id, record))
        if not eligible:
            raise RuntimeError(f"Could not select a balanced demo for target {target}")
        _, _, selected = min(eligible, key=lambda item: (item[0], item[1]))
        chosen.append(selected)
        used_prompt_nums.add(examples_by_id[selected.example_id].prompt_num)

    demos: list[dict[str, str]] = []
    for band, record in zip(("low", "mid", "high"), chosen):
        example = examples_by_id[record.example_id]
        demos.append(
            {
                "band": band,
                "example_id": example.id,
                "prompt_num": example.prompt_num,
                "user": build_user_prompt(example.prompt, example.essay),
                "assistant": build_target_json(
                    record.integer_scores, record.rationales
                ),
            }
        )
    return demos


def all_demo_orders(demos: Sequence[Mapping[str, str]]) -> list[list[dict[str, str]]]:
    if len(demos) != 3:
        raise ValueError("Exactly three demos are required")
    return [[dict(item) for item in order] for order in itertools.permutations(demos)]


def save_demos(path: str | Path, demos: Sequence[Mapping[str, str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([dict(demo) for demo in demos], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_demos(path: str | Path) -> list[dict[str, str]]:
    value: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("Demo file must contain exactly three examples")
    demos = [dict(item) for item in value]
    for demo in demos:
        if not {"user", "assistant"}.issubset(demo):
            raise ValueError("Each demo needs user and assistant fields")
    return demos
