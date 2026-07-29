from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .generation import ChatGenerator, parse_exact_json_object
from .prompts import LLM_JUDGE_SYSTEM_PROMPT, build_critic_user_prompt
from .rationales import validate_critic
from .schemas import DIMENSIONS, JUDGE_ITEMS, EssayExample


def load_saved_predictions(path: str | Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            essay_id = str(value["essay_id"])
            judge = value["judge"]
            if not isinstance(judge, Mapping):
                raise ValueError(f"{path}:{line_number}: invalid judge object")
            result[essay_id] = dict(judge)
    return result


def audit_rationales(
    generator: ChatGenerator,
    examples: Sequence[EssayExample],
    prediction_path: str | Path,
    output_path: str | Path,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    predictions = load_saved_predictions(prediction_path)
    rows: list[dict[str, Any]] = []
    item_scores = {
        dimension: {item: [] for item in JUDGE_ITEMS} for dimension in DIMENSIONS
    }
    selected = list(examples[:limit] if limit else examples)
    for index, example in enumerate(selected, 1):
        prediction = predictions.get(example.document_id)
        if prediction is None:
            raise ValueError(f"Missing prediction for {example.document_id}")
        scores = {
            dimension: int(prediction[dimension]["score"]) for dimension in DIMENSIONS
        }
        rationales = {
            dimension: str(prediction[dimension]["rationale"])
            for dimension in DIMENSIONS
        }
        raw = generator.generate(
            [
                {"role": "system", "content": LLM_JUDGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_critic_user_prompt(
                        example.prompt,
                        example.essay,
                        scores,
                        rationales,
                    ),
                },
            ],
            max_new_tokens=1200,
        )
        critic, critic_mean, _ = validate_critic(
            parse_exact_json_object(raw),
            minimum_item_score=1,
            minimum_mean_score=1.0,
        )
        for dimension in DIMENSIONS:
            for item in JUDGE_ITEMS:
                item_scores[dimension][item].append(
                    critic[dimension][item]["score"]
                )
        rows.append(
            {
                "essay_id": example.document_id,
                "critic_mean": critic_mean,
                "critic": critic,
            }
        )
        print(f"[{index}/{len(selected)}] {example.id}: judge={critic_mean:.2f}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    report = {
        "examples": len(rows),
        "mean": sum(row["critic_mean"] for row in rows) / max(len(rows), 1),
        "items": {
            dimension: {
                item: sum(scores) / max(len(scores), 1)
                for item, scores in item_scores[dimension].items()
            }
            for dimension in DIMENSIONS
        },
        "below_3_rate": sum(
            score < 3
            for dimension in DIMENSIONS
            for item in JUDGE_ITEMS
            for score in item_scores[dimension][item]
        )
        / max(len(rows) * len(DIMENSIONS) * len(JUDGE_ITEMS), 1),
    }
    output_path.with_suffix(".metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report
