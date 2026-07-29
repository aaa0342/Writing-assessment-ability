from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .schemas import DIMENSIONS, EssayExample, RationaleRecord


def round_half_up(value: float) -> int:
    return max(1, min(5, int(math.floor(float(value) + 0.5))))


def integer_scores(example: EssayExample) -> dict[str, int]:
    return {
        dimension: round_half_up(getattr(example.scores, dimension))
        for dimension in DIMENSIONS
    }


def load_examples(path: str | Path) -> list[EssayExample]:
    path = Path(path)
    examples: list[EssayExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                examples.append(EssayExample.from_mapping(value))
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not examples:
        raise ValueError(f"No examples found in {path}")
    ids = [example.id for example in examples]
    if len(ids) != len(set(ids)):
        duplicates = [key for key, count in Counter(ids).items() if count > 1]
        raise ValueError(f"Duplicate ids in {path}: {duplicates[:5]}")
    return examples


def load_rationale_records(path: str | Path) -> list[RationaleRecord]:
    path = Path(path)
    records: list[RationaleRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(RationaleRecord.from_mapping(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return records


def save_jsonl(path: str | Path, rows: Iterable[Mapping[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def shuffled(examples: Sequence[EssayExample], seed: int = 42) -> list[EssayExample]:
    result = list(examples)
    random.Random(seed).shuffle(result)
    return result


def index_examples(examples: Sequence[EssayExample]) -> dict[str, EssayExample]:
    return {example.id: example for example in examples}


def validate_rationale_coverage(
    examples: Sequence[EssayExample],
    records: Sequence[RationaleRecord],
    minimum_keep_rate: float = 0.90,
) -> dict[str, float | int]:
    expected = {example.id for example in examples}
    passed = {record.example_id for record in records if record.passed}
    unknown = passed.difference(expected)
    if unknown:
        raise ValueError(f"Rationale records contain unknown ids: {sorted(unknown)[:5]}")
    keep_rate = len(passed) / len(expected)
    if keep_rate < minimum_keep_rate:
        raise RuntimeError(
            f"Rationale keep rate {keep_rate:.3%} is below {minimum_keep_rate:.1%}"
        )
    return {
        "examples": len(expected),
        "passed": len(passed),
        "keep_rate": keep_rate,
    }
