from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .data import index_examples
from .prompts import OFFICIAL_SYSTEM_PROMPT, build_target_json, build_user_prompt
from .schemas import DIMENSIONS, EssayExample, RationaleRecord


@dataclass(slots=True)
class TrainingFeature:
    example_id: str
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    score_positions: list[int]
    continuous_scores: list[float]


def _score_character_positions(target_json: str) -> list[int]:
    positions: list[int] = []
    cursor = 0
    for dimension in DIMENSIONS:
        marker = f'"{dimension}":{{"score":'
        marker_start = target_json.find(marker, cursor)
        if marker_start < 0:
            raise ValueError(f"Could not find score marker for {dimension}")
        digit_position = marker_start + len(marker)
        if target_json[digit_position] not in "12345":
            raise ValueError(f"Score for {dimension} is not a one-digit integer")
        positions.append(digit_position)
        cursor = digit_position + 1
    return positions


def tokenize_training_example(
    example: EssayExample,
    record: RationaleRecord,
    tokenizer: Any,
    *,
    max_length: int,
) -> TrainingFeature:
    if not record.passed:
        raise ValueError(f"Rationale record did not pass: {record.example_id}")
    messages = [
        {"role": "system", "content": OFFICIAL_SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(example.prompt, example.essay)},
    ]
    prefix = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    target_json = build_target_json(record.integer_scores, record.rationales)
    full_text = prefix + target_json + "<|im_end|>\n"
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = list(encoded["input_ids"])
    if input_ids[: len(prefix_ids)] != list(prefix_ids):
        raise RuntimeError("Prompt tokenization is not a stable prefix of the target")
    if len(input_ids) > max_length:
        raise ValueError(
            f"{example.id} serializes to {len(input_ids)} tokens, over max_length={max_length}"
        )
    labels = [-100] * len(prefix_ids) + input_ids[len(prefix_ids) :]
    score_positions: list[int] = []
    offsets = encoded["offset_mapping"]
    for relative_char_position in _score_character_positions(target_json):
        absolute_position = len(prefix) + relative_char_position
        matching = [
            index
            for index, (start, end) in enumerate(offsets)
            if start <= absolute_position < end
        ]
        if len(matching) != 1:
            raise RuntimeError(
                f"Could not uniquely align score character for {example.id}: {matching}"
            )
        token_position = matching[0]
        decoded = tokenizer.decode([input_ids[token_position]])
        expected = target_json[relative_char_position]
        if expected not in decoded:
            raise RuntimeError(
                f"Score token is not independently represented: {decoded!r} vs {expected!r}"
            )
        score_positions.append(token_position)
    return TrainingFeature(
        example_id=example.id,
        input_ids=input_ids,
        attention_mask=[1] * len(input_ids),
        labels=labels,
        score_positions=score_positions,
        continuous_scores=[
            float(getattr(example.scores, dimension)) for dimension in DIMENSIONS
        ],
    )


def build_training_features(
    examples: Sequence[EssayExample],
    records: Sequence[RationaleRecord],
    tokenizer: Any,
    *,
    max_length: int,
) -> list[TrainingFeature]:
    records_by_id: dict[str, RationaleRecord] = {
        record.example_id: record for record in records if record.passed
    }
    features: list[TrainingFeature] = []
    for example in examples:
        record = records_by_id.get(example.id)
        if record is None:
            continue
        features.append(
            tokenize_training_example(
                example,
                record,
                tokenizer,
                max_length=max_length,
            )
        )
    if not features:
        raise ValueError("No passed rationale examples were tokenized")
    return features


class OrdinalDataCollator:
    def __init__(self, pad_token_id: int, pad_to_multiple_of: int = 8) -> None:
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: Sequence[TrainingFeature]) -> Mapping[str, Any]:
        import torch

        max_length = max(len(feature.input_ids) for feature in features)
        if self.pad_to_multiple_of:
            remainder = max_length % self.pad_to_multiple_of
            if remainder:
                max_length += self.pad_to_multiple_of - remainder
        input_ids: list[list[int]] = []
        attention_masks: list[list[int]] = []
        labels: list[list[int]] = []
        score_positions: list[list[int]] = []
        continuous_scores: list[list[float]] = []
        for feature in features:
            pad = max_length - len(feature.input_ids)
            input_ids.append(feature.input_ids + [self.pad_token_id] * pad)
            attention_masks.append(feature.attention_mask + [0] * pad)
            labels.append(feature.labels + [-100] * pad)
            score_positions.append(feature.score_positions)
            continuous_scores.append(feature.continuous_scores)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "score_positions": torch.tensor(score_positions, dtype=torch.long),
            "continuous_scores": torch.tensor(continuous_scores, dtype=torch.float32),
        }
