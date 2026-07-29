from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .data import integer_scores, round_half_up, save_jsonl
from .generation import ChatGenerator, TransformersChatGenerator, parse_exact_json_object
from .prompts import OFFICIAL_SYSTEM_PROMPT, build_submission_chat_template, build_user_prompt
from .schemas import DIMENSIONS, EssayExample


@dataclass(frozen=True, slots=True)
class ParsedPrediction:
    scores: dict[str, int]
    rationales: dict[str, str]
    raw: str
    attempts: int
    valid: bool
    error: str = ""

    def as_dict(self, example_id: str) -> dict[str, Any]:
        return {
            "essay_id": example_id,
            "judge": {
                dimension: {
                    "score": self.scores[dimension],
                    "rationale": self.rationales[dimension],
                }
                for dimension in DIMENSIONS
            },
            "_valid": self.valid,
            "_attempts": self.attempts,
            "_error": self.error,
            "_raw": self.raw,
        }


def parse_competition_output(text: str) -> ParsedPrediction:
    value = parse_exact_json_object(text)
    if set(value) != set(DIMENSIONS):
        raise ValueError(f"Top-level keys must be exactly {DIMENSIONS}")
    scores: dict[str, int] = {}
    rationales: dict[str, str] = {}
    for dimension in DIMENSIONS:
        section = value[dimension]
        if not isinstance(section, Mapping) or set(section) != {"score", "rationale"}:
            raise ValueError(
                f"{dimension} must contain exactly score and rationale"
            )
        score = section["score"]
        if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
            raise ValueError(f"{dimension}.score must be an integer in [1, 5]")
        rationale = section["rationale"]
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError(f"{dimension}.rationale must be a non-empty string")
        if "```" in rationale or "<think>" in rationale:
            raise ValueError(f"Forbidden formatting in {dimension}.rationale")
        scores[dimension] = score
        rationales[dimension] = rationale.strip()
    return ParsedPrediction(
        scores=scores,
        rationales=rationales,
        raw=text,
        attempts=1,
        valid=True,
    )


def predict_example(
    generator: ChatGenerator,
    example: EssayExample,
    *,
    max_new_tokens: int = 512,
    retries: int = 2,
) -> ParsedPrediction:
    messages = [
        {"role": "system", "content": OFFICIAL_SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(example.prompt, example.essay)},
    ]
    last_raw = ""
    last_error = ""
    for attempt in range(1, retries + 2):
        try:
            last_raw = generator.generate(messages, max_new_tokens=max_new_tokens)
            prediction = parse_competition_output(last_raw)
            return ParsedPrediction(
                scores=prediction.scores,
                rationales=prediction.rationales,
                raw=last_raw,
                attempts=attempt,
                valid=True,
            )
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
    return ParsedPrediction(
        scores={dimension: 0 for dimension in DIMENSIONS},
        rationales={dimension: "" for dimension in DIMENSIONS},
        raw=last_raw,
        attempts=retries + 1,
        valid=False,
        error=last_error,
    )


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average_rank = (cursor + 1 + end) / 2
        for position in range(cursor, end):
            ranks[order[position]] = average_rank
        cursor = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
    )
    left_var = sum((x - left_mean) ** 2 for x in left)
    right_var = sum((y - right_mean) ** 2 for y in right)
    if left_var == 0 or right_var == 0:
        return 0.0
    return numerator / math.sqrt(left_var * right_var)


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    return _pearson(_average_ranks(left), _average_ranks(right))


def score_predictions(
    examples: Sequence[EssayExample],
    predictions: Sequence[ParsedPrediction],
) -> dict[str, Any]:
    if len(examples) != len(predictions):
        raise ValueError("Examples and predictions have different lengths")
    metrics: dict[str, Any] = {
        "examples": len(examples),
        "valid": sum(prediction.valid for prediction in predictions),
        "parse_rate": sum(prediction.valid for prediction in predictions)
        / max(len(predictions), 1),
        "first_attempt_rate": sum(
            prediction.valid and prediction.attempts == 1 for prediction in predictions
        )
        / max(len(predictions), 1),
        "dimensions": {},
    }
    for dimension in DIMENSIONS:
        gold = [float(getattr(example.scores, dimension)) for example in examples]
        predicted = [float(prediction.scores[dimension]) for prediction in predictions]
        rmse = math.sqrt(
            sum((target - value) ** 2 for target, value in zip(gold, predicted))
            / len(gold)
        )
        metrics["dimensions"][dimension] = {
            "rmse": rmse,
            "spearman": spearman(gold, predicted),
        }
    metrics["macro_rmse"] = sum(
        metrics["dimensions"][dimension]["rmse"] for dimension in DIMENSIONS
    ) / len(DIMENSIONS)
    metrics["macro_spearman"] = sum(
        metrics["dimensions"][dimension]["spearman"] for dimension in DIMENSIONS
    ) / len(DIMENSIONS)
    return metrics


def integer_oracle_metrics(examples: Sequence[EssayExample]) -> dict[str, Any]:
    predictions = [
        ParsedPrediction(
            scores=integer_scores(example),
            rationales={dimension: "oracle" for dimension in DIMENSIONS},
            raw="",
            attempts=1,
            valid=True,
        )
        for example in examples
    ]
    return score_predictions(examples, predictions)


def evaluate_generator(
    generator: ChatGenerator,
    examples: Sequence[EssayExample],
    *,
    output_path: str | Path | None = None,
    max_new_tokens: int = 512,
    retries: int = 2,
) -> tuple[list[ParsedPrediction], dict[str, Any]]:
    predictions: list[ParsedPrediction] = []
    for index, example in enumerate(examples, 1):
        prediction = predict_example(
            generator,
            example,
            max_new_tokens=max_new_tokens,
            retries=retries,
        )
        predictions.append(prediction)
        print(
            f"[{index}/{len(examples)}] {example.id}: "
            f"valid={prediction.valid} attempts={prediction.attempts}"
        )
    metrics = score_predictions(examples, predictions)
    if output_path is not None:
        save_jsonl(
            output_path,
            (
                prediction.as_dict(example.document_id)
                for example, prediction in zip(examples, predictions)
            ),
        )
        report_path = Path(output_path).with_suffix(".metrics.json")
        report_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return predictions, metrics


def load_adapter_generator(
    base_model_id: str,
    adapter_path: str | Path,
) -> TransformersChatGenerator:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    adapter_path = str(adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path, trust_remote_code=False, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = AutoModelForImageTextToText.from_pretrained(
        base_model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return TransformersChatGenerator(model=model, tokenizer=tokenizer)


def compare_prediction_sets(
    left: Sequence[ParsedPrediction],
    right: Sequence[ParsedPrediction],
) -> dict[str, Any]:
    if len(left) != len(right):
        raise ValueError("Prediction sets have different lengths")
    score_matches = 0
    exact_matches = 0
    valid_matches = 0
    mismatches: list[int] = []
    for index, (first, second) in enumerate(zip(left, right)):
        same_scores = first.scores == second.scores
        same_raw = first.raw.strip() == second.raw.strip()
        same_valid = first.valid == second.valid
        score_matches += same_scores
        exact_matches += same_raw
        valid_matches += same_valid
        if not (same_scores and same_valid):
            mismatches.append(index)
    total = len(left)
    return {
        "examples": total,
        "score_match_rate": score_matches / max(total, 1),
        "exact_text_match_rate": exact_matches / max(total, 1),
        "validity_match_rate": valid_matches / max(total, 1),
        "mismatch_indices": mismatches,
    }


def choose_demo_order(
    generator: TransformersChatGenerator,
    examples: Sequence[EssayExample],
    demo_orders: Sequence[Sequence[Mapping[str, str]]],
    *,
    limit: int | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """Evaluate all six orders; highest parse rate, then equal metric rank wins."""

    subset = list(examples[:limit] if limit else examples)
    reports: list[dict[str, Any]] = []
    original_template = generator.tokenizer.chat_template
    try:
        for index, order in enumerate(demo_orders):
            generator.tokenizer.chat_template = build_submission_chat_template(order)
            _, metrics = evaluate_generator(
                generator,
                subset,
                max_new_tokens=512,
                retries=0,
            )
            reports.append({"order": index, **metrics})
    finally:
        generator.tokenizer.chat_template = original_template
    rmse_ranks = _average_ranks([report["macro_rmse"] for report in reports])
    inverse_spearman_ranks = _average_ranks(
        [-report["macro_spearman"] for report in reports]
    )
    for report, rmse_rank, spearman_rank in zip(
        reports, rmse_ranks, inverse_spearman_ranks
    ):
        report["metric_rank"] = 0.5 * rmse_rank + 0.5 * spearman_rank
    winner = min(
        reports,
        key=lambda report: (
            -report["first_attempt_rate"],
            report["metric_rank"],
            report["macro_rmse"],
        ),
    )
    return [dict(item) for item in demo_orders[winner["order"]]], reports
