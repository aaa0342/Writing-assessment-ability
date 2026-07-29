from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .data import integer_scores, load_rationale_records
from .generation import ChatGenerator, parse_exact_json_object
from .prompts import (
    LLM_JUDGE_SYSTEM_PROMPT,
    build_critic_user_prompt,
    build_evidence_prompt,
    build_rationale_prompt,
)
from .schemas import DIMENSIONS, JUDGE_ITEMS, EssayExample, RationaleRecord

EVIDENCE_SYSTEM_PROMPT = """너는 한국어 논증문에서 검증 가능한 채점 증거만 추출하는 분석가이다.
반드시 제공된 essay_text 내부의 연속된 원문 구절만 quote로 사용하고 JSON 객체 하나만 출력하라."""

RATIONALE_SYSTEM_PROMPT = """너는 한국어 논증문 채점 근거 작성자이다.
주어진 점수는 변경하지 않는다. 검증된 증거만 사용해 영역별 근거를 쓰고 JSON 객체 하나만 출력하라."""


def _validate_evidence(value: Mapping[str, Any], essay: str) -> dict[str, Any]:
    validated: dict[str, Any] = {}
    for dimension in DIMENSIONS:
        section = value.get(dimension)
        if not isinstance(section, Mapping):
            raise ValueError(f"Missing evidence section: {dimension}")
        validated[dimension] = {}
        evidence_count = 0
        for polarity in ("strengths", "weaknesses"):
            raw_items = section.get(polarity, [])
            if not isinstance(raw_items, list):
                raise ValueError(f"{dimension}.{polarity} must be a list")
            items: list[dict[str, str]] = []
            for raw in raw_items[:3]:
                if not isinstance(raw, Mapping):
                    continue
                quote = str(raw.get("quote", "")).strip()
                observation = str(raw.get("observation", "")).strip()
                if not quote or not observation:
                    continue
                if not 4 <= len(quote) <= 30:
                    continue
                if quote not in essay:
                    raise ValueError(f"Hallucinated quote in {dimension}: {quote!r}")
                items.append({"quote": quote, "observation": observation})
                evidence_count += 1
            validated[dimension][polarity] = items
        if evidence_count == 0:
            raise ValueError(f"No verifiable evidence for {dimension}")
    return validated


def _validate_rationales(value: Mapping[str, Any]) -> dict[str, str]:
    if set(value) != set(DIMENSIONS):
        raise ValueError(f"Rationale keys must be exactly {DIMENSIONS}")
    result: dict[str, str] = {}
    for dimension in DIMENSIONS:
        rationale = str(value[dimension]).strip()
        if not 20 <= len(rationale) <= 260:
            raise ValueError(
                f"{dimension} rationale length must be between 20 and 260 characters"
            )
        if "```" in rationale or "<think>" in rationale:
            raise ValueError(f"Forbidden formatting in {dimension} rationale")
        result[dimension] = rationale
    return result


def validate_critic(
    value: Mapping[str, Any],
    *,
    minimum_item_score: int = 4,
    minimum_mean_score: float = 4.25,
) -> tuple[dict[str, Any], float, bool]:
    normalized: dict[str, Any] = {}
    scores: list[int] = []
    for dimension in DIMENSIONS:
        section = value.get(dimension)
        if not isinstance(section, Mapping):
            raise ValueError(f"Missing critic section: {dimension}")
        normalized[dimension] = {}
        for item in JUDGE_ITEMS:
            judgment = section.get(item)
            if not isinstance(judgment, Mapping):
                raise ValueError(f"Missing critic item: {dimension}.{item}")
            score = judgment.get("score")
            if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
                raise ValueError(f"Invalid critic score: {dimension}.{item}={score!r}")
            evidence = str(judgment.get("evidence", "")).strip()
            normalized[dimension][item] = {"evidence": evidence, "score": score}
            scores.append(score)
    mean_score = sum(scores) / len(scores)
    passed = min(scores) >= minimum_item_score and mean_score >= minimum_mean_score
    return normalized, mean_score, passed


def critic_feedback(critic: Mapping[str, Any]) -> str:
    problems: list[str] = []
    for dimension in DIMENSIONS:
        for item in JUDGE_ITEMS:
            judgment = critic.get(dimension, {}).get(item, {})
            score = judgment.get("score", 0)
            if score < 4:
                problems.append(
                    f"{dimension}.{item}={score}: {judgment.get('evidence', '근거 부족')}"
                )
    return "\n".join(problems)


class RationaleBuilder:
    def __init__(
        self,
        generator: ChatGenerator,
        *,
        max_attempts: int = 3,
        minimum_item_score: int = 4,
        minimum_mean_score: float = 4.25,
    ) -> None:
        self.generator = generator
        self.max_attempts = max_attempts
        self.minimum_item_score = minimum_item_score
        self.minimum_mean_score = minimum_mean_score

    def build(self, example: EssayExample) -> RationaleRecord:
        scores = integer_scores(example)
        last_evidence: dict[str, Any] = {}
        last_rationales = {dimension: "" for dimension in DIMENSIONS}
        last_critic: dict[str, Any] = {}
        last_mean = 0.0
        feedback = ""
        for attempt in range(1, self.max_attempts + 1):
            try:
                evidence_text = self.generator.generate(
                    [
                        {"role": "system", "content": EVIDENCE_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": build_evidence_prompt(
                                example.prompt,
                                example.essay,
                                scores,
                                previous_feedback=feedback,
                            ),
                        },
                    ],
                    max_new_tokens=900,
                )
                last_evidence = _validate_evidence(
                    parse_exact_json_object(evidence_text), example.essay
                )
                rationale_text = self.generator.generate(
                    [
                        {"role": "system", "content": RATIONALE_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": build_rationale_prompt(
                                example.prompt,
                                example.essay,
                                scores,
                                last_evidence,
                                previous_feedback=feedback,
                            ),
                        },
                    ],
                    max_new_tokens=500,
                )
                last_rationales = _validate_rationales(
                    parse_exact_json_object(rationale_text)
                )
                critic_text = self.generator.generate(
                    [
                        {"role": "system", "content": LLM_JUDGE_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": build_critic_user_prompt(
                                example.prompt,
                                example.essay,
                                scores,
                                last_rationales,
                            ),
                        },
                    ],
                    max_new_tokens=1200,
                )
                last_critic, last_mean, passed = validate_critic(
                    parse_exact_json_object(critic_text),
                    minimum_item_score=self.minimum_item_score,
                    minimum_mean_score=self.minimum_mean_score,
                )
                if passed:
                    return RationaleRecord(
                        example_id=example.id,
                        integer_scores=scores,
                        rationales=last_rationales,
                        evidence=last_evidence,
                        critic=last_critic,
                        critic_mean=last_mean,
                        attempts=attempt,
                        passed=True,
                    )
                feedback = critic_feedback(last_critic)
            except Exception as exc:
                feedback = f"이전 생성 검증 실패: {type(exc).__name__}: {exc}"
        return RationaleRecord(
            example_id=example.id,
            integer_scores=scores,
            rationales=last_rationales,
            evidence=last_evidence,
            critic=last_critic,
            critic_mean=last_mean,
            attempts=self.max_attempts,
            passed=False,
        )


def generate_rationale_file(
    examples: Sequence[EssayExample],
    builder: RationaleBuilder,
    output_path: str | Path,
    *,
    resume: bool = True,
) -> dict[str, int | float]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, RationaleRecord] = {}
    if resume and output_path.is_file():
        existing = {
            record.example_id: record for record in load_rationale_records(output_path)
        }
    mode = "a" if existing else "w"
    with output_path.open(mode, encoding="utf-8", newline="\n") as handle:
        for index, example in enumerate(examples, 1):
            if example.id in existing and existing[example.id].passed:
                continue
            record = builder.build(example)
            handle.write(json.dumps(record.as_dict(), ensure_ascii=False) + "\n")
            handle.flush()
            existing[example.id] = record
            print(
                f"[{index}/{len(examples)}] {example.id}: "
                f"passed={record.passed} critic_mean={record.critic_mean:.2f}"
            )
    passed = sum(record.passed for record in existing.values())
    return {
        "total": len(existing),
        "passed": passed,
        "failed": len(existing) - passed,
        "keep_rate": passed / len(existing) if existing else 0.0,
    }
