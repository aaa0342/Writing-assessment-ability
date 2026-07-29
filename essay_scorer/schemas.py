from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

DIMENSIONS = ("content", "organization", "expression")
JUDGE_ITEMS = (
    "domain_match",
    "score_rationale_consistency",
    "specificity",
    "groundedness",
)


@dataclass(frozen=True, slots=True)
class ScoreTriplet:
    content: float
    organization: float
    expression: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ScoreTriplet":
        scores = {dimension: float(value[dimension]) for dimension in DIMENSIONS}
        for dimension, score in scores.items():
            if not 1.0 <= score <= 5.0:
                raise ValueError(f"{dimension} score outside [1, 5]: {score}")
        return cls(**scores)

    def as_dict(self) -> dict[str, float]:
        return {dimension: float(getattr(self, dimension)) for dimension in DIMENSIONS}

    def mean(self) -> float:
        return sum(self.as_dict().values()) / len(DIMENSIONS)


@dataclass(frozen=True, slots=True)
class EssayExample:
    id: str
    document_id: str
    prompt_num: str
    prompt: str
    essay: str
    scores: ScoreTriplet

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EssayExample":
        required = {"id", "document_id", "prompt_num", "prompt", "essay", "score"}
        missing = required.difference(value)
        if missing:
            raise ValueError(f"Missing fields: {sorted(missing)}")
        return cls(
            id=str(value["id"]),
            document_id=str(value["document_id"]),
            prompt_num=str(value["prompt_num"]),
            prompt=str(value["prompt"]).strip(),
            essay=str(value["essay"]).strip(),
            scores=ScoreTriplet.from_mapping(value["score"]),
        )


@dataclass(frozen=True, slots=True)
class RationaleRecord:
    example_id: str
    integer_scores: Mapping[str, int]
    rationales: Mapping[str, str]
    evidence: Mapping[str, Any]
    critic: Mapping[str, Any]
    critic_mean: float
    attempts: int
    passed: bool

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RationaleRecord":
        return cls(
            example_id=str(value["example_id"]),
            integer_scores={
                dimension: int(value["integer_scores"][dimension]) for dimension in DIMENSIONS
            },
            rationales={
                dimension: str(value["rationales"][dimension]) for dimension in DIMENSIONS
            },
            evidence=dict(value.get("evidence", {})),
            critic=dict(value.get("critic", {})),
            critic_mean=float(value.get("critic_mean", 0.0)),
            attempts=int(value.get("attempts", 0)),
            passed=bool(value.get("passed", False)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "integer_scores": dict(self.integer_scores),
            "rationales": dict(self.rationales),
            "evidence": dict(self.evidence),
            "critic": dict(self.critic),
            "critic_mean": self.critic_mean,
            "attempts": self.attempts,
            "passed": self.passed,
        }
