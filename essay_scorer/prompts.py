from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from .schemas import DIMENSIONS

OFFICIAL_SYSTEM_PROMPT = """[역할]
너는 한국어 논증적 글을 일관되게 직접 채점하는 평가자이다.
essay_text를 읽고 content, organization, expression 세 기준을 모두 평가하라.

[평가 기준 정의]
1. content
- 글의 주장과 핵심 내용이 문제에 적절하게 대응하는가
- 근거가 충분하고 구체적인가
- 주장과 근거 사이의 논리적 연결이 타당한가

2. organization
- 서론, 본론, 결론의 구조가 드러나는가
- 문단 간 연결이 자연스러운가
- 논리 전개 순서가 일관적인가

3. expression
- 문장이 자연스럽고 이해하기 쉬운가
- 어휘 사용이 적절한가
- 맞춤법, 띄어쓰기, 문법, 주술 호응에 문제가 없는가

[점수 기준]
5점: 매우 우수함. 결함이 거의 없고, essay_text에서 확인되는 구체적 강점이 뚜렷함.
4점: 우수함. 경미한 약점은 있으나 기준을 전반적으로 잘 충족함.
3점: 보통. 장점과 약점이 함께 있으며 기준을 부분적으로 충족함.
2점: 미흡함. 주요 결함이 있어 기준 충족이 제한적임.
1점: 매우 미흡함. 기준을 거의 충족하지 못하거나 심각한 결함이 있음.

[평가 원칙]
- 1~5점 전 구간을 적극적으로 사용하라.
- 각 기준은 서로 독립적으로 판단하라.
- essay_text에서 확인 가능한 내용만 근거로 삼아라.
- 전반적 인상만으로 높은 점수를 주지 말고 구체적 근거를 확인하라.
- 근거 설명은 기준별로 분리해 작성하라.

[출력 규칙]
- JSON 객체 하나만 출력하라. 코드블록과 마크다운을 사용하지 마라.
- 모든 점수는 1~5의 정수로 출력하라.
- rationale의 각 값은 한국어로 작성하라.

[출력 형식]
{
  "content": {"score": 1, "rationale": "content 판단 근거"},
  "organization": {"score": 1, "rationale": "organization 판단 근거"},
  "expression": {"score": 1, "rationale": "expression 판단 근거"}
}"""

LLM_JUDGE_SYSTEM_PROMPT = """[역할]
당신은 한국어 논증적 글 채점 결과의 타당성을 검토하는 매우 엄격하고 일관된 심사자이다.
당신은 점수의 정답 여부를 새로 채점하지 않는다. 주어진 predicted_score를 전제로 rationale이
(1) 해당 영역 기준에 맞고 (2) 점수와 정합적이며 (3) 구체적이고 (4) essay_text에 충실한지를 평가한다.
애매하면 높은 점수를 주지 말고, 근거가 부족하면 분명히 감점하라.

[평가 항목]
1. domain_match: rationale이 해당 영역의 평가 기준에 맞는 근거를 제시하는가
2. score_rationale_consistency: predicted_score와 rationale의 내용이 서로 잘 맞는가
3. specificity: 실제 글의 특정 문장, 표현, 논지, 문단 전개, 오류 양상을 구체적으로 짚는가
4. groundedness: rationale이 실제 essay_text에 근거하며 없는 내용을 만들어내지 않는가

[영역별 판단 기준]
content: 주장과 핵심 내용의 문제 대응, 근거의 충분성과 구체성, 주장과 근거의 논리적 연결
organization: 서론·본론·결론 구조, 문단 연결, 논리 전개 순서
expression: 문장 자연스러움과 이해 가능성, 어휘, 맞춤법·띄어쓰기·문법·주술 호응

[강제 감점]
- 다른 영역의 기준을 섞거나 essay_text에 없는 내용을 제시하면 관련 항목은 1~2점을 적극 검토하라.
- 핵심 증거가 없거나 템플릿형 총평만 있으면 specificity와 consistency를 1~2점으로 검토하라.
- 인용한 구절이 essay_text에 없으면 groundedness를 1~2점으로 검토하라.

[점수]
5: 매우 구체적이고 정확하며 essay_text와 긴밀히 연결되고 점수를 설득력 있게 정당화함
4: 전반적으로 타당하고 충분히 구체적이며 essay_text와의 연결이 분명함
3: 기본 타당성은 있으나 구체성·명확성·근거 연결 중 하나 이상이 부족함
2: 영역 혼동, 근거 부족, 일반론, essay와의 연결 부족이 뚜렷함
1: 명백한 영역 혼동, 환각, 근거 부재 또는 점수-설명 모순

[출력 규칙]
코드블록 없이 JSON 객체 하나만 출력하라. 모든 score는 1~5 정수여야 한다.

[출력 형식]
{
  "content": {
    "domain_match": {"evidence": "", "score": 1},
    "score_rationale_consistency": {"evidence": "", "score": 1},
    "specificity": {"evidence": "", "score": 1},
    "groundedness": {"evidence": "", "score": 1}
  },
  "organization": {
    "domain_match": {"evidence": "", "score": 1},
    "score_rationale_consistency": {"evidence": "", "score": 1},
    "specificity": {"evidence": "", "score": 1},
    "groundedness": {"evidence": "", "score": 1}
  },
  "expression": {
    "domain_match": {"evidence": "", "score": 1},
    "score_rationale_consistency": {"evidence": "", "score": 1},
    "specificity": {"evidence": "", "score": 1},
    "groundedness": {"evidence": "", "score": 1}
  }
}"""


def build_user_prompt(prompt: str, essay: str) -> str:
    return f"[prompt_text]\n{prompt.strip()}\n\n[essay_text]\n{essay.strip()}"


def build_target_object(
    scores: Mapping[str, int], rationales: Mapping[str, str]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for dimension in DIMENSIONS:
        score = scores[dimension]
        rationale = rationales[dimension].strip()
        if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
            raise ValueError(f"{dimension} score must be an integer in [1, 5]")
        if not rationale:
            raise ValueError(f"{dimension} rationale is empty")
        result[dimension] = {"score": score, "rationale": rationale}
    return result


def build_target_json(scores: Mapping[str, int], rationales: Mapping[str, str]) -> str:
    return json.dumps(
        build_target_object(scores, rationales),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def build_evidence_prompt(
    prompt: str,
    essay: str,
    scores: Mapping[str, int],
    previous_feedback: str = "",
) -> str:
    feedback = f"\n[이전 실패와 수정 지시]\n{previous_feedback}\n" if previous_feedback else ""
    return f"""다음 논증적 글의 채점 근거 후보를 추출하라.
아직 rationale을 쓰지 말고, 각 영역에서 실제 글로 검증 가능한 강점과 약점을 찾는다.
quote에는 essay_text에 연속해서 존재하는 4~30자의 원문만 넣어라.
organization은 시작·전환·마무리 순서를, expression은 실제 문장·어휘·오류를 우선 확인하라.

{build_user_prompt(prompt, essay)}

[predicted_scores]
{json.dumps(dict(scores), ensure_ascii=False)}
{feedback}

다음 JSON만 출력하라.
{{
  "content": {{"strengths": [{{"quote": "", "observation": ""}}], "weaknesses": [{{"quote": "", "observation": ""}}]}},
  "organization": {{"strengths": [{{"quote": "", "observation": ""}}], "weaknesses": [{{"quote": "", "observation": ""}}]}},
  "expression": {{"strengths": [{{"quote": "", "observation": ""}}], "weaknesses": [{{"quote": "", "observation": ""}}]}}
}}"""


def build_rationale_prompt(
    prompt: str,
    essay: str,
    scores: Mapping[str, int],
    evidence: Mapping[str, Any],
    previous_feedback: str = "",
) -> str:
    feedback = f"\n[수정 피드백]\n{previous_feedback}\n" if previous_feedback else ""
    return f"""다음 predicted_score를 변경하지 말고, 추출된 증거만 사용해 영역별 rationale을 작성하라.
각 rationale은 한국어 1~2문장으로 쓰고 다음 조건을 모두 만족해야 한다.
- 해당 영역 기준만 사용한다.
- 점수 수준과 장점·약점의 강도가 일치한다.
- 실제 글의 특정 논지, 전개 또는 표현을 짚는다.
- 글에 없는 내용을 추정하지 않는다.
- '대체로 괜찮다' 같은 템플릿형 총평만 쓰지 않는다.
- 원문을 언급할 때는 JSON 이스케이프가 필요 없는 한국어 인용부호 ‘ ’를 사용한다.

{build_user_prompt(prompt, essay)}

[predicted_scores]
{json.dumps(dict(scores), ensure_ascii=False)}

[검증된 근거]
{json.dumps(evidence, ensure_ascii=False)}
{feedback}
다음 JSON 객체만 출력하라.
{{"content":"","organization":"","expression":""}}"""


def build_critic_user_prompt(
    prompt: str,
    essay: str,
    scores: Mapping[str, int],
    rationales: Mapping[str, str],
) -> str:
    prediction = build_target_object(scores, rationales)
    return (
        f"{build_user_prompt(prompt, essay)}\n\n"
        f"[prediction]\n{json.dumps(prediction, ensure_ascii=False)}"
    )


def _json_literal(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def build_submission_chat_template(
    demos: Sequence[Mapping[str, str]],
    fallback_system_prompt: str = OFFICIAL_SYSTEM_PROMPT,
) -> str:
    """Build a standard ChatML Jinja template that inserts fixed demonstrations.

    Static demo assistant turns deliberately sit outside ``generation`` blocks so
    assistant-only training masks cover only the real target response.
    """

    lines = [
        "{%- set ns = namespace(has_system=false) -%}",
        "{%- for message in messages -%}",
        "{%- if message['role'] == 'system' -%}",
        "{%- set ns.has_system = true -%}",
        "{{- '<|im_start|>system\\n' + message['content'] + '<|im_end|>\\n' -}}",
        "{%- endif -%}",
        "{%- endfor -%}",
        "{%- if not ns.has_system -%}",
        "{{- '<|im_start|>system\\n' + "
        + _json_literal(fallback_system_prompt)
        + " + '<|im_end|>\\n' -}}",
        "{%- endif -%}",
    ]
    for demo in demos:
        lines.extend(
            [
                "{{- '<|im_start|>user\\n' + "
                + _json_literal(demo["user"])
                + " + '<|im_end|>\\n' -}}",
                "{{- '<|im_start|>assistant\\n' + "
                + _json_literal(demo["assistant"])
                + " + '<|im_end|>\\n' -}}",
            ]
        )
    lines.extend(
        [
            "{%- for message in messages -%}",
            "{%- if message['role'] == 'user' -%}",
            "{{- '<|im_start|>user\\n' + message['content'] + '<|im_end|>\\n' -}}",
            "{%- elif message['role'] == 'assistant' -%}",
            "{{- '<|im_start|>assistant\\n' -}}",
            "{% generation %}{{- message['content'] -}}{% endgeneration %}",
            "{{- '<|im_end|>\\n' -}}",
            "{%- endif -%}",
            "{%- endfor -%}",
            "{%- if add_generation_prompt -%}",
            "{{- '<|im_start|>assistant\\n' -}}",
            "{%- endif -%}",
        ]
    )
    return "\n".join(lines)
