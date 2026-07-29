from __future__ import annotations

import json
import unittest
from pathlib import Path

from essay_scorer.data import load_examples, round_half_up
from essay_scorer.colab import choose_vllm_context_length
from essay_scorer.evaluation import (
    ParsedPrediction,
    compare_prediction_sets,
    integer_oracle_metrics,
    parse_competition_output,
)
from essay_scorer.generation import LocalOpenAIChatGenerator, parse_exact_json_object
from essay_scorer.prompts import (
    OFFICIAL_SYSTEM_PROMPT,
    build_evidence_prompt,
    build_submission_chat_template,
    build_target_json,
    build_user_prompt,
)
from essay_scorer.rationales import validate_critic
from essay_scorer.schemas import DIMENSIONS, JUDGE_ITEMS
from essay_scorer.training import ordinal_score_loss


class CoreTests(unittest.TestCase):
    def test_colab_context_length_by_vram(self) -> None:
        self.assertEqual(choose_vllm_context_length(80), 32768)
        self.assertEqual(choose_vllm_context_length(40), 16384)
        self.assertEqual(choose_vllm_context_length(24), 8192)
        with self.assertRaises(RuntimeError):
            choose_vllm_context_length(16)

    def test_round_half_up(self) -> None:
        self.assertEqual(round_half_up(1.0), 1)
        self.assertEqual(round_half_up(2.49), 2)
        self.assertEqual(round_half_up(2.5), 3)
        self.assertEqual(round_half_up(5.0), 5)

    def test_official_prompt_and_user_shape(self) -> None:
        self.assertIn("모든 점수는 1~5의 정수", OFFICIAL_SYSTEM_PROMPT)
        user = build_user_prompt("주제", "본문")
        self.assertEqual(user, "[prompt_text]\n주제\n\n[essay_text]\n본문")

    def test_target_and_strict_parser(self) -> None:
        scores = {dimension: 3 for dimension in DIMENSIONS}
        rationales = {dimension: f"{dimension}의 구체적인 판단 근거" for dimension in DIMENSIONS}
        text = build_target_json(scores, rationales)
        prediction = parse_competition_output(text)
        self.assertTrue(prediction.valid)
        self.assertEqual(prediction.scores, scores)
        with self.assertRaises(ValueError):
            parse_competition_output("설명\n" + text)
        decimal = json.loads(text)
        decimal["content"]["score"] = 3.2
        with self.assertRaises(ValueError):
            parse_competition_output(json.dumps(decimal, ensure_ascii=False))

    def test_exact_json_rejects_markdown(self) -> None:
        self.assertEqual(parse_exact_json_object('{"a":1}'), {"a": 1})
        with self.assertRaises(ValueError):
            parse_exact_json_object('```json\n{"a":1}\n```')

    def test_local_endpoint_guard(self) -> None:
        LocalOpenAIChatGenerator("http://127.0.0.1:8000/v1", "model")
        with self.assertRaises(ValueError):
            LocalOpenAIChatGenerator("https://external.example/v1", "model")

    def test_chat_template_contains_demos_and_generation_mask(self) -> None:
        demos = [
            {"user": f"user-{index}", "assistant": f"assistant-{index}"}
            for index in range(3)
        ]
        template = build_submission_chat_template(demos)
        for index in range(3):
            self.assertIn(f"user-{index}", template)
            self.assertIn(f"assistant-{index}", template)
        self.assertIn("{% generation %}", template)
        self.assertNotIn("<think>", template)

    def test_retry_feedback_changes_evidence_prompt(self) -> None:
        first = build_evidence_prompt(
            "주제", "검증 가능한 본문 문장입니다.", {dimension: 3 for dimension in DIMENSIONS}
        )
        second = build_evidence_prompt(
            "주제",
            "검증 가능한 본문 문장입니다.",
            {dimension: 3 for dimension in DIMENSIONS},
            previous_feedback="인용문이 본문에 없음",
        )
        self.assertNotEqual(first, second)
        self.assertIn("인용문이 본문에 없음", second)

    def test_prediction_set_comparison(self) -> None:
        prediction = ParsedPrediction(
            scores={dimension: 3 for dimension in DIMENSIONS},
            rationales={dimension: "근거" for dimension in DIMENSIONS},
            raw='{"ok":true}',
            attempts=1,
            valid=True,
        )
        report = compare_prediction_sets([prediction], [prediction])
        self.assertEqual(report["score_match_rate"], 1.0)
        self.assertEqual(report["exact_text_match_rate"], 1.0)

    def test_ordinal_score_loss_is_finite(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch not installed")
        logits = torch.zeros((1, 3, 8), dtype=torch.float32)
        positions = torch.tensor([[0, 1, 2]], dtype=torch.long)
        scores = torch.tensor([[1.5, 3.0, 4.75]], dtype=torch.float32)
        kl_loss, mse_loss = ordinal_score_loss(
            logits, positions, scores, [0, 1, 2, 3, 4]
        )
        self.assertTrue(torch.isfinite(kl_loss))
        self.assertTrue(torch.isfinite(mse_loss))

    def test_critic_validation(self) -> None:
        critic = {
            dimension: {
                item: {"evidence": "구체적인 검토", "score": 4}
                for item in JUDGE_ITEMS
            }
            for dimension in DIMENSIONS
        }
        _, mean_score, passed = validate_critic(
            critic, minimum_item_score=4, minimum_mean_score=4.0
        )
        self.assertEqual(mean_score, 4.0)
        self.assertTrue(passed)

    def test_real_data_and_integer_oracle(self) -> None:
        paths = sorted(Path(".").glob("*validation.jsonl"))
        if not paths:
            self.skipTest("Competition validation data not present")
        examples = load_examples(paths[0])
        self.assertEqual(len(examples), 400)
        metrics = integer_oracle_metrics(examples)
        self.assertAlmostEqual(
            metrics["dimensions"]["content"]["rmse"], 0.296395, places=5
        )
        self.assertEqual(metrics["parse_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
