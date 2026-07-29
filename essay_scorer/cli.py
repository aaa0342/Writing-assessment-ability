from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

from .audit import audit_rationales
from .config import RunConfig
from .data import (
    load_examples,
    load_rationale_records,
    validate_rationale_coverage,
)
from .evaluation import (
    choose_demo_order,
    compare_prediction_sets,
    evaluate_generator,
    integer_oracle_metrics,
    load_adapter_generator,
)
from .export import merge_adapter, upload_public_model, vllm_docker_command, write_model_card
from .fewshot import (
    all_demo_orders,
    load_demos,
    save_demos,
    select_balanced_demos,
)
from .generation import LocalOpenAIChatGenerator, TransformersChatGenerator
from .rationales import RationaleBuilder, generate_rationale_file
from .smoke import smoke_test_vllm
from .training import train_loraplus


def _generator_from_args(args: argparse.Namespace):
    if args.backend == "local-vllm":
        return LocalOpenAIChatGenerator(
            base_url=args.base_url,
            model_id=args.model,
        )
    return TransformersChatGenerator.from_pretrained(args.model)


def command_analyze(args: argparse.Namespace) -> None:
    train = load_examples(args.train)
    validation = load_examples(args.validation)
    report = {
        "train": len(train),
        "validation": len(validation),
        "integer_oracle_validation": integer_oracle_metrics(validation),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_rationales(args: argparse.Namespace) -> None:
    examples = load_examples(args.data)
    generator = _generator_from_args(args)
    builder = RationaleBuilder(generator, max_attempts=args.max_attempts)
    report = generate_rationale_file(
        examples,
        builder,
        args.output,
        resume=not args.no_resume,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_select_demos(args: argparse.Namespace) -> None:
    examples = load_examples(args.train)
    records = load_rationale_records(args.rationales)
    validate_rationale_coverage(examples, records)
    demos = select_balanced_demos(examples, records)
    save_demos(args.output, demos)
    print(json.dumps(demos, ensure_ascii=False, indent=2))


def command_train(args: argparse.Namespace) -> None:
    config = RunConfig(
        base_model_id=args.model,
        output_dir=Path(args.output),
        epochs=args.epochs,
        loraplus_lr_ratio=args.ratio,
    )
    examples = load_examples(args.train)
    records = load_rationale_records(args.rationales)
    validate_rationale_coverage(examples, records, config.min_rationale_keep_rate)
    if args.extra_data or args.extra_rationales:
        if not (args.extra_data and args.extra_rationales):
            raise ValueError("--extra-data and --extra-rationales must be used together")
        extra_examples = load_examples(args.extra_data)
        extra_records = load_rationale_records(args.extra_rationales)
        validate_rationale_coverage(
            extra_examples, extra_records, config.min_rationale_keep_rate
        )
        examples = [*examples, *extra_examples]
        records = [*records, *extra_records]
    demos = load_demos(args.demos)
    manifest = train_loraplus(config, examples, records, demos)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def command_evaluate(args: argparse.Namespace) -> None:
    examples = load_examples(args.validation)
    generator = load_adapter_generator(args.model, args.adapter)
    _, metrics = evaluate_generator(
        generator,
        examples,
        output_path=args.output,
        retries=2,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def command_merge(args: argparse.Namespace) -> None:
    manifest = merge_adapter(args.model, args.adapter, args.output)
    write_model_card(args.output, base_model_id=args.model)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def command_push(args: argparse.Namespace) -> None:
    url = upload_public_model(args.model_dir, args.repo_id)
    print(url)
    print(vllm_docker_command(args.repo_id))


def command_select_demo_order(args: argparse.Namespace) -> None:
    examples = load_examples(args.validation)
    demos = load_demos(args.demos)
    generator = TransformersChatGenerator.from_pretrained(args.model)
    chosen, reports = choose_demo_order(
        generator,
        examples,
        all_demo_orders(demos),
        limit=args.limit,
    )
    save_demos(args.output, chosen)
    report_path = Path(args.output).with_suffix(".orders.json")
    report_path.write_text(
        json.dumps(reports, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"chosen": chosen, "reports": reports}, ensure_ascii=False, indent=2))


def command_smoke(args: argparse.Namespace) -> None:
    result = smoke_test_vllm(
        args.base_url,
        prompt=args.prompt,
        essay=args.essay,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def command_audit(args: argparse.Namespace) -> None:
    examples = load_examples(args.validation)
    generator = _generator_from_args(args)
    report = audit_rationales(
        generator,
        examples,
        args.predictions,
        args.output,
        limit=args.limit,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_verify_merge(args: argparse.Namespace) -> None:
    import torch

    examples = load_examples(args.validation)[: args.limit]
    adapter_generator = load_adapter_generator(args.model, args.adapter)
    adapter_predictions, _ = evaluate_generator(
        adapter_generator, examples, retries=0
    )
    del adapter_generator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    merged_generator = TransformersChatGenerator.from_pretrained(args.merged)
    merged_predictions, _ = evaluate_generator(
        merged_generator, examples, retries=0
    )
    report = compare_prediction_sets(adapter_predictions, merged_predictions)
    if report["score_match_rate"] != 1.0:
        raise RuntimeError(f"Merged model score mismatch: {report}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--train", default="글쓰기채점능력평가2026_train.jsonl")
    analyze.add_argument(
        "--validation", default="글쓰기채점능력평가2026_validation.jsonl"
    )
    analyze.set_defaults(func=command_analyze)

    rationales = subparsers.add_parser("generate-rationales")
    rationales.add_argument("--data", required=True)
    rationales.add_argument("--output", required=True)
    rationales.add_argument("--model", default="Qwen/Qwen3.5-9B")
    rationales.add_argument(
        "--backend", choices=("transformers", "local-vllm"), default="local-vllm"
    )
    rationales.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    rationales.add_argument("--max-attempts", type=int, default=3)
    rationales.add_argument("--no-resume", action="store_true")
    rationales.set_defaults(func=command_rationales)

    demos = subparsers.add_parser("select-demos")
    demos.add_argument("--train", required=True)
    demos.add_argument("--rationales", required=True)
    demos.add_argument("--output", default="artifacts/demos.json")
    demos.set_defaults(func=command_select_demos)

    train = subparsers.add_parser("train")
    train.add_argument("--train", required=True)
    train.add_argument("--rationales", required=True)
    train.add_argument("--extra-data")
    train.add_argument("--extra-rationales")
    train.add_argument("--demos", required=True)
    train.add_argument("--model", default="Qwen/Qwen3.5-9B")
    train.add_argument("--output", required=True)
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--ratio", type=float, choices=(8.0, 16.0), default=16.0)
    train.set_defaults(func=command_train)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--validation", required=True)
    evaluate.add_argument("--adapter", required=True)
    evaluate.add_argument("--model", default="Qwen/Qwen3.5-9B")
    evaluate.add_argument("--output", required=True)
    evaluate.set_defaults(func=command_evaluate)

    merge = subparsers.add_parser("merge")
    merge.add_argument("--adapter", required=True)
    merge.add_argument("--model", default="Qwen/Qwen3.5-9B")
    merge.add_argument("--output", required=True)
    merge.set_defaults(func=command_merge)

    push = subparsers.add_parser("push")
    push.add_argument("--model-dir", required=True)
    push.add_argument("--repo-id", required=True)
    push.set_defaults(func=command_push)

    order = subparsers.add_parser("select-demo-order")
    order.add_argument("--validation", required=True)
    order.add_argument("--demos", required=True)
    order.add_argument("--output", default="artifacts/demos_ordered.json")
    order.add_argument("--model", default="Qwen/Qwen3.5-9B")
    order.add_argument("--limit", type=int)
    order.set_defaults(func=command_select_demo_order)

    smoke = subparsers.add_parser("smoke-vllm")
    smoke.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    smoke.add_argument("--prompt", required=True)
    smoke.add_argument("--essay", required=True)
    smoke.set_defaults(func=command_smoke)

    audit = subparsers.add_parser("audit-rationales")
    audit.add_argument("--validation", required=True)
    audit.add_argument("--predictions", required=True)
    audit.add_argument("--output", required=True)
    audit.add_argument("--model", default="Qwen/Qwen3.5-9B")
    audit.add_argument(
        "--backend", choices=("transformers", "local-vllm"), default="local-vllm"
    )
    audit.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    audit.add_argument("--limit", type=int)
    audit.set_defaults(func=command_audit)

    verify = subparsers.add_parser("verify-merge")
    verify.add_argument("--validation", required=True)
    verify.add_argument("--adapter", required=True)
    verify.add_argument("--merged", required=True)
    verify.add_argument("--model", default="Qwen/Qwen3.5-9B")
    verify.add_argument("--limit", type=int, default=30)
    verify.set_defaults(func=command_verify_merge)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
