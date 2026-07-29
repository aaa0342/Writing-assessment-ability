from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def merge_adapter(
    base_model_id: str,
    adapter_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        AutoTokenizer,
        GenerationConfig,
    )

    adapter_path = Path(adapter_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path, trust_remote_code=False, use_fast=True
    )
    base = AutoModelForImageTextToText.from_pretrained(
        base_model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    merged = PeftModel.from_pretrained(base, adapter_path).merge_and_unload()
    merged.config.use_cache = True
    merged.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size="4GB",
    )
    tokenizer.save_pretrained(output_dir)
    try:
        processor = AutoProcessor.from_pretrained(
            base_model_id, trust_remote_code=False
        )
        if hasattr(processor, "tokenizer"):
            processor.tokenizer.chat_template = tokenizer.chat_template
        processor.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    except Exception as exc:
        print(f"Processor save warning: {type(exc).__name__}: {exc}")
    GenerationConfig(
        do_sample=False,
        max_new_tokens=512,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    ).save_pretrained(output_dir)
    forbidden = [
        path.name
        for path in output_dir.iterdir()
        if path.name in {"adapter_config.json", "adapter_model.safetensors"}
        or path.name.startswith("modeling_")
    ]
    if forbidden:
        raise RuntimeError(f"Forbidden files in merged repository: {forbidden}")
    config_value = json.loads(
        (output_dir / "config.json").read_text(encoding="utf-8")
    )
    architectures = config_value.get("architectures", [])
    if "Qwen3_5ForConditionalGeneration" not in architectures:
        raise RuntimeError(f"Unexpected merged architecture: {architectures}")
    manifest = {
        "base_model_id": base_model_id,
        "adapter_path": str(adapter_path),
        "output_dir": str(output_dir),
        "architectures": architectures,
        "files": sorted(path.name for path in output_dir.iterdir()),
    }
    (output_dir / "export_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def write_model_card(output_dir: str | Path, *, base_model_id: str) -> None:
    output_dir = Path(output_dir)
    card = f"""---
license: apache-2.0
base_model: {base_model_id}
pipeline_tag: text-generation
library_name: transformers
tags:
  - qwen3.5
  - korean
  - essay-scoring
  - lora-plus
---

# Korean argumentative essay scorer

Qwen3.5-9B model fine-tuned with LoRA+ for the 2026 Korean essay-scoring
competition. The repository contains merged BF16 weights and a standard chat
template that disables thinking output and inserts three fixed format examples.

The model returns one JSON object containing integer scores from 1 to 5 and
Korean rationales for `content`, `organization`, and `expression`.
"""
    (output_dir / "README.md").write_text(card, encoding="utf-8")


def upload_public_model(
    output_dir: str | Path,
    repo_id: str,
    *,
    token_env: str = "HF_TOKEN",
) -> str:
    from huggingface_hub import HfApi

    token = os.environ.get(token_env)
    if not token:
        raise RuntimeError(f"Set {token_env}; never place the token in source code")
    if repo_id.startswith("http://") or repo_id.startswith("https://"):
        raise ValueError("repo_id must be in <org>/<repo> form")
    if repo_id.count("/") != 1:
        raise ValueError("repo_id must contain exactly one slash")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(output_dir),
        commit_message="Upload merged Qwen3.5-9B LoRA+ essay scorer",
    )
    info = api.model_info(repo_id=repo_id, token=False)
    if getattr(info, "gated", False) or getattr(info, "private", False):
        raise RuntimeError("Uploaded repository is not public and ungated")
    return f"https://huggingface.co/{repo_id}"


def vllm_docker_command(repo_id: str) -> str:
    return (
        "docker run --rm --gpus all -p 8000:8000 "
        "vllm/vllm-openai:latest "
        f"{repo_id} --host 0.0.0.0 --port 8000 "
        "--tensor-parallel-size 1 --pipeline-parallel-size 1 "
        "--dtype bfloat16 --max-model-len 32768"
    )
