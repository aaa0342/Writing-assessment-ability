from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

from .config import RunConfig
from .data import shuffled
from .prompts import build_submission_chat_template
from .schemas import EssayExample, RationaleRecord
from .tokenization import OrdinalDataCollator, TrainingFeature, build_training_features


def enumerate_language_linear_modules(model: Any) -> list[str]:
    import torch

    names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and name.startswith("model.language_model.")
        and not name.endswith("lm_head")
    ]
    if not names:
        raise RuntimeError(
            "No language-model Linear modules found; check the Qwen3.5 architecture"
        )
    forbidden = [name for name in names if "visual" in name or "vision" in name]
    if forbidden:
        raise RuntimeError(f"Vision modules leaked into LoRA targets: {forbidden[:5]}")
    return names


def load_loraplus_model_and_tokenizer(
    config: RunConfig,
    demos: Sequence[dict[str, str]],
) -> tuple[Any, Any, list[str]]:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_id,
        trust_remote_code=False,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.chat_template = build_submission_chat_template(demos)
    model = AutoModelForImageTextToText.from_pretrained(
        config.base_model_id,
        dtype=torch.bfloat16,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    targets = enumerate_language_linear_modules(model)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules=targets,
        use_dora=False,
        use_rslora=False,
    )
    model = get_peft_model(model, peft_config)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.config.use_cache = False
    return model, tokenizer, targets


def digit_token_ids(tokenizer: Any) -> list[int]:
    result: list[int] = []
    for digit in "12345":
        ids = tokenizer.encode(digit, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(
                f"Ordinal loss requires one token per digit; {digit!r} -> {ids}"
            )
        result.append(ids[0])
    if len(set(result)) != 5:
        raise RuntimeError("Digit token ids are not unique")
    return result


def ordinal_score_loss(
    logits: Any,
    score_logit_positions: Any,
    continuous_scores: Any,
    digit_ids: Sequence[int],
) -> tuple[Any, Any]:
    """Continuous-label supervision at the three generated integer score tokens."""

    import torch
    import torch.nn.functional as F

    batch_size = logits.shape[0]
    batch_index = torch.arange(batch_size, device=logits.device)[:, None]
    score_logits = logits[
        batch_index, score_logit_positions.to(logits.device)
    ]
    digit_tensor = torch.tensor(digit_ids, dtype=torch.long, device=logits.device)
    digit_logits = score_logits.index_select(-1, digit_tensor).float()
    log_probs = F.log_softmax(digit_logits, dim=-1)
    values = continuous_scores.to(log_probs.device).clamp(1.0, 5.0)
    lower = torch.floor(values).long()
    upper = torch.ceil(values).long()
    lower_weight = torch.where(
        lower == upper,
        torch.ones_like(values),
        upper.float() - values,
    )
    upper_weight = torch.where(
        lower == upper,
        torch.zeros_like(values),
        values - lower.float(),
    )
    target = torch.zeros_like(log_probs)
    target.scatter_add_(-1, (lower - 1).unsqueeze(-1), lower_weight.unsqueeze(-1))
    target.scatter_add_(-1, (upper - 1).unsqueeze(-1), upper_weight.unsqueeze(-1))
    kl_loss = -(target * log_probs).sum(-1).mean()
    score_values = torch.arange(1, 6, dtype=torch.float32, device=log_probs.device)
    expected_scores = (log_probs.exp() * score_values).sum(-1)
    mse_loss = F.mse_loss(expected_scores, values)
    return kl_loss, mse_loss


def train_loraplus(
    config: RunConfig,
    examples: Sequence[EssayExample],
    rationale_records: Sequence[RationaleRecord],
    demos: Sequence[dict[str, str]],
) -> dict[str, Any]:
    import torch
    from accelerate import Accelerator
    from peft.optimizers import create_loraplus_optimizer
    from torch.utils.data import DataLoader
    from transformers import get_cosine_schedule_with_warmup, set_seed

    config.validate(require_data=False)
    config.ensure_directories()
    set_seed(config.seed)
    model, tokenizer, targets = load_loraplus_model_and_tokenizer(config, demos)
    ordered_examples = shuffled(examples, seed=config.seed)
    features = build_training_features(
        ordered_examples,
        rationale_records,
        tokenizer,
        max_length=config.max_length,
    )
    collator = OrdinalDataCollator(tokenizer.pad_token_id)
    dataloader = DataLoader(
        features,
        batch_size=config.micro_batch_size,
        shuffle=True,
        collate_fn=collator,
        pin_memory=True,
    )
    optimizer = create_loraplus_optimizer(
        model=model,
        optimizer_cls=torch.optim.AdamW,
        lr=config.lora_a_lr,
        loraplus_lr_ratio=config.loraplus_lr_ratio,
        weight_decay=config.weight_decay,
    )
    updates_per_epoch = math.ceil(
        len(dataloader) / config.gradient_accumulation_steps
    )
    total_updates = updates_per_epoch * config.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=math.ceil(total_updates * config.warmup_ratio),
        num_training_steps=total_updates,
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision="bf16",
    )
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    digit_ids = digit_token_ids(tokenizer)
    history: list[dict[str, float | int]] = []
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, config.epochs + 1):
        model.train()
        running = {"loss": 0.0, "ce": 0.0, "kl": 0.0, "mse": 0.0}
        steps = 0
        for batch in dataloader:
            score_positions = batch.pop("score_positions")
            continuous_scores = batch.pop("continuous_scores")
            with accelerator.accumulate(model):
                # The 248k-token vocabulary makes full-sequence logits wasteful.
                # With the tested micro-batch size of one, request logits only at
                # positions that predict assistant target tokens.
                if batch["input_ids"].shape[0] != 1:
                    raise RuntimeError("Memory-efficient ordinal training requires batch size 1")
                label_positions = torch.nonzero(
                    batch["labels"][0] != -100, as_tuple=False
                ).flatten()
                predictor_positions = label_positions - 1
                if torch.any(predictor_positions < 0):
                    raise RuntimeError("Assistant target unexpectedly starts at token zero")
                model_inputs = {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                }
                outputs = model(
                    **model_inputs,
                    use_cache=False,
                    logits_to_keep=predictor_positions,
                )
                target_labels = batch["labels"][0, label_positions]
                ce_loss = torch.nn.functional.cross_entropy(
                    outputs.logits[0].float(), target_labels
                )
                ordinal_predictors = (
                    score_positions[0].to(predictor_positions.device) - 1
                )
                score_logit_positions = torch.searchsorted(
                    predictor_positions, ordinal_predictors
                )
                recovered = predictor_positions[score_logit_positions]
                if not torch.equal(recovered, ordinal_predictors):
                    raise RuntimeError("Could not align ordinal score logits")
                score_logit_positions = score_logit_positions.unsqueeze(0)
                kl_loss, mse_loss = ordinal_score_loss(
                    outputs.logits,
                    score_logit_positions,
                    continuous_scores,
                    digit_ids,
                )
                loss = (
                    ce_loss
                    + config.ordinal_kl_weight * kl_loss
                    + config.ordinal_mse_weight * mse_loss
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), config.max_grad_norm
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            running["loss"] += float(loss.detach())
            running["ce"] += float(ce_loss.detach())
            running["kl"] += float(kl_loss.detach())
            running["mse"] += float(mse_loss.detach())
            steps += 1
        epoch_metrics = {
            "epoch": epoch,
            **{key: value / max(steps, 1) for key, value in running.items()},
        }
        history.append(epoch_metrics)
        accelerator.print(json.dumps(epoch_metrics, ensure_ascii=False))
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            checkpoint_dir = config.output_dir / f"epoch-{epoch}"
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(
                checkpoint_dir,
                safe_serialization=True,
            )
            tokenizer.save_pretrained(checkpoint_dir)
    manifest = {
        "config": config.as_serializable_dict(),
        "examples": len(features),
        "target_modules": targets,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "history": history,
    }
    if accelerator.is_main_process:
        manifest_path = config.output_dir / "training_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return manifest
