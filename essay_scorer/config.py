from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class RunConfig:
    """Single source of truth for training and export settings."""

    base_model_id: str = "Qwen/Qwen3.5-9B"
    train_path: Path = Path("글쓰기채점능력평가2026_train.jsonl")
    validation_path: Path = Path("글쓰기채점능력평가2026_validation.jsonl")
    artifacts_dir: Path = Path("artifacts")
    output_dir: Path = Path("artifacts/checkpoints")
    merged_dir: Path = Path("artifacts/merged_model")
    seed: int = 42
    max_length: int = 8192
    max_new_tokens: int = 512
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    epochs: int = 3
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_a_lr: float = 5e-5
    loraplus_lr_ratio: float = 16.0
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    ordinal_kl_weight: float = 0.5
    ordinal_mse_weight: float = 0.5
    rationale_max_attempts: int = 3
    rationale_min_item_score: int = 4
    rationale_min_mean_score: float = 4.25
    min_rationale_keep_rate: float = 0.90

    def validate(self, require_data: bool = True) -> None:
        if require_data:
            for path in (self.train_path, self.validation_path):
                if not path.is_file():
                    raise FileNotFoundError(path)
        if self.max_length < 2048:
            raise ValueError("max_length must be at least 2048")
        if self.micro_batch_size != 1:
            raise ValueError("The tested 48 GB configuration uses micro_batch_size=1")
        if self.loraplus_lr_ratio <= 1:
            raise ValueError("The final model must use LoRA+, so loraplus_lr_ratio must be > 1")
        if not 0 < self.min_rationale_keep_rate <= 1:
            raise ValueError("min_rationale_keep_rate must be in (0, 1]")

    def ensure_directories(self) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def as_serializable_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key, value in tuple(result.items()):
            if isinstance(value, Path):
                result[key] = str(value)
        return result
