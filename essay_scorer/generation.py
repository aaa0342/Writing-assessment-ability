from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence
from urllib.parse import urlparse


class ChatGenerator(Protocol):
    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_new_tokens: int = 512,
    ) -> str: ...


def parse_exact_json_object(text: str) -> dict[str, Any]:
    """Parse one JSON object and reject markdown or surrounding commentary."""

    stripped = text.strip()
    if not stripped.startswith("{") or not stripped.endswith("}"):
        raise ValueError("Output is not a bare JSON object")
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise ValueError("Output JSON must be an object")
    return value


@dataclass(slots=True)
class TransformersChatGenerator:
    model: Any
    tokenizer: Any

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str,
        *,
        dtype: str = "bfloat16",
        device_map: str = "auto",
    ) -> "TransformersChatGenerator":
        import torch
        from transformers import AutoModelForImageTextToText, AutoTokenizer

        torch_dtype = getattr(torch, dtype)
        tokenizer = AutoTokenizer.from_pretrained(
            model_id_or_path,
            trust_remote_code=False,
            use_fast=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForImageTextToText.from_pretrained(
            model_id_or_path,
            dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=False,
            low_cpu_mem_usage=True,
        )
        model.eval()
        return cls(model=model, tokenizer=tokenizer)

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_new_tokens: int = 512,
    ) -> str:
        import torch

        rendered = self.tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        completion = generated[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(completion, skip_special_tokens=True).strip()


@dataclass(slots=True)
class LocalOpenAIChatGenerator:
    """Client for a local vLLM server.

    External hosts are rejected deliberately because the competition prohibits
    external model APIs.
    """

    base_url: str
    model_id: str
    api_key: str = "EMPTY"

    def __post_init__(self) -> None:
        host = (urlparse(self.base_url).hostname or "").lower()
        if host not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Only a local OpenAI-compatible endpoint is allowed")

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_new_tokens: int = 512,
    ) -> str:
        from openai import OpenAI

        client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model_id,
            messages=[dict(message) for message in messages],
            max_tokens=max_new_tokens,
            temperature=0.0,
            seed=42,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Local model returned an empty completion")
        return content.strip()
