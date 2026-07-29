from __future__ import annotations

import json
from typing import Any
from urllib.request import Request, urlopen

from .evaluation import parse_competition_output
from .prompts import OFFICIAL_SYSTEM_PROMPT, build_user_prompt


def _get_json(url: str) -> Any:
    with urlopen(url, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError(f"{url} returned HTTP {response.status}")
        return json.loads(response.read().decode("utf-8"))


def smoke_test_vllm(
    base_url: str,
    *,
    prompt: str,
    essay: str,
    seed: int = 42,
) -> dict[str, Any]:
    health_url = base_url.removesuffix("/v1") + "/health"
    with urlopen(health_url, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError(f"Health check returned HTTP {response.status}")
    models = _get_json(base_url.rstrip("/") + "/models")
    model_id = models["data"][0]["id"]
    body = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": OFFICIAL_SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(prompt, essay)},
        ],
        "max_tokens": 512,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": seed,
    }
    request = Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=300) as response:
        payload = json.loads(response.read().decode("utf-8"))
    content = payload["choices"][0]["message"]["content"]
    parsed = parse_competition_output(content)
    return {
        "health": 200,
        "model_id": model_id,
        "valid": parsed.valid,
        "scores": parsed.scores,
        "raw": content,
    }
