from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from .data import load_examples, load_rationale_records


@dataclass(slots=True)
class ColabRunConfig:
    train_path: Path
    validation_path: Path
    artifacts_dir: Path
    model_id: str = "Qwen/Qwen3.5-9B"
    ratio: float = 16.0
    epochs: int = 3
    server_timeout_seconds: int = 3600
    verify_limit: int = 10


def gpu_memory_gib() -> float:
    command = [
        "nvidia-smi",
        "--query-gpu=memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "NVIDIA GPU를 찾지 못했습니다. Colab에서 런타임 > 런타임 유형 변경 > "
            "GPU를 선택한 뒤 다시 실행하세요."
        ) from exc
    first_device = result.stdout.strip().splitlines()[0]
    return float(first_device) / 1024.0


def choose_vllm_context_length(memory_gib: float) -> int:
    if memory_gib >= 70:
        return 32768
    if memory_gib >= 38:
        return 16384
    if memory_gib >= 22:
        return 8192
    raise RuntimeError(
        f"GPU VRAM {memory_gib:.1f} GiB로는 Qwen3.5-9B BF16 파이프라인을 "
        "실행할 수 없습니다. A100 40GB 이상을 선택하세요."
    )


def _print_command(command: list[str]) -> None:
    print("\n$", " ".join(shlex.quote(part) for part in command), flush=True)


def _run_module(arguments: list[str]) -> None:
    command = [sys.executable, "-m", "essay_scorer", *arguments]
    _print_command(command)
    subprocess.run(command, check=True)


def _health_ready() -> bool:
    try:
        with urlopen("http://127.0.0.1:8000/health", timeout=2) as response:
            return response.status == 200
    except (OSError, URLError):
        return False


def _tail(path: Path, lines: int = 40) -> str:
    if not path.is_file():
        return "(로그 파일 없음)"
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    )


def _start_vllm(
    model_id: str,
    max_model_len: int,
    log_path: Path,
    timeout_seconds: int,
) -> tuple[subprocess.Popen[str], object]:
    executable = shutil.which("vllm")
    if not executable:
        raise RuntimeError(
            "vLLM이 설치되지 않았습니다. colab.ipynb의 설치 셀부터 모두 실행하세요."
        )
    if _health_ready():
        raise RuntimeError(
            "8000번 포트에 기존 vLLM 서버가 실행 중입니다. Colab 런타임을 "
            "재시작한 뒤 '모두 실행'을 누르세요."
        )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")
    command = [
        executable,
        "serve",
        model_id,
        "--language-model-only",
        "--max-model-len",
        str(max_model_len),
        "--reasoning-parser",
        "qwen3",
        "--gpu-memory-utilization",
        "0.90",
    ]
    _print_command(command)
    process = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    started_at = time.monotonic()
    next_update = 0
    while time.monotonic() - started_at < timeout_seconds:
        return_code = process.poll()
        if return_code is not None:
            log_handle.flush()
            log_handle.close()
            raise RuntimeError(
                f"vLLM이 종료 코드 {return_code}로 중단됐습니다.\n\n"
                f"{_tail(log_path)}"
            )
        if _health_ready():
            print("vLLM 서버 준비 완료", flush=True)
            return process, log_handle
        elapsed = int(time.monotonic() - started_at)
        if elapsed >= next_update:
            print(
                f"vLLM 모델 다운로드/로딩 중... {elapsed}초 "
                f"(로그: {log_path})",
                flush=True,
            )
            next_update = elapsed + 30
        time.sleep(5)
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
    log_handle.close()
    raise TimeoutError(
        f"{timeout_seconds}초 안에 vLLM이 준비되지 않았습니다.\n\n{_tail(log_path)}"
    )


def _stop_vllm(process: subprocess.Popen[str] | None, log_handle: object | None) -> None:
    had_server = process is not None or log_handle is not None
    if process is not None and process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=30)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
    if log_handle is not None:
        log_handle.close()  # type: ignore[attr-defined]
    if had_server:
        print("vLLM 서버 종료 및 GPU 메모리 반환 완료", flush=True)


def _rationale_keep_rate(data_path: Path, rationale_path: Path) -> float:
    if not rationale_path.is_file():
        return 0.0
    expected = {example.id for example in load_examples(data_path)}
    passed = {
        record.example_id
        for record in load_rationale_records(rationale_path)
        if record.passed and record.example_id in expected
    }
    return len(passed) / len(expected)


def _ensure_rationales(config: ColabRunConfig, max_model_len: int) -> None:
    train_output = config.artifacts_dir / "train_rationales.jsonl"
    validation_output = config.artifacts_dir / "validation_rationales.jsonl"
    pairs = (
        (config.train_path, train_output),
        (config.validation_path, validation_output),
    )
    if all(_rationale_keep_rate(data, output) >= 0.90 for data, output in pairs):
        print("검증된 rationale가 이미 있어 생성 단계를 건너뜁니다.", flush=True)
        return

    process: subprocess.Popen[str] | None = None
    log_handle: object | None = None
    try:
        process, log_handle = _start_vllm(
            config.model_id,
            max_model_len,
            config.artifacts_dir / "vllm_generation.log",
            config.server_timeout_seconds,
        )
        for round_index in range(1, 3):
            for data_path, output_path in pairs:
                keep_rate = _rationale_keep_rate(data_path, output_path)
                if keep_rate >= 0.90:
                    print(
                        f"{output_path.name}: keep_rate={keep_rate:.1%}, 건너뜀",
                        flush=True,
                    )
                    continue
                print(
                    f"{output_path.name}: 생성 라운드 {round_index}, "
                    f"현재 keep_rate={keep_rate:.1%}",
                    flush=True,
                )
                _run_module(
                    [
                        "generate-rationales",
                        "--data",
                        str(data_path),
                        "--output",
                        str(output_path),
                        "--model",
                        config.model_id,
                    ]
                )
            if all(
                _rationale_keep_rate(data, output) >= 0.90
                for data, output in pairs
            ):
                break
        failures = [
            f"{output.name}={_rationale_keep_rate(data, output):.1%}"
            for data, output in pairs
            if _rationale_keep_rate(data, output) < 0.90
        ]
        if failures:
            raise RuntimeError(
                "rationale keep_rate가 90%에 미달했습니다: " + ", ".join(failures)
            )
    finally:
        _stop_vllm(process, log_handle)


def _load_metrics(prediction_path: Path) -> dict[str, object]:
    metrics_path = prediction_path.with_suffix(".metrics.json")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _select_best_epoch(config: ColabRunConfig) -> int:
    reports: list[tuple[int, dict[str, object]]] = []
    for epoch in range(1, config.epochs + 1):
        prediction_path = (
            config.artifacts_dir / f"main_epoch_{epoch}_predictions.jsonl"
        )
        metrics_path = prediction_path.with_suffix(".metrics.json")
        if not metrics_path.is_file():
            _run_module(
                [
                    "evaluate",
                    "--validation",
                    str(config.validation_path),
                    "--adapter",
                    str(config.artifacts_dir / "main" / f"epoch-{epoch}"),
                    "--model",
                    config.model_id,
                    "--output",
                    str(prediction_path),
                ]
            )
        reports.append((epoch, _load_metrics(prediction_path)))

    def objective(item: tuple[int, dict[str, object]]) -> tuple[float, float, float]:
        _, metrics = item
        return (
            -float(metrics["first_attempt_rate"]),
            float(metrics["macro_rmse"]),
            -float(metrics["macro_spearman"]),
        )

    best_epoch, _ = min(reports, key=objective)
    selection = {
        "best_epoch": best_epoch,
        "ratio": config.ratio,
        "reports": [
            {
                "epoch": epoch,
                "first_attempt_rate": metrics["first_attempt_rate"],
                "parse_rate": metrics["parse_rate"],
                "macro_rmse": metrics["macro_rmse"],
                "macro_spearman": metrics["macro_spearman"],
            }
            for epoch, metrics in reports
        ],
    }
    (config.artifacts_dir / "model_selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(selection, ensure_ascii=False, indent=2), flush=True)
    return best_epoch


def run_colab_pipeline(config: ColabRunConfig) -> dict[str, object]:
    if not config.train_path.is_file() or not config.validation_path.is_file():
        raise FileNotFoundError(
            "학습 데이터가 없습니다. GitHub 저장소 루트에서 실행하세요."
        )
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    memory_gib = gpu_memory_gib()
    if memory_gib < 38:
        raise RuntimeError(
            f"현재 GPU는 {memory_gib:.1f} GiB입니다. 이 원클릭 노트북은 현재 "
            "BF16 LoRA+ 구현을 보존하므로 A100 40GB 이상이 필요합니다."
        )
    max_model_len = choose_vllm_context_length(memory_gib)
    try:
        import torchaudio  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        pass
    except RuntimeError as exc:
        if "compiled with different CUDA versions" in str(exc):
            raise RuntimeError(
                "Colab 기본 torchaudio와 PyTorch CUDA 빌드가 충돌합니다. "
                "`uv pip uninstall --system torchaudio`를 실행한 뒤 다시 "
                "시도하세요."
            ) from exc
        raise
    print(
        f"GPU VRAM={memory_gib:.1f} GiB, vLLM max_model_len={max_model_len}",
        flush=True,
    )
    _run_module(
        [
            "analyze",
            "--train",
            str(config.train_path),
            "--validation",
            str(config.validation_path),
        ]
    )
    _ensure_rationales(config, max_model_len)

    demos_path = config.artifacts_dir / "demos.json"
    if not demos_path.is_file():
        _run_module(
            [
                "select-demos",
                "--train",
                str(config.train_path),
                "--rationales",
                str(config.artifacts_dir / "train_rationales.jsonl"),
                "--output",
                str(demos_path),
            ]
        )

    main_output = config.artifacts_dir / "main"
    if not (main_output / f"epoch-{config.epochs}").is_dir():
        _run_module(
            [
                "train",
                "--train",
                str(config.train_path),
                "--rationales",
                str(config.artifacts_dir / "train_rationales.jsonl"),
                "--demos",
                str(demos_path),
                "--model",
                config.model_id,
                "--ratio",
                str(config.ratio),
                "--epochs",
                str(config.epochs),
                "--output",
                str(main_output),
            ]
        )

    best_epoch = _select_best_epoch(config)
    final_output = config.artifacts_dir / "final"
    final_adapter = final_output / f"epoch-{best_epoch}"
    if not final_adapter.is_dir():
        _run_module(
            [
                "train",
                "--train",
                str(config.train_path),
                "--rationales",
                str(config.artifacts_dir / "train_rationales.jsonl"),
                "--extra-data",
                str(config.validation_path),
                "--extra-rationales",
                str(config.artifacts_dir / "validation_rationales.jsonl"),
                "--demos",
                str(demos_path),
                "--model",
                config.model_id,
                "--ratio",
                str(config.ratio),
                "--epochs",
                str(best_epoch),
                "--output",
                str(final_output),
            ]
        )

    merged_output = config.artifacts_dir / "merged_model"
    if not (merged_output / "config.json").is_file():
        _run_module(
            [
                "merge",
                "--adapter",
                str(final_adapter),
                "--model",
                config.model_id,
                "--output",
                str(merged_output),
            ]
        )
    verification_path = config.artifacts_dir / "merge_verified.json"
    if not verification_path.is_file():
        _run_module(
            [
                "verify-merge",
                "--validation",
                str(config.validation_path),
                "--adapter",
                str(final_adapter),
                "--merged",
                str(merged_output),
                "--model",
                config.model_id,
                "--limit",
                str(config.verify_limit),
            ]
        )
        verification_path.write_text(
            json.dumps({"verified": True, "limit": config.verify_limit}, indent=2),
            encoding="utf-8",
        )

    repo_id = os.environ.get("HF_REPO_ID", "").strip()
    if repo_id and os.environ.get("HF_TOKEN"):
        _run_module(
            [
                "push",
                "--model-dir",
                str(merged_output),
                "--repo-id",
                repo_id,
            ]
        )
    else:
        print(
            "HF_TOKEN/HF_REPO_ID가 없어 업로드만 건너뜁니다. 병합 모델은 "
            f"{merged_output}에 있습니다.",
            flush=True,
        )
    return {
        "gpu_memory_gib": memory_gib,
        "vllm_max_model_len": max_model_len,
        "best_epoch": best_epoch,
        "ratio": config.ratio,
        "merged_model": str(merged_output),
        "hf_repo_id": repo_id or None,
    }
