# Qwen3.5-9B LoRA+ 글쓰기 채점 모델

2026 글쓰기 채점 능력 평가의 과제기술서와 Hugging Face URL 제출 규정을
그대로 반영한 재현 가능한 학습·제출 파이프라인입니다.

핵심 설계는 다음과 같습니다.

- 운영 측의 공식 system prompt 및 `[prompt_text]`/`[essay_text]` 입력 사용
- `content`, `organization`, `expression`의 1~5 정수 JSON 출력
- 원본 소수 라벨을 보존하는 score-token ordinal KL 및 기댓값 MSE 보조 손실
- Qwen3.5-9B 언어 계층 전체에 LoRA+ 적용, 비전 인코더 완전 동결
- 근거 추출 → rationale 생성 → 12항목 LLM Judge 검증
- 검증된 저·중·고 3-shot을 표준 Jinja chat template에 자동 삽입
- `<think>` 및 Markdown 출력을 차단하고 adapter를 BF16 전체 모델로 병합
- 공개·ungated Hugging Face 저장소와 표준 vLLM Docker 명령 검증

## 환경

- Linux, Python 3.11
- NVIDIA L40S/A100 48GB 이상 1장
- CUDA가 구성된 PyTorch 환경

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

Qwen3.5를 지원하는 stable vLLM이 없다면 별도 추론 환경에서 공식 nightly
wheel을 사용합니다.

```bash
uv pip install vllm --torch-backend=auto \
  --extra-index-url https://wheels.vllm.ai/nightly
```

## 1. 데이터 및 정수 출력 하한 확인

```bash
python -m essay_scorer analyze
```

## 2. rationale 자기증류

로컬 vLLM 서버를 실행합니다. 이 단계의 `--language-model-only`는 데이터
생성 효율을 위한 것이며 최종 제출 호환성 시험에서는 사용하지 않습니다.

```bash
vllm serve Qwen/Qwen3.5-9B \
  --language-model-only \
  --max-model-len 32768 \
  --reasoning-parser qwen3
```

다른 셸에서 train과 validation의 rationale을 생성합니다.

```bash
python -m essay_scorer generate-rationales \
  --data 글쓰기채점능력평가2026_train.jsonl \
  --output artifacts/train_rationales.jsonl

python -m essay_scorer generate-rationales \
  --data 글쓰기채점능력평가2026_validation.jsonl \
  --output artifacts/validation_rationales.jsonl
```

외부 API 주소는 코드에서 거부합니다. `local-vllm` backend는 localhost만
허용합니다.

## 3. 3-shot 선정

```bash
python -m essay_scorer select-demos \
  --train 글쓰기채점능력평가2026_train.jsonl \
  --rationales artifacts/train_rationales.jsonl \
  --output artifacts/demos.json
```

필요하면 6개 순서를 validation에서 비교합니다.

```bash
python -m essay_scorer select-demo-order \
  --validation 글쓰기채점능력평가2026_validation.jsonl \
  --demos artifacts/demos.json \
  --output artifacts/demos_ordered.json
```

## 4. LoRA+ 파일럿 및 본 학습

ratio 8과 16을 각각 1 epoch 실행하고 validation 점수를 비교합니다.

```bash
accelerate launch -m essay_scorer train \
  --train 글쓰기채점능력평가2026_train.jsonl \
  --rationales artifacts/train_rationales.jsonl \
  --demos artifacts/demos_ordered.json \
  --ratio 8 --epochs 1 --output artifacts/pilot-ratio-8

accelerate launch -m essay_scorer train \
  --train 글쓰기채점능력평가2026_train.jsonl \
  --rationales artifacts/train_rationales.jsonl \
  --demos artifacts/demos_ordered.json \
  --ratio 16 --epochs 1 --output artifacts/pilot-ratio-16
```

선택한 ratio로 3 epoch 학습합니다.

```bash
accelerate launch -m essay_scorer train \
  --train 글쓰기채점능력평가2026_train.jsonl \
  --rationales artifacts/train_rationales.jsonl \
  --demos artifacts/demos_ordered.json \
  --ratio 16 --epochs 3 --output artifacts/main
```

## 5. 평가

```bash
python -m essay_scorer evaluate \
  --validation 글쓰기채점능력평가2026_validation.jsonl \
  --adapter artifacts/main/epoch-3 \
  --output artifacts/validation_predictions.jsonl
```

출력은 첫 시도 JSON 파싱률, 영역별 RMSE·Spearman, macro 지표를 기록합니다.
파싱 실패는 과제 규정대로 점수 0으로 계산합니다.

과제기술서의 12개 LLM Judge 항목을 동일한 프롬프트로 모의 감사할 수
있습니다.

```bash
python -m essay_scorer audit-rationales \
  --validation 글쓰기채점능력평가2026_validation.jsonl \
  --predictions artifacts/validation_predictions.jsonl \
  --output artifacts/rationale_audit.jsonl
```

## 6. train+validation 최종 재학습

선택된 ratio와 epoch 수를 고정한 뒤 두 데이터셋을 함께 학습합니다.

```bash
accelerate launch -m essay_scorer train \
  --train 글쓰기채점능력평가2026_train.jsonl \
  --rationales artifacts/train_rationales.jsonl \
  --extra-data 글쓰기채점능력평가2026_validation.jsonl \
  --extra-rationales artifacts/validation_rationales.jsonl \
  --demos artifacts/demos_ordered.json \
  --ratio 16 --epochs 3 --output artifacts/final
```

## 7. 병합 및 공개 업로드

```bash
python -m essay_scorer merge \
  --adapter artifacts/final/epoch-3 \
  --output artifacts/merged_model

python -m essay_scorer verify-merge \
  --validation 글쓰기채점능력평가2026_validation.jsonl \
  --adapter artifacts/final/epoch-3 \
  --merged artifacts/merged_model \
  --limit 30

export HF_TOKEN=...
python -m essay_scorer push \
  --model-dir artifacts/merged_model \
  --repo-id <org>/<repo>
```

토큰은 코드나 노트북에 저장하지 않습니다. 업로드되는 모델은 adapter-only가
아니라 `Qwen3_5ForConditionalGeneration` 전체 BF16 병합 모델입니다.

## 8. 제출 형태 그대로 vLLM 검증

`push` 명령이 출력하는 다음 형태의 명령을 사용합니다.

```bash
docker run --rm --gpus all -p 8000:8000 \
  vllm/vllm-openai:latest \
  <org>/<repo> \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 32768
```

최종 시험에서는 `--language-model-only`, custom code, 추가 런타임 패키지를
사용하지 않습니다. 이 명령과 `/health`, `/v1/models`,
`/v1/chat/completions`가 모두 통과해야 제출 준비 완료로 판단합니다.
