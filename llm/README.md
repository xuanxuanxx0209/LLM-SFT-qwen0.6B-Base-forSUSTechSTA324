# LLM Runtime Guide

This workspace now uses `/dev/shm/llm` as the default runtime root.

## Runtime Layout

- `scripts`: `/dev/shm/llm/scripts` -> symlink to the maintained scripts in `/home/ubuntu/llm/scripts`
- `models`: `/dev/shm/llm/models` -> symlink to `/dev/shm/models`
- `data`: `/dev/shm/llm/data`
- `result`: `/dev/shm/llm/result`
- `cache`: `/dev/shm/llm/cache`

The Python and shell entrypoints were migrated so their default model, data, result, and cache paths resolve under `/dev/shm/llm`.

## Environment

```bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate llm
export LLM_RUNTIME_ROOT=/dev/shm/llm
```

## 1. Async vLLM Inference

Verified script:

```bash
python /dev/shm/llm/scripts/run_qwen3_vllm_async_benchmark.py
```

Default behavior:

- model: `/dev/shm/llm/models/Qwen3-0.6B`
- input: `/dev/shm/llm/data/valid_1000.jsonl`
- output: `/dev/shm/llm/result/vllm_qwen3_06b_async_<timestamp>/`
- starts a local OpenAI-compatible vLLM server
- sends async chat requests
- saves `outputs.jsonl` and `summary.json`
- renders performance and sample generations in the terminal

Useful overrides:

```bash
python /dev/shm/llm/scripts/run_qwen3_vllm_async_benchmark.py \
  --max-samples 1000 \
  --concurrency 8 \
  --max-tokens 192 \
  --port 18083 \
  --display-samples 3
```

Latest verified run:

- `/dev/shm/llm/result/vllm_qwen3_06b_async_20260317_073619`

## 2. SFT Smoke Test

Verified script:

```bash
python /dev/shm/llm/scripts/run_sft_smoke_test.py
```

What it does:

- loads `/dev/shm/llm/models/Qwen3-0.6B-Base`
- selects the shortest 30 training samples and 30 validation samples
- writes them into `/dev/shm/llm/data/smoke/`
- runs the OpenRLHF SFT orchestrator
- evaluates the checkpoint with vLLM

Main output:

- run root: `/dev/shm/llm/result/sft_smoke_<timestamp>/`
- subset data: `/dev/shm/llm/data/smoke/sft_train_30.jsonl`
- subset valid: `/dev/shm/llm/data/smoke/valid_30.jsonl`

Latest verified run:

- `/dev/shm/llm/result/sft_smoke_20260317_074132`

## 3. GRPO Smoke Test

Verified script:

```bash
ray stop --force >/dev/null 2>&1 || true
python /dev/shm/llm/scripts/run_grpo_smoke_test.py
```

What it does:

- loads `/dev/shm/llm/models/Qwen3-0.6B-Base`
- selects the shortest 30 GRPO prompt samples and 30 validation samples
- writes them into `/dev/shm/llm/data/smoke/`
- runs OpenRLHF GRPO training with local reward script
- evaluates the checkpoint with vLLM

Main output:

- run root: `/dev/shm/llm/result/grpo_smoke_<timestamp>/`
- subset data: `/dev/shm/llm/data/smoke/grpo_train_30.jsonl`
- subset valid: `/dev/shm/llm/data/smoke/grpo_valid_30.jsonl`

Latest verified run:

- `/dev/shm/llm/result/grpo_smoke_20260317_074648`

## Notes

- For inference, the default model is `Qwen3-0.6B`.
- For SFT and GRPO smoke tests, the default training model is `Qwen3-0.6B-Base`.
- If you want to override any default path, set one of:
  - `LLM_RUNTIME_ROOT`
  - `LLM_MODEL_ROOT`
  - `LLM_DATA_ROOT`
  - `LLM_RESULT_ROOT`
  - `LLM_CACHE_ROOT`
  - `MODEL_PATH`
