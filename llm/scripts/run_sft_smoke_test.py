#!/usr/bin/env python3
"""Run a bounded 30-sample OpenRLHF SFT smoke test."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from path_utils import infer_default_model_path, resolve_llm_env_executable, runtime_data_root, runtime_result_root


MODEL_CANDIDATE_NAMES = ("Qwen3-0.6B-Base", "Qwen3-0.6B-BASE", "Qwen3-0.6B")
SYSTEM_PROMPT = (
    "You are a careful math solver. Solve the problem step by step. "
    "End your response with a final line in the format: #### <answer>"
)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def select_shortest_sft_records(tokenizer, records: list[dict], limit: int) -> list[dict]:
    scored: list[tuple[int, int, dict]] = []
    for index, record in enumerate(records):
        token_ids = tokenizer.apply_chat_template(
            record["messages"],
            tokenize=True,
            add_generation_prompt=False,
        )
        scored.append((len(token_ids), index, record))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [record for _, _, record in scored[:limit]]


def select_shortest_valid_records(tokenizer, records: list[dict], limit: int) -> list[dict]:
    scored: list[tuple[int, int, dict]] = []
    for index, record in enumerate(records):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": record["question"]},
        ]
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        scored.append((len(token_ids), index, record))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [record for _, _, record in scored[:limit]]


def build_parser() -> argparse.ArgumentParser:
    data_root = runtime_data_root(PROJECT_ROOT)
    result_root = runtime_result_root(PROJECT_ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default=str(infer_default_model_path(PROJECT_ROOT, MODEL_CANDIDATE_NAMES)),
    )
    parser.add_argument(
        "--train-dataset",
        default=str(data_root / "deepseek-v3.2-speciale-openr1-math-3k.plus_distilled_corpus_400k_with_cot-filtered.system_prompt.sft_train.jsonl"),
    )
    parser.add_argument(
        "--validation-input",
        default=str(data_root / "valid_1000.jsonl"),
    )
    parser.add_argument(
        "--chat-template-file",
        default=str(SCRIPT_DIR / "sft_async_pipeline" / "templates" / "qwen3_06b_base_eot_chat_template.jinja"),
    )
    parser.add_argument("--sample-count", type=int, default=30)
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--micro-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-7)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-max-new-tokens", type=int, default=256)
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--train-gpu", default="0")
    parser.add_argument("--eval-device-id", default="0")
    return parser


def run_command(cmd: list[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log_handle:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env={**os.environ, "LLM_RUNTIME_ROOT": "/dev/shm/llm"},
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return_code = proc.wait()
    if return_code != 0:
        raise RuntimeError(f"SFT smoke test failed with return code {return_code}. See {log_path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    model_path = Path(args.model_path).resolve()
    data_root = runtime_data_root(PROJECT_ROOT)
    result_root = runtime_result_root(PROJECT_ROOT)
    smoke_data_root = data_root / "smoke"
    run_root = Path(args.run_root).resolve() if args.run_root else result_root / f"sft_smoke_{utc_stamp()}"
    run_root.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    train_records = select_shortest_sft_records(tokenizer, load_jsonl(Path(args.train_dataset)), args.sample_count)
    valid_records = select_shortest_valid_records(tokenizer, load_jsonl(Path(args.validation_input)), args.sample_count)

    train_subset = smoke_data_root / "sft_train_30.jsonl"
    valid_subset = smoke_data_root / "valid_30.jsonl"
    write_jsonl(train_subset, train_records)
    write_jsonl(valid_subset, valid_records)

    python_bin = resolve_llm_env_executable("python")
    cmd = [
        python_bin,
        str(SCRIPT_DIR / "sft_async_pipeline" / "run_openrlhf_sft_train_eval_best.py"),
        "--model-path",
        str(model_path),
        "--train-dataset",
        str(train_subset),
        "--validation-input",
        str(valid_subset),
        "--chat-template-file",
        str(Path(args.chat_template_file).resolve()),
        "--run-root",
        str(run_root),
        "--max-len",
        str(args.max_len),
        "--max-epochs",
        str(args.max_epochs),
        "--micro-train-batch-size",
        str(args.micro_train_batch_size),
        "--gradient-accumulation",
        str(args.gradient_accumulation),
        "--learning-rate",
        str(args.learning_rate),
        "--train-gpu",
        args.train_gpu,
        "--eval-device-id",
        args.eval_device_id,
        "--validation-limit",
        str(args.sample_count),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--eval-max-new-tokens",
        str(args.eval_max_new_tokens),
        "--tb-run-name",
        "sft_smoke_test",
        "--attn-implementation",
        "eager",
        "--eval-output-dir-name",
        "eval_valid30",
        "--final-model-dir-name",
        "best_smoke_model",
        "--skip-curve-generation",
        "--no-adam-offload",
    ]
    log_path = run_root / "smoke_launcher.log"
    run_command(cmd, log_path)

    summary_path = run_root / "evaluation_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing SFT evaluation summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(json.dumps({"run_root": str(run_root), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
