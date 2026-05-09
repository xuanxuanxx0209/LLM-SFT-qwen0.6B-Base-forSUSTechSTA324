#!/usr/bin/env python3
"""Run validation-set generation and scoring for one SFT checkpoint."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from transformers import AutoTokenizer

from score_valid_outputs import answers_match, clean_answer, extract_prediction, load_jsonl


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent


def count_output_tokens(tokenizer_path: Path, output_path: Path) -> tuple[int, float, int, int]:
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)
    total_tokens = 0
    max_tokens = 0
    min_tokens = None
    records = load_jsonl(output_path)
    for record in records:
        n = len(tokenizer(record.get("generated_text", ""), add_special_tokens=False)["input_ids"])
        total_tokens += n
        max_tokens = max(max_tokens, n)
        min_tokens = n if min_tokens is None else min(min_tokens, n)
    avg_tokens = 0.0 if min_tokens is None else total_tokens / len(records)
    return total_tokens, avg_tokens, min_tokens or 0, max_tokens


def compute_accuracy(output_path: Path) -> tuple[int, int, float]:
    records = load_jsonl(output_path)
    total = len(records)
    correct = 0
    for record in records:
        prediction = extract_prediction(record.get("generated_text", ""))
        gold = clean_answer(record.get("answer", ""))
        if answers_match(prediction, gold):
            correct += 1
    accuracy = 0.0 if total == 0 else correct / total
    return correct, total, accuracy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--validation-input", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--engine", choices=["auto", "vllm", "transformers"], default="auto")
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--device-ids", default="0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--requested-max-new-tokens", type=int, default=4096)
    parser.add_argument("--chat-template-file", default=None)
    parser.add_argument(
        "--vllm-attention-backend",
        default="auto",
        choices=["auto", "FLASH_ATTN", "TRITON_ATTN", "FLASHINFER", "FLEX_ATTENTION"],
    )
    parser.add_argument("--vllm-enforce-eager", action="store_true", default=False)
    parser.add_argument("--vllm-disable-compilation", action="store_true", default=False)
    parser.add_argument(
        "--vllm-scheduler-mode",
        default="full_queue",
        choices=["static_batch", "full_queue", "server_async"],
    )
    parser.add_argument("--attn-implementation", default="eager")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    tokenizer_path = Path(args.tokenizer_path) if args.tokenizer_path is not None else model_path
    output_root = Path(args.output_root)
    eval_dir = output_root / args.label
    eval_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    engine_used = args.engine

    vllm_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "run_vllm_valid_single_gpu.py"),
        "--model-path",
        str(model_path),
        "--tokenizer-path",
        str(tokenizer_path),
        "--input",
        args.validation_input,
        "--output-dir",
        str(eval_dir),
        "--limit",
        str(args.limit),
        "--num-workers",
        str(args.num_workers),
        "--device-ids",
        args.device_ids,
        "--batch-size",
        str(args.batch_size),
        "--requested-max-new-tokens",
        str(args.requested_max_new_tokens),
        "--attention-backend",
        args.vllm_attention_backend,
        "--scheduler-mode",
        args.vllm_scheduler_mode,
    ]
    if args.vllm_enforce_eager:
        vllm_cmd.append("--enforce-eager")
    if args.vllm_disable_compilation:
        vllm_cmd.append("--disable-compilation")
    hf_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "run_hf_valid_single_gpu.py"),
        "--model-path",
        str(model_path),
        "--tokenizer-path",
        str(tokenizer_path),
        "--input",
        args.validation_input,
        "--output-dir",
        str(eval_dir),
        "--limit",
        str(args.limit),
        "--device-id",
        args.device_ids.split(",")[0].strip(),
        "--batch-size",
        str(args.batch_size),
        "--requested-max-new-tokens",
        str(args.requested_max_new_tokens),
        "--attn-implementation",
        args.attn_implementation,
    ]
    if args.chat_template_file is not None:
        vllm_cmd.extend(["--chat-template-file", args.chat_template_file])
        hf_cmd.extend(["--chat-template-file", args.chat_template_file])

    if args.engine == "vllm":
        subprocess.run(vllm_cmd, check=True)
    elif args.engine == "transformers":
        if args.num_workers != 1:
            raise ValueError("Transformers engine currently supports only --num-workers 1.")
        subprocess.run(hf_cmd, check=True)
    else:
        try:
            subprocess.run(vllm_cmd, check=True)
            engine_used = "vllm"
        except subprocess.CalledProcessError:
            if args.num_workers != 1:
                raise
            subprocess.run(hf_cmd, check=True)
            engine_used = "transformers"
    elapsed = time.perf_counter() - start

    output_path = eval_dir / f"{model_path.name}.valid_1000.outputs.jsonl"
    report_path = eval_dir / f"{model_path.name}.valid_1000.score.md"
    score_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "score_valid_outputs.py"),
        "--input",
        str(output_path),
        "--report",
        str(report_path),
    ]
    subprocess.run(score_cmd, check=True)

    correct, total, accuracy = compute_accuracy(output_path)
    total_output_tokens, avg_output_tokens, min_output_tokens, max_output_tokens = count_output_tokens(
        tokenizer_path, output_path
    )

    summary = {
        "label": args.label,
        "model_path": str(model_path),
        "tokenizer_path": str(tokenizer_path),
        "validation_input": args.validation_input,
        "output_dir": str(eval_dir),
        "engine": engine_used,
        "num_workers": args.num_workers,
        "device_ids": args.device_ids.split(","),
        "requested_max_new_tokens": args.requested_max_new_tokens,
        "limit": args.limit,
        "chat_template_file": args.chat_template_file,
        "vllm_attention_backend": args.vllm_attention_backend,
        "vllm_enforce_eager": args.vllm_enforce_eager,
        "vllm_disable_compilation": args.vllm_disable_compilation,
        "vllm_scheduler_mode": args.vllm_scheduler_mode,
        "elapsed_seconds": elapsed,
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "total_output_tokens": total_output_tokens,
        "avg_output_tokens": avg_output_tokens,
        "min_output_tokens": min_output_tokens,
        "max_output_tokens": max_output_tokens,
        "output_tokens_per_second": 0.0 if elapsed == 0 else total_output_tokens / elapsed,
    }
    summary_path = eval_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
