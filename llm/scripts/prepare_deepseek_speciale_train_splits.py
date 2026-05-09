#!/usr/bin/env python3
"""Prepare SFT and RLHF training splits from the DeepSeek speciale dataset.

Rules:
- tokenize each sample via the local chat template
- samples with token_length < 4096 form the short pool
- about 2/3 of the short pool goes to SFT, the rest goes to RLHF
- samples with 4096 <= token_length < 6000 also go to RLHF
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from transformers import AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from path_utils import infer_default_model_path, runtime_data_root

MODEL_CANDIDATE_NAMES = ("Qwen3-0.6B-Base", "Qwen3-0.6B-BASE", "Qwen3-0.6B", "Qwen2.5-Math-1.5B")


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["_source_index"] = idx
            records.append(record)
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            output = {k: v for k, v in record.items() if not k.startswith("_")}
            f.write(json.dumps(output, ensure_ascii=False) + "\n")


def maybe_override_chat_template(tokenizer, chat_template_file: str | None) -> None:
    if chat_template_file is None:
        return
    tokenizer.chat_template = Path(chat_template_file).read_text(encoding="utf-8")


def token_length(tokenizer, messages: list[dict]) -> int:
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    return len(input_ids)


def main() -> None:
    data_root = runtime_data_root(PROJECT_ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(data_root / "deepseek-v3.2-speciale-openr1-math-3k.jsonl"),
    )
    parser.add_argument(
        "--model-path",
        default=str(infer_default_model_path(PROJECT_ROOT, MODEL_CANDIDATE_NAMES)),
    )
    parser.add_argument("--chat-template-file", default=None)
    parser.add_argument(
        "--sft-output",
        default=str(data_root / "deepseek-v3.2-speciale-openr1-math-3k.sft_train.jsonl"),
    )
    parser.add_argument(
        "--rlhf-output",
        default=str(data_root / "deepseek-v3.2-speciale-openr1-math-3k.rlhf_train.jsonl"),
    )
    parser.add_argument(
        "--metadata-output",
        default=str(data_root / "deepseek-v3.2-speciale-openr1-math-3k.train_split.metadata.json"),
    )
    parser.add_argument("--sft-max-tokens", type=int, default=4096)
    parser.add_argument("--rlhf-max-tokens", type=int, default=6000)
    parser.add_argument("--sft-short-ratio", type=float, default=2 / 3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    sft_output = Path(args.sft_output)
    rlhf_output = Path(args.rlhf_output)
    metadata_output = Path(args.metadata_output)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    maybe_override_chat_template(tokenizer, args.chat_template_file)
    records = load_jsonl(input_path)

    short_pool = []
    rlhf_extra_pool = []
    too_long_pool = []

    for record in records:
        n_tokens = token_length(tokenizer, record["messages"])
        record["_token_length"] = n_tokens
        if n_tokens < args.sft_max_tokens:
            short_pool.append(record)
        elif n_tokens < args.rlhf_max_tokens:
            rlhf_extra_pool.append(record)
        else:
            too_long_pool.append(record)

    rng = random.Random(args.seed)
    rng.shuffle(short_pool)

    sft_count = int(round(len(short_pool) * args.sft_short_ratio))
    sft_count = max(0, min(sft_count, len(short_pool)))
    sft_records = short_pool[:sft_count]
    rlhf_records = short_pool[sft_count:] + rlhf_extra_pool

    write_jsonl(sft_output, sft_records)
    write_jsonl(rlhf_output, rlhf_records)

    metadata = {
        "input_path": str(input_path),
        "model_path": args.model_path,
        "chat_template_file": args.chat_template_file,
        "seed": args.seed,
        "sft_max_tokens_exclusive": args.sft_max_tokens,
        "rlhf_max_tokens_exclusive": args.rlhf_max_tokens,
        "sft_short_ratio": args.sft_short_ratio,
        "total_records": len(records),
        "short_pool_records": len(short_pool),
        "rlhf_extra_pool_records": len(rlhf_extra_pool),
        "too_long_records": len(too_long_pool),
        "sft_records": len(sft_records),
        "rlhf_from_short_pool_records": len(short_pool) - len(sft_records),
        "rlhf_total_records": len(rlhf_records),
        "combined_output_records": len(sft_records) + len(rlhf_records),
        "sft_output": str(sft_output),
        "rlhf_output": str(rlhf_output),
    }
    metadata_output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
