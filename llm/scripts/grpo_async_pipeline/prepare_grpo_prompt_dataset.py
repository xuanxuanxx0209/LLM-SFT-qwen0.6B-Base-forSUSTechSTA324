#!/usr/bin/env python3
"""Prepare a prompt-only GRPO dataset from the local RLHF math training data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

from answer_utils import extract_gold_label


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
COMMON_SCRIPT_DIR = PROJECT_ROOT / "scripts"
if str(COMMON_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_SCRIPT_DIR))

from path_utils import infer_default_model_path, runtime_data_root

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful math solver. Solve the problem step by step. "
    "End your response with a final line in the format: #### <answer>"
)
MODEL_CANDIDATE_NAMES = ("Qwen3-0.6B-Base", "Qwen3-0.6B-BASE", "Qwen3-0.6B")


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def maybe_inject_system_prompt(messages: list[dict], system_prompt: str) -> tuple[list[dict], bool]:
    if not system_prompt.strip():
        return messages, False

    updated = [dict(message) for message in messages]
    if updated and updated[0].get("role") == "system":
        if str(updated[0].get("content", "")).strip():
            return updated, False
        updated[0]["content"] = system_prompt
        return updated, True

    return [{"role": "system", "content": system_prompt}, *updated], True


def percentile(sorted_values: list[int], q: float) -> int:
    if not sorted_values:
        return 0
    index = int(round((len(sorted_values) - 1) * q))
    return sorted_values[index]


def main() -> None:
    data_root = runtime_data_root(PROJECT_ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(data_root / "deepseek-v3.2-speciale-openr1-math-3k.rlhf_train.jsonl"),
    )
    parser.add_argument(
        "--output",
        default=str(data_root / "deepseek-v3.2-speciale-openr1-math-3k.grpo_prompt_train.jsonl"),
    )
    parser.add_argument(
        "--metadata-output",
        default=str(data_root / "deepseek-v3.2-speciale-openr1-math-3k.grpo_prompt_train.metadata.json"),
    )
    parser.add_argument("--model-path", default=str(infer_default_model_path(PROJECT_ROOT, MODEL_CANDIDATE_NAMES)))
    parser.add_argument(
        "--max-label-chars",
        type=int,
        default=120,
        help="Keep only concise exact-match targets; set to 0 to disable length filtering.",
    )
    parser.add_argument(
        "--chat-template-file",
        default=str(SCRIPT_DIR / "templates" / "qwen3_06b_base_eot_chat_template.jinja"),
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="Injected when the RLHF record has no non-empty system prompt.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    metadata_output_path = Path(args.metadata_output).resolve()
    model_path = Path(args.model_path).resolve()
    chat_template_file = Path(args.chat_template_file).resolve()

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    tokenizer.chat_template = chat_template_file.read_text(encoding="utf-8")

    raw_records = load_jsonl(input_path)
    prepared_records = []
    skipped = {
        "empty_messages": 0,
        "no_assistant": 0,
        "empty_prompt": 0,
        "empty_label": 0,
        "unreliable_label": 0,
        "overlong_label": 0,
    }
    prompt_lengths = []
    injected_system_prompt_count = 0

    for source_index, record in enumerate(raw_records):
        messages = record.get("messages", [])
        if not messages:
            skipped["empty_messages"] += 1
            continue
        if messages[-1].get("role") != "assistant":
            skipped["no_assistant"] += 1
            continue

        prompt_messages = messages[:-1]
        if not prompt_messages:
            skipped["empty_prompt"] += 1
            continue

        assistant_content = messages[-1].get("content", "")
        label = extract_gold_label(assistant_content)
        if not label:
            skipped["unreliable_label"] += 1
            continue
        if args.max_label_chars > 0 and len(label) > args.max_label_chars:
            skipped["overlong_label"] += 1
            continue

        prompt_messages, injected_system_prompt = maybe_inject_system_prompt(prompt_messages, args.system_prompt)
        if injected_system_prompt:
            injected_system_prompt_count += 1

        prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        prompt_token_length = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
        prompt_lengths.append(prompt_token_length)

        prepared_records.append(
            {
                "source_index": source_index,
                "messages": prompt_messages,
                "label": label,
                "gold_response": assistant_content,
                "prompt_token_length": prompt_token_length,
            }
        )

    write_jsonl(output_path, prepared_records)

    sorted_prompt_lengths = sorted(prompt_lengths)
    metadata = {
        "input": str(input_path),
        "output": str(output_path),
        "model_path": str(model_path),
        "chat_template_file": str(chat_template_file),
        "system_prompt": args.system_prompt,
        "injected_system_prompt_count": injected_system_prompt_count,
        "max_label_chars": args.max_label_chars,
        "raw_records": len(raw_records),
        "prepared_records": len(prepared_records),
        "skipped": skipped,
        "prompt_token_stats": {
            "min": min(sorted_prompt_lengths) if sorted_prompt_lengths else 0,
            "p50": percentile(sorted_prompt_lengths, 0.50),
            "p90": percentile(sorted_prompt_lengths, 0.90),
            "p95": percentile(sorted_prompt_lengths, 0.95),
            "p99": percentile(sorted_prompt_lengths, 0.99),
            "max": max(sorted_prompt_lengths) if sorted_prompt_lengths else 0,
        },
    }
    metadata_output_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
