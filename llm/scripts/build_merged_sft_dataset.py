#!/usr/bin/env python3
"""Merge the current SFT dataset with the distilled corpus into one SFT JSONL.

Output schema matches the existing OpenRLHF SFT data:
{"messages": [{"role": ... , "content": ...}, ...]}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from path_utils import runtime_data_root

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful assistant. "
    "Reason step by step inside <think>...</think>, "
    "then give the final answer clearly and directly."
)


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


def normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    return text.strip()


def ensure_assistant_content_has_think_and_answer(thinking: str, solution: str) -> str:
    thinking = normalize_text(thinking)
    solution = normalize_text(solution)

    if thinking.startswith("<think>") and "</think>" in thinking:
        think_block = thinking
    else:
        think_block = f"<think>{thinking}</think>"

    if solution:
        return f"{think_block}\n{solution}"
    return think_block


def normalize_existing_sft_record(record: dict, system_prompt: str) -> dict:
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        raise ValueError("Existing SFT record does not contain a valid messages list.")

    normalized = []
    for idx, message in enumerate(messages):
        role = message.get("role", "")
        content = normalize_text(message.get("content", ""))
        if idx == 0 and role == "system":
            content = system_prompt
        normalized.append({"role": role, "content": content})

    if normalized[0]["role"] != "system":
        normalized.insert(0, {"role": "system", "content": system_prompt})
    return {"messages": normalized}


def convert_distilled_record(record: dict, system_prompt: str) -> dict:
    user_content = normalize_text(record.get("problem", ""))
    assistant_content = ensure_assistant_content_has_think_and_answer(
        thinking=record.get("thinking", ""),
        solution=record.get("solution", ""),
    )
    if not user_content or not assistant_content:
        raise ValueError("Distilled record is missing problem/thinking/solution content.")

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def main() -> None:
    data_root = runtime_data_root(PROJECT_ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--existing-sft",
        default=str(data_root / "deepseek-v3.2-speciale-openr1-math-3k.sft_train.jsonl"),
    )
    parser.add_argument(
        "--distilled-corpus",
        default=str(data_root / "distilled_corpus_400k_with_cot-filtered.jsonl"),
    )
    parser.add_argument(
        "--output",
        default=str(data_root / "deepseek-v3.2-speciale-openr1-math-3k.plus_distilled_corpus_400k_with_cot-filtered.system_prompt.sft_train.jsonl"),
    )
    parser.add_argument(
        "--metadata-output",
        default=str(data_root / "deepseek-v3.2-speciale-openr1-math-3k.plus_distilled_corpus_400k_with_cot-filtered.system_prompt.sft_train.metadata.json"),
    )
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--distilled-category", default=None)
    args = parser.parse_args()

    existing_sft_path = Path(args.existing_sft)
    distilled_corpus_path = Path(args.distilled_corpus)
    output_path = Path(args.output)
    metadata_output_path = Path(args.metadata_output)

    existing_sft_records = load_jsonl(existing_sft_path)
    distilled_records = load_jsonl(distilled_corpus_path)
    if args.distilled_category is not None:
        distilled_records = [record for record in distilled_records if record.get("category") == args.distilled_category]

    normalized_existing = [normalize_existing_sft_record(record, args.system_prompt) for record in existing_sft_records]
    normalized_distilled = [convert_distilled_record(record, args.system_prompt) for record in distilled_records]

    merged_records = normalized_existing + normalized_distilled
    write_jsonl(output_path, merged_records)

    metadata = {
        "existing_sft": str(existing_sft_path),
        "distilled_corpus": str(distilled_corpus_path),
        "output": str(output_path),
        "system_prompt": args.system_prompt,
        "distilled_category": args.distilled_category,
        "existing_sft_records": len(normalized_existing),
        "distilled_records": len(normalized_distilled),
        "merged_records": len(merged_records),
    }
    metadata_output_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
