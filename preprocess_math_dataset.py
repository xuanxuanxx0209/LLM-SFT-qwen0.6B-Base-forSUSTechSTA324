#!/usr/bin/env python3
"""Preprocess math dataset to <think></think>\boxed{} format."""

import json
import sys
from pathlib import Path


def extract_boxed(text: str):
    """Extract reasoning and \\boxed{answer} from text."""
    # Find last \\boxed{ (handle both single and double backslash from JSON)
    idx = text.rfind('\\\\boxed{')
    prefix = '\\\\boxed{'
    if idx == -1:
        idx = text.rfind('\\boxed{')
        prefix = '\\boxed{'

    if idx == -1:
        return None, None

    reasoning = text[:idx].strip()
    start = idx + len(prefix)
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1

    answer = text[start:i-1]
    full_boxed = prefix + answer + '}'
    return reasoning, full_boxed


def main():
    input_path = Path("/home/ubuntu/jgy-dataset/final_cleaned_high_quality.jsonl")
    output_path = Path("/home/ubuntu/jgy-dataset/math_sft_think_boxed.jsonl")

    SYSTEM_PROMPT = (
        "You are a helpful math assistant. When solving math problems, "
        "first think step by step inside <think> tags, then provide your final answer in \\boxed{}."
    )

    kept = 0
    skipped = 0
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            data = json.loads(line)
            messages = data.get("messages", [])

            # Remove empty system messages and add our system prompt
            messages = [m for m in messages if not (m.get("role") == "system" and not m.get("content", "").strip())]

            # Check if first message is system
            if messages and messages[0].get("role") == "system":
                messages[0]["content"] = SYSTEM_PROMPT
            else:
                messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

            # Last message should be assistant
            if not messages or messages[-1].get("role") != "assistant":
                skipped += 1
                continue

            assistant_content = messages[-1]["content"]
            reasoning, boxed = extract_boxed(assistant_content)

            if reasoning is None:
                # No boxed found: wrap whole response in <think> without boxed
                messages[-1]["content"] = f"<think>{assistant_content}</think>"
                kept += 1
            else:
                # Reformat to <think>reasoning</think>\boxed{answer}
                messages[-1]["content"] = f"<think>{reasoning}</think>{boxed}"
                kept += 1

            fout.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")

    print(f"Preprocessing complete: {kept} kept, {skipped} skipped.")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
