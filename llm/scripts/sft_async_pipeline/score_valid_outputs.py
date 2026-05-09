#!/usr/bin/env python3
"""Score valid-set generations and write a Markdown summary."""

from __future__ import annotations

import argparse
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path


FINAL_PATTERNS = [
    re.compile(r"####\s*([^\n]+)"),
    re.compile(r"Final Answer\s*[:：]\s*([^\n]+)", re.IGNORECASE),
    re.compile(r"The answer is\s*([^\n.]+)", re.IGNORECASE),
    re.compile(r"Answer\s*[:：]\s*([^\n]+)", re.IGNORECASE),
]

BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]+)\}")
XML_ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")
FRACTION_PATTERN = re.compile(r"-?\d+\s*/\s*-?\d+")
PLACEHOLDER_BITS = (
    "<answer>",
    "#####",
    "uetype",
    "</user>",
    "</system>",
    "assistant",
    "#user",
)


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def clean_answer(text: str | None) -> str:
    if text is None:
        return ""
    text = text.strip()
    text = text.replace(",", "")
    text = text.replace("$", "")
    text = text.replace("%", "")
    text = text.strip(" .`*[]()")

    if text.lower() == "none":
        return "None"

    fractions = FRACTION_PATTERN.findall(text)
    if len(fractions) == 1 and len(NUMBER_PATTERN.findall(text)) >= 2:
        return fractions[0].replace(" ", "")

    numbers = NUMBER_PATTERN.findall(text)
    if len(numbers) == 1:
        return numbers[0]

    if len(numbers) >= 2 and len(set(numbers)) == 1:
        return numbers[0]

    return text


def is_plausible_fragment(text: str | None) -> bool:
    if text is None:
        return False
    text = text.strip()
    if not text or re.search(r"[A-Za-z0-9]", text) is None:
        return False
    lowered = text.lower()
    return not any(bit in lowered for bit in PLACEHOLDER_BITS)


def extract_prediction(text: str) -> str:
    for candidate in reversed(XML_ANSWER_PATTERN.findall(text)):
        if is_plausible_fragment(candidate):
            return clean_answer(candidate)

    for candidate in reversed(BOXED_PATTERN.findall(text)):
        if is_plausible_fragment(candidate):
            return clean_answer(candidate)

    for pattern in FINAL_PATTERNS:
        matches = pattern.findall(text)
        for candidate in reversed(matches):
            if is_plausible_fragment(candidate):
                return clean_answer(candidate)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines[-8:]):
        if not is_plausible_fragment(line):
            continue
        fractions = FRACTION_PATTERN.findall(line)
        if len(fractions) == 1:
            return fractions[0].replace(" ", "")
        numbers = NUMBER_PATTERN.findall(line)
        if len(numbers) == 1:
            return numbers[0]

    numbers = NUMBER_PATTERN.findall(text.replace(",", ""))
    if numbers:
        return clean_answer(numbers[-1])
    return clean_answer(lines[-1] if lines else "")


def answers_match(prediction: str, gold: str) -> bool:
    pred = clean_answer(prediction)
    ref = clean_answer(gold)
    if pred == ref:
        return True

    try:
        return Decimal(pred) == Decimal(ref)
    except (InvalidOperation, ValueError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Model output JSONL path.")
    parser.add_argument("--report", required=True, help="Markdown report output path.")
    args = parser.parse_args()

    input_path = Path(args.input)
    report_path = Path(args.report)
    records = load_jsonl(input_path)
    total = len(records)
    correct = 0
    samples = []

    for record in records:
        prediction = extract_prediction(record.get("generated_text", ""))
        gold = clean_answer(record.get("answer", ""))
        matched = answers_match(prediction, gold)
        if matched:
            correct += 1
        if len(samples) < 5:
            samples.append(
                {
                    "source_index": record.get("source_index"),
                    "gold": gold,
                    "prediction": prediction,
                    "matched": matched,
                }
            )

    accuracy = 0.0 if total == 0 else correct / total
    model_name = records[0].get("model_name", "unknown_model") if records else "unknown_model"

    lines = [
        f"# {model_name} valid_1000 score",
        "",
        f"- input: `{input_path}`",
        f"- total: `{total}`",
        f"- correct: `{correct}`",
        f"- accuracy: `{accuracy:.4%}`",
        "",
        "## Sample predictions",
        "",
    ]

    for sample in samples:
        lines.append(
            f"- source_index={sample['source_index']}, gold=`{sample['gold']}`, "
            f"prediction=`{sample['prediction']}`, matched=`{sample['matched']}`"
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
