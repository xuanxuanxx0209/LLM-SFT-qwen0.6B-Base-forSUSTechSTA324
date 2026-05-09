#!/usr/bin/env python3
"""Score valid-set generations and write a Markdown summary."""

from __future__ import annotations

import argparse
from pathlib import Path

from answer_utils import answers_match, clean_answer, extract_prediction, load_jsonl


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
