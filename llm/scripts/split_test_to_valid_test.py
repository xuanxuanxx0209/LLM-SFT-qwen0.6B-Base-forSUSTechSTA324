#!/usr/bin/env python3
"""Split the source JSONL into fixed-size valid/test JSONL files.

This version enforces that the validation and test splits come from disjoint
`seed_question` pools. It keeps the original fields intact and only adds:
- source_index: original 0-based line number in the source file
- split: target split name, either "valid" or "test"
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from path_utils import runtime_data_root


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["source_index"] = idx
            records.append(record)
    return records


def write_jsonl(path: Path, records: list[dict], split_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            output = dict(record)
            output["split"] = split_name
            f.write(json.dumps(output, ensure_ascii=False) + "\n")


def main() -> None:
    data_root = runtime_data_root(PROJECT_ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(data_root / "test-00000-of-00001.jsonl"),
        help="Source JSONL file.",
    )
    parser.add_argument(
        "--valid-output",
        default=str(data_root / "valid_1000.jsonl"),
        help="Output JSONL for the sampled validation set.",
    )
    parser.add_argument(
        "--test-output",
        default=str(data_root / "test_1000.jsonl"),
        help="Output JSONL for the sampled test set.",
    )
    parser.add_argument("--valid-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    valid_output = Path(args.valid_output)
    test_output = Path(args.test_output)

    records = load_jsonl(input_path)
    rng = random.Random(args.seed)
    by_seed_question: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_seed_question[record["seed_question"]].append(record)

    seed_questions = list(by_seed_question)
    if len(seed_questions) < 2:
        raise ValueError("Need at least 2 distinct seed_question values.")

    rng.shuffle(seed_questions)
    midpoint = len(seed_questions) // 2
    valid_seed_questions = set(seed_questions[:midpoint])
    test_seed_questions = set(seed_questions[midpoint:])

    if not valid_seed_questions or not test_seed_questions:
        raise RuntimeError("Failed to create two non-empty seed_question partitions.")
    if valid_seed_questions & test_seed_questions:
        raise RuntimeError("valid/test seed_question overlap detected, which should be impossible.")

    valid_pool = []
    for seed_question in valid_seed_questions:
        valid_pool.extend(by_seed_question[seed_question])

    test_pool = []
    for seed_question in test_seed_questions:
        test_pool.extend(by_seed_question[seed_question])

    if len(valid_pool) < args.valid_size:
        raise ValueError(
            f"Validation pool too small after seed_question split: "
            f"need {args.valid_size}, found {len(valid_pool)}."
        )
    if len(test_pool) < args.test_size:
        raise ValueError(
            f"Test pool too small after seed_question split: "
            f"need {args.test_size}, found {len(test_pool)}."
        )

    rng.shuffle(valid_pool)
    rng.shuffle(test_pool)
    valid_records = valid_pool[: args.valid_size]
    test_records = test_pool[: args.test_size]

    write_jsonl(valid_output, valid_records, "valid")
    write_jsonl(test_output, test_records, "test")

    print(f"source={input_path}")
    print(f"total_records={len(records)}")
    print(f"unique_seed_questions={len(seed_questions)}")
    print(f"valid_seed_questions={len(valid_seed_questions)}")
    print(f"test_seed_questions={len(test_seed_questions)}")
    print(f"valid_pool_records={len(valid_pool)}")
    print(f"test_pool_records={len(test_pool)}")
    print(f"valid_records={len(valid_records)} -> {valid_output}")
    print(f"test_records={len(test_records)} -> {test_output}")
    print(f"seed={args.seed}")


if __name__ == "__main__":
    main()
