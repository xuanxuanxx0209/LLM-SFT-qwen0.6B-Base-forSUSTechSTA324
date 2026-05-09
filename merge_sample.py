import json
import random
import sys
from pathlib import Path

BASE = Path("/home/ubuntu/jgy-dataset/final_en_sft_scoring_aligned.jsonl")
EXTRA = Path("/home/ubuntu/jgy-dataset/general_data_200.jsonl")
OUTPUT = Path("/home/ubuntu/jgy-dataset/final_en_sft_scoring_aligned_plus_general100.jsonl")
SAMPLE_N = 100
SEED = 42


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[skip] {path.name} line {line_no}: {e}", file=sys.stderr)
    return rows


def main() -> None:
    base = read_jsonl(BASE)
    extra = read_jsonl(EXTRA)

    if len(extra) < SAMPLE_N:
        print(f"!! extra only has {len(extra)} rows, cannot sample {SAMPLE_N}", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(SEED)
    sampled = rng.sample(extra, SAMPLE_N)

    base_ids = {r.get("id") for r in base if r.get("id") is not None}
    overlap = [r.get("id") for r in sampled if r.get("id") in base_ids]
    if overlap:
        print(f"[warn] {len(overlap)} sampled ids already exist in base, e.g. {overlap[:5]}", file=sys.stderr)

    merged = base + sampled

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"base    : {len(base)} rows  ({BASE})")
    print(f"sampled : {len(sampled)} rows  (from {len(extra)} in {EXTRA.name}, seed={SEED})")
    print(f"output  : {len(merged)} rows  -> {OUTPUT}")


if __name__ == "__main__":
    main()
