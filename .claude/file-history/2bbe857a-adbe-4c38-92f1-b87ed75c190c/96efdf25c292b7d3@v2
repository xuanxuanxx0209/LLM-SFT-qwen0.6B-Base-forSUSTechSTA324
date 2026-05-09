import json
import re
import sys
from pathlib import Path


def extract_boxed_contents(text: str) -> list[str]:
    """Extract all \\boxed{...} contents with proper nested-brace matching."""
    results = []
    i = 0
    key = r"\boxed{"
    while True:
        idx = text.find(key, i)
        if idx == -1:
            break
        start = idx + len(key)
        depth = 1
        j = start
        while j < len(text) and depth > 0:
            ch = text[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if depth == 0:
            results.append(text[start:j])
            i = j + 1
        else:
            results.append(text[start:])
            break
    return results


def is_none_value(s: str) -> bool:
    """Treat empty/whitespace and 'None'/'null' (any case) as none."""
    stripped = s.strip()
    if not stripped:
        return True
    return stripped.lower() in {"none", "null", "n/a", "na"}


def main(path: str) -> None:
    p = Path(path)
    none_count = 0
    no_boxed_count = 0
    total = 0
    none_ids = []

    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[skip] line {line_no} JSON error: {e}", file=sys.stderr)
                continue

            # Search across all string fields, but answer/solution are most likely
            haystacks = []
            for k in ("answer", "solution", "question"):
                v = obj.get(k)
                if isinstance(v, str):
                    haystacks.append(v)
            text = "\n".join(haystacks)

            boxes = extract_boxed_contents(text)
            if not boxes:
                no_boxed_count += 1
                continue

            if any(is_none_value(b) for b in boxes):
                none_count += 1
                none_ids.append(obj.get("id", f"line:{line_no}"))

    print(f"Total records      : {total}")
    print(f"No \\boxed found   : {no_boxed_count}")
    print(f"\\boxed is None    : {none_count}")
    if none_ids:
        print("\nIDs with None inside \\boxed:")
        for i in none_ids[:50]:
            print(f"  - {i}")
        if len(none_ids) > 50:
            print(f"  ... and {len(none_ids) - 50} more")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/jgy-dataset/final_en_sft_scoring_aligned.jsonl"
    main(target)
