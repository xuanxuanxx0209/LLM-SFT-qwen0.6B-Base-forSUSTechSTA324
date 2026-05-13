#!/usr/bin/env python3
"""Build LoRA training dataset for critical-thinking weakness.

Strategy:
- Synthesize ~150 'missing info' problems by removing a $-amount or numeric phrase
  from clean GSM8K-ish questions in 7460_cleaned.jsonl. Generate templated answer
  ending in \\boxed{None}.
- Add ~400 unmodified samples from same pool as a replay buffer to prevent
  catastrophic forgetting on other categories.
- Cross-check 0 overlap with /home/ubuntu/validation-set/valid_1000.jsonl.

Output: /home/ubuntu/jgy-dataset/lora_weakness_train.jsonl  (messages format)
"""
import json
import random
import re
from pathlib import Path

SOURCE = "/home/ubuntu/jgy-dataset/7460_cleaned.jsonl"
EVAL_SET = "/home/ubuntu/validation-set/valid_1000.jsonl"
OUT_PATH = "/home/ubuntu/jgy-dataset/lora_weakness_train_v4.jsonl"

TARGET_CT = 150
TARGET_REPLAY = 700
SEED = 42

# Patterns: (regex, description for "we don't know X")
# Order matters — try most specific first.
REMOVE_PATTERNS = [
    (re.compile(r'\bat\s+\$[\d.,]+(?:\s+(?:per|each|apiece|a\s+\w+))?\b', re.I), "price"),
    (re.compile(r'\bcosts?\s+\$[\d.,]+(?:\s+(?:per|each|apiece|a\s+\w+))?\b', re.I), "cost"),
    (re.compile(r'\bfor\s+\$[\d.,]+(?:\s+(?:per|each|apiece|a\s+\w+))?\b', re.I), "price"),
    (re.compile(r'\bsells?\s+(?:for|at)\s+\$[\d.,]+\b', re.I), "selling price"),
    (re.compile(r'\bpriced\s+at\s+\$[\d.,]+\b', re.I), "price"),
    (re.compile(r'\$[\d,]+(?:\.\d+)?(?:\s+(?:per|each|apiece|a\s+\w+))?\b', re.I), "amount"),
    (re.compile(r'\b\d+\s+(?:dollars?|cents?|bucks?)(?:\s+(?:per|each|apiece|a\s+\w+))?\b', re.I), "amount"),
    (re.compile(r'\b\d+(?:\.\d+)?\s+(?:miles?\s+per\s+(?:hour|day|gallon))\b', re.I), "rate"),
    (re.compile(r'\b\d+(?:\.\d+)?\s*%\s+(?:discount|off|tax|interest)\b', re.I), "rate"),
]

ANSWER_TEMPLATES = [
    "We don't have enough information about the {what} in the problem. Without it, we cannot compute a numeric answer.\n\n\\boxed{{None}}",
    "The problem does not specify the {what}. Therefore, the answer cannot be determined from the given information.\n\n\\boxed{{None}}",
    "Critical information is missing: the {what} is not provided. We cannot solve this without it.\n\n\\boxed{{None}}",
    "The {what} is not given in the problem statement, so the question is unanswerable as stated.\n\n\\boxed{{None}}",
    "There is insufficient information to solve this problem; specifically, the {what} is not provided.\n\n\\boxed{{None}}",
]


def get_user_q(item):
    for m in item.get("messages", []):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def looks_gsm_like(q):
    """Cheap filter: GSM8K-style word problem heuristic."""
    if not q or len(q) < 30 or len(q) > 800:
        return False
    # Must contain at least one digit and end with '?'
    if not re.search(r"\d", q):
        return False
    if not q.rstrip().endswith("?"):
        return False
    # Reject latex-heavy problems
    if q.count("\\") > 2 or "\\frac" in q or "\\int" in q or "$$" in q:
        return False
    # Reject chemistry/physics jargon
    bad = ["mol", "kg/m", "Hz", "kelvin", "molar", "C_p", "molarity"]
    if any(b in q for b in bad):
        return False
    return True


def perturb(q):
    """Remove ONE numeric phrase. Return (perturbed_q, missing_what) or None."""
    for pat, label in REMOVE_PATTERNS:
        m = pat.search(q)
        if not m:
            continue
        new_q = (q[: m.start()] + q[m.end():]).strip()
        # Clean up double-spaces and stray punctuation introduced
        new_q = re.sub(r"\s+", " ", new_q)
        new_q = re.sub(r"\s+([,.;:?!])", r"\1", new_q)
        new_q = re.sub(r",\s*\.", ".", new_q)
        # Reject if removal made the sentence too short or weird
        if len(new_q) < 20 or len(new_q) > 800:
            continue
        if not new_q.rstrip().endswith("?"):
            continue
        return new_q, label
    return None


def main():
    rng = random.Random(SEED)

    # Load eval set questions for leakage check
    eval_qs = set()
    with open(EVAL_SET) as f:
        for line in f:
            eval_qs.add(json.loads(line).get("question", "").strip())
    print(f"[load] eval_qs (valid_1000): {len(eval_qs)}")

    # Load clean pool
    with open(SOURCE) as f:
        pool = [json.loads(l) for l in f]
    rng.shuffle(pool)
    print(f"[load] clean pool (7460_cleaned): {len(pool)}")

    # Synthesize critical-thinking samples
    ct_synth = []
    for item in pool:
        if len(ct_synth) >= TARGET_CT:
            break
        q = get_user_q(item)
        if not looks_gsm_like(q):
            continue
        if q.strip() in eval_qs:
            continue
        result = perturb(q)
        if result is None:
            continue
        new_q, label = result
        if new_q.strip() in eval_qs:  # double-check post-perturbation
            continue
        ans = rng.choice(ANSWER_TEMPLATES).format(what=label)
        ct_synth.append({
            "messages": [
                {"role": "system", "content": ""},
                {"role": "user", "content": new_q},
                {"role": "assistant", "content": ans},
            ],
            "id": f"ct_synth_{len(ct_synth):04d}",
        })
    print(f"[synthesize] critical-thinking samples: {len(ct_synth)}")

    # Replay buffer: random untouched samples (skip any whose question is in eval set)
    used_qs = {get_user_q(item).strip() for item in pool[: TARGET_CT * 5]}  # rough
    replay = []
    for item in pool[TARGET_CT * 5:]:
        if len(replay) >= TARGET_REPLAY:
            break
        q = get_user_q(item).strip()
        if q in eval_qs:
            continue
        if q in used_qs:
            continue
        replay.append({
            "messages": item["messages"],
            "id": f"replay_{len(replay):04d}",
        })
    print(f"[replay] unmodified samples: {len(replay)}")

    # Combine and shuffle
    all_items = ct_synth + replay
    rng.shuffle(all_items)

    # Final leakage check
    leaks = 0
    for it in all_items:
        q = get_user_q(it).strip()
        if q in eval_qs:
            leaks += 1
    print(f"[verify] leak count (must be 0): {leaks}")
    assert leaks == 0, "DATA LEAKAGE: aborting before write."

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for it in all_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"[write] {len(all_items)} samples → {OUT_PATH}")
    print(f"  - critical-thinking synthesized: {len(ct_synth)}")
    print(f"  - replay buffer: {len(replay)}")

    # Print a couple synthesized examples
    print("\n=== Sample synthesized critical-thinking ===")
    for s in ct_synth[:2]:
        print(f"User: {s['messages'][1]['content']}")
        print(f"Asst: {s['messages'][2]['content']}")
        print()


if __name__ == "__main__":
    main()
