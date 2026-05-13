#!/usr/bin/env python3
"""Eval V3+LoRA-merged on valid_1000 and dump per-item results
with perturbation_type, then print accuracy bucketed by category.

Companion to run_eval_with_dump.py (which evaluates the V3 baseline)."""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/home/ubuntu/validation-set")
from score_valid_outputs import extract_prediction, answers_match

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

MODEL_PATH = "/home/ubuntu/qwen3-0.6B-mathsft-V3-lora-merged-v6"
DATA_PATH = "/home/ubuntu/validation-set/valid_1000.jsonl"
DUMP_PATH = "/home/ubuntu/eval_dumps/V3_lora_merged_v6_per_item.jsonl"
TAG = "V3+LoRA-v6"

SYSTEM_PROMPT = (
    "You are a clear, careful, and reliable assistant. "
    "If reasoning is included, it must appear only under the Thinking section. "
    "The final user-facing reply must be written as a complete answer under Answer. "
    "Do not produce any tool call, function call, plugin request, or external action format."
)

print(f"[{TAG}] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

print(f"[{TAG}] Loading model with vLLM...")
llm = LLM(
    model=MODEL_PATH,
    trust_remote_code=True,
    dtype="bfloat16",
    max_model_len=4096,
    gpu_memory_utilization=0.9,
)

print(f"[{TAG}] Reading dataset...")
items = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        items.append(json.loads(line))
print(f"[{TAG}] Total questions: {len(items)}")

prompts = []
for it in items:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": it["question"]},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompts.append(prompt)

sampling_params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=2048,
)

print(f"[{TAG}] Running inference...")
outputs = llm.generate(prompts, sampling_params)

Path(DUMP_PATH).parent.mkdir(parents=True, exist_ok=True)
correct = 0
total = len(outputs)
bucket_n = defaultdict(int)
bucket_correct = defaultdict(int)

with open(DUMP_PATH, "w", encoding="utf-8") as fdump:
    for i, out in enumerate(outputs):
        generated = out.outputs[0].text
        gold = str(items[i]["answer"])
        pred = extract_prediction(generated)
        match = answers_match(pred, gold)
        ptype = items[i].get("perturbation_type", "<none>")

        if match:
            correct += 1
            bucket_correct[ptype] += 1
        bucket_n[ptype] += 1

        fdump.write(json.dumps({
            "idx": i,
            "question": items[i]["question"],
            "gold": gold,
            "pred": pred,
            "generated": generated,
            "match": bool(match),
            "perturbation_type": ptype,
        }, ensure_ascii=False) + "\n")

        if i < 3:
            print(f"--- Example {i+1} ---")
            print(f"Q: {items[i]['question']}")
            print(f"Generated: {generated[:400]}")
            print(f"Pred: {pred} | Gold: {gold} | Match: {match} | Type: {ptype}")

accuracy = correct / total * 100
print(f"\n[{TAG}] Overall: {correct} / {total}  ({accuracy:.2f}%)")
print(f"[{TAG}] Per-category accuracy (sorted asc):\n")
print(f"{'perturbation_type':<42}{'N':>5}{'correct':>9}{'acc':>9}")
print("-" * 65)
rows = []
for ptype, n in bucket_n.items():
    c = bucket_correct[ptype]
    rows.append((ptype, n, c, c / n * 100))
rows.sort(key=lambda r: r[3])
for ptype, n, c, acc in rows:
    print(f"{ptype:<42}{n:>5}{c:>9}{acc:>8.2f}%")

print(f"\n[{TAG}] Dump written to {DUMP_PATH}")
