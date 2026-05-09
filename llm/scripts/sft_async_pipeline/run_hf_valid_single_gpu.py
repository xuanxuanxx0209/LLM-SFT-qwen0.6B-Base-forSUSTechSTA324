#!/usr/bin/env python3
"""Run valid-set generation on a single GPU with Transformers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = (
    "You are a careful math solver. Solve the problem step by step. "
    "End your response with a final line in the format: #### <answer>"
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


def get_model_max_len(model_path: Path) -> int:
    config_path = model_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return int(config.get("max_position_embeddings", 4096))


def maybe_override_chat_template(tokenizer, chat_template_file: str | None) -> None:
    if chat_template_file is None:
        return
    tokenizer.chat_template = Path(chat_template_file).read_text(encoding="utf-8")


def build_prompts(records: list[dict], tokenizer) -> tuple[list[dict], int]:
    prompt_token_lengths = []
    updated = []
    for record in records:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": record["question"]},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        item = dict(record)
        item["prompt"] = prompt
        item["prompt_token_length"] = len(prompt_ids)
        updated.append(item)
        prompt_token_lengths.append(len(prompt_ids))
    return updated, max(prompt_token_lengths)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--device-id", default="0")
    parser.add_argument("--chat-template-file", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--requested-max-new-tokens", type=int, default=4096)
    parser.add_argument("--attn-implementation", default="eager")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(input_path)
    records = records[: args.limit]
    if not records:
        raise ValueError("No records found for inference.")

    tokenizer_path = Path(args.tokenizer_path) if args.tokenizer_path is not None else model_path
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)
    maybe_override_chat_template(tokenizer, args.chat_template_file)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    prepared_records, max_prompt_tokens = build_prompts(records, tokenizer)
    model_max_len = get_model_max_len(model_path)
    prompt_buffer = 32
    effective_max_new_tokens = min(args.requested_max_new_tokens, model_max_len - max_prompt_tokens - prompt_buffer)
    if effective_max_new_tokens <= 0:
        raise ValueError(
            f"No room left for generation: model_max_len={model_max_len}, "
            f"max_prompt_tokens={max_prompt_tokens}, prompt_buffer={prompt_buffer}"
        )

    torch.cuda.set_device(f"cuda:{args.device_id}")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    )
    model.to("cuda")
    model.eval()

    outputs = []
    for start in range(0, len(prepared_records), args.batch_size):
        batch_records = prepared_records[start : start + args.batch_size]
        batch_prompts = [record["prompt"] for record in batch_records]
        encoded = tokenizer(
            batch_prompts,
            add_special_tokens=False,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        encoded = {key: value.to("cuda") for key, value in encoded.items()}

        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=effective_max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        prompt_width = encoded["input_ids"].shape[1]
        generated_texts = tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True)

        for record, text in zip(batch_records, generated_texts):
            outputs.append(
                {
                    "source_index": record["source_index"],
                    "split": record.get("split", "valid"),
                    "question": record["question"],
                    "answer": record["answer"],
                    "solution": record.get("solution"),
                    "perturbation_type": record.get("perturbation_type"),
                    "seed_question": record.get("seed_question"),
                    "seed_solution": record.get("seed_solution"),
                    "seed_answer": record.get("seed_answer"),
                    "model_name": model_path.name,
                    "worker_index": 0,
                    "prompt_token_length": record.get("prompt_token_length"),
                    "generated_text": text,
                }
            )

    output_path = output_dir / f"{model_path.name}.valid_1000.outputs.jsonl"
    write_jsonl(output_path, outputs)

    metadata = {
        "engine": "transformers",
        "model_name": model_path.name,
        "model_path": str(model_path),
        "tokenizer_path": str(tokenizer_path),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "num_records": len(outputs),
        "batch_size": args.batch_size,
        "device_id": args.device_id,
        "model_max_len": model_max_len,
        "max_prompt_tokens": max_prompt_tokens,
        "requested_max_new_tokens": args.requested_max_new_tokens,
        "effective_max_new_tokens": effective_max_new_tokens,
        "chat_template_file": args.chat_template_file,
        "attn_implementation": args.attn_implementation,
    }
    metadata_path = output_dir / f"{model_path.name}.valid_1000.metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
