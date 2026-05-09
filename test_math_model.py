#!/usr/bin/env python3
"""Quick inference test for the math SFT model."""

import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/Qwen3-0.6B-Math-SFT"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto",
)

SYSTEM_PROMPT = (
    "You are a helpful math assistant. When solving math problems, "
    "first think step by step inside <think> tags, then provide your final answer in \\boxed{}."
)

questions = [
    "What is 17 minus 9?",
    "What is the sum of the first 5 positive integers?",
    "Solve for x: 2x + 5 = 15",
]

for q in questions:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": q},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    attention_mask = (inputs != tokenizer.pad_token_id).long()
    inputs = inputs.to(model.device)
    attention_mask = attention_mask.to(model.device)

    outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_new_tokens=256,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    print(f"Q: {q}")
    print(f"A: {response.strip()}")
    print("-" * 40)
