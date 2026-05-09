#!/usr/bin/env python3
"""
Task 5: Compare three generation paths:
1. manual_decode (hand-written autoregressive loop)
2. transformers.generate() (built-in generation)
3. vLLM API (serving framework)
"""

import time
import os
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

BASE_URL = "http://localhost:8000"
API_KEY = "token-abc123"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

MODEL_PATH = "/home/ubuntu/models/Qwen3-0.6B"
PROMPT = "What is the capital of France? Answer briefly."
MAX_NEW_TOKENS = 30


def vllm_chat(prompt, temperature=0.0, max_tokens=MAX_NEW_TOKENS):
    """Call vLLM API."""
    payload = {
        "model": MODEL_PATH,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    t0 = time.perf_counter()
    r = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        headers=HEADERS,
        json=payload,
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    dt = time.perf_counter() - t0

    return {
        "text": data["choices"][0]["message"]["content"],
        "latency_s": dt,
        "method": "vLLM API",
    }


def transformers_generate(prompt, model, tokenizer, device, temperature=0.0, max_tokens=MAX_NEW_TOKENS):
    """Use transformers generate() method."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
    dt = time.perf_counter() - t0

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {
        "text": text,
        "latency_s": dt,
        "method": "transformers.generate()",
    }


def manual_decode(prompt, model, tokenizer, device, max_tokens=MAX_NEW_TOKENS):
    """Hand-written autoregressive decoding."""
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    generated = input_ids.clone()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(max_tokens):
            outputs = model(input_ids=generated)
            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    dt = time.perf_counter() - t0

    text = tokenizer.decode(generated[0], skip_special_tokens=True)

    return {
        "text": text,
        "latency_s": dt,
        "method": "manual_decode",
    }


def main():
    print("=" * 60)
    print("Task 5: Compare Three Generation Paths")
    print("=" * 60)
    print(f"Prompt: {PROMPT}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print()

    # Load model for local methods
    print("Loading model for local methods...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    print(f"Model loaded on {device}")
    print()

    results = []

    # Method 1: vLLM API
    print("--- Method 1: vLLM API ---")
    result = vllm_chat(PROMPT, temperature=0.0)
    results.append(result)
    print(f"Text: {result['text'][:100]}...")
    print(f"Latency: {result['latency_s']:.3f}s")
    print()

    # Method 2: transformers.generate()
    print("--- Method 2: transformers.generate() ---")
    result = transformers_generate(PROMPT, model, tokenizer, device, temperature=0.0)
    results.append(result)
    print(f"Text: {result['text'][:100]}...")
    print(f"Latency: {result['latency_s']:.3f}s")
    print()

    # Method 3: manual_decode
    print("--- Method 3: manual_decode (hand-written) ---")
    result = manual_decode(PROMPT, model, tokenizer, device, max_tokens=30)
    results.append(result)
    print(f"Text: {result['text'][:100]}...")
    print(f"Latency: {result['latency_s']:.3f}s")
    print()

    # Summary comparison
    print("=" * 60)
    print("Summary Comparison")
    print("=" * 60)
    print(f"{'Method':<30} {'Latency (s)':<15}")
    print("-" * 45)
    for r in results:
        print(f"{r['method']:<30} {r['latency_s']:<15.3f}")

    # Analysis
    print()
    print("=" * 60)
    print("Analysis: Three Generation Paths")
    print("=" * 60)

    print("""
1. manual_decode (hand-written autoregressive loop):
   - BEST FOR: Understanding how generation works at the lowest level
   - PROS: Complete transparency, educational value, full control
   - CONS: Inefficient (recomputes all past tokens each step), no KV cache,
           no batching, not suitable for production
   - USE WHEN: Learning, debugging, research experiments

2. transformers.generate() (built-in generation):
   - BEST FOR: Local development, prototyping, single-user applications
   - PROS: Easy to use, many built-in features (beam search, sampling, etc.),
           KV cache support, well-tested
   - CONS: Not optimized for high-throughput serving, limited concurrency,
           no advanced serving features
   - USE WHEN: Building local tools, experimenting, small-scale deployments

3. vLLM API (serving framework):
   - BEST FOR: Production serving, high-throughput applications
   - PROS: Optimized KV cache management (PagedAttention), concurrent request
           handling, OpenAI-compatible API, monitoring/metrics, batching
   - CONS: More complex setup, requires running a server
   - USE WHEN: Building APIs, serving multiple users, benchmarking

Key Takeaway:
- manual_decode teaches you HOW generation works
- transformers.generate() is for USING generation locally
- vLLM is for SERVING generation at scale
""")

    print("=" * 60)
    print("Task 5 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
