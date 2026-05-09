#!/usr/bin/env python3
"""
Decoding strategy comparison script.
Compare greedy, sampling, top-k, top-p across different prompts.
"""

import time
import csv
import os
import requests
import statistics

BASE_URL = "http://localhost:8000"
API_KEY = "token-abc123"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}


# Prompts: one stable task, one open-ended task
PROMPTS = {
    "stable_math": "What is 15 + 27? Only output the number.",
    "stable_fact": "What is the capital of France? Answer in one word.",
    "open_creative": "Write a short slogan for a coffee shop.",
    "open_story": "Write the opening sentence of a mystery story.",
}


# Configurations to compare
CONFIGS = [
    {"name": "A_greedy", "temperature": 0.0, "top_p": 1.0, "top_k": None},
    {"name": "B_low_temp", "temperature": 0.2, "top_p": 1.0, "top_k": 20},
    {"name": "C_medium", "temperature": 0.7, "top_p": 1.0, "top_k": 20},
    {"name": "D_high", "temperature": 1.0, "top_p": 0.9, "top_k": 50},
]

REPEATS = 3
MAX_TOKENS = 64


def chat_once(prompt, model, temperature=0.0, top_p=1.0, top_k=None, max_tokens=128):
    """Send a single chat completion request."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if top_k is not None:
        payload["top_k"] = top_k

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
        "usage": data.get("usage", {}),
        "latency_s": dt,
    }


def get_available_model():
    """Fetch /v1/models and return the first model id."""
    r = requests.get(f"{BASE_URL}/v1/models", headers=HEADERS, timeout=20)
    r.raise_for_status()
    data = r.json()
    return data["data"][0]["id"]


def run_experiments():
    """Run all decoding strategy experiments."""
    print("=" * 60)
    print("Task 2: Decoding Strategy Comparison")
    print("=" * 60)

    os.makedirs("/home/ubuntu/lab_inference/results", exist_ok=True)

    model = get_available_model()
    print(f"Model: {model}")
    print(f"Configs: {[c['name'] for c in CONFIGS]}")
    print(f"Prompts: {list(PROMPTS.keys())}")
    print(f"Repeats: {REPEATS}")
    print()

    results = []

    for prompt_name, prompt in PROMPTS.items():
        print(f"\n--- Prompt: {prompt_name} ---")
        print(f"Text: {prompt}")

        for config in CONFIGS:
            print(f"\n  Config {config['name']}: temp={config['temperature']}, top_p={config['top_p']}, top_k={config['top_k']}")

            for i in range(REPEATS):
                result = chat_once(
                    prompt, model,
                    temperature=config["temperature"],
                    top_p=config["top_p"],
                    top_k=config["top_k"],
                    max_tokens=MAX_TOKENS,
                )

                row = {
                    "prompt_name": prompt_name,
                    "prompt": prompt,
                    "config": config["name"],
                    "temperature": config["temperature"],
                    "top_p": config["top_p"],
                    "top_k": config["top_k"] if config["top_k"] else "",
                    "repeat": i + 1,
                    "latency_s": round(result["latency_s"], 4),
                    "output_tokens": result.get("usage", {}).get("completion_tokens", 0),
                    "text": result["text"].replace("\n", "\\n"),
                }
                results.append(row)

                print(f"    Run {i+1}: {result['latency_s']:.3f}s, {row['output_tokens']} tokens")

    # Write to CSV
    csv_path = "/home/ubuntu/lab_inference/results/decoding_compare.csv"
    fieldnames = ["prompt_name", "prompt", "config", "temperature", "top_p", "top_k", "repeat", "latency_s", "output_tokens", "text"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {csv_path}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    for prompt_name in PROMPTS.keys():
        print(f"\n{prompt_name}:")
        for config in CONFIGS:
            rows = [r for r in results if r["prompt_name"] == prompt_name and r["config"] == config["name"]]
            if rows:
                latencies = [r["latency_s"] for r in rows]
                texts = [r["text"] for r in rows]
                unique_texts = len(set(texts))
                avg_latency = statistics.mean(latencies)
                avg_tokens = statistics.mean([r["output_tokens"] for r in rows])
                print(f"  {config['name']}: avg_latency={avg_latency:.3f}s, avg_tokens={avg_tokens:.1f}, unique_outputs={unique_texts}/{REPEATS}")

    print("\n" + "=" * 60)
    print("Task 2 Complete!")
    print("=" * 60)

    return results


if __name__ == "__main__":
    run_experiments()
