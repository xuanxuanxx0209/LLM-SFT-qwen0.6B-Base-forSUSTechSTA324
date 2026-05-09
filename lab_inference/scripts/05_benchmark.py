#!/usr/bin/env python3
"""
Concurrency / Throughput / Latency Benchmark.

This script runs benchmarks at different concurrency levels to measure:
- p50 latency
- p95 latency
- Wall clock time
- Estimated tokens/second

Experiment A: Concurrency sweep (1, 2, 4, 8)
Experiment B: Workload comparison (short answers vs long reasoning)
"""

import time
import csv
import os
import requests
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "http://localhost:8000"
API_KEY = "token-abc123"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# Prompts
SHORT_PROMPT = "What is 2 + 2? Answer with just the number."
LONG_PROMPT = "Explain the difference between arrays and linked lists, including their time complexities for various operations. Provide examples of when to use each."

CONCURRENCY_LEVELS = [1, 2, 4, 8]
N_REQUESTS = 16
MAX_TOKENS = 64


def chat_once(prompt, model, temperature=0.0, max_tokens=MAX_TOKENS):
    """Send a single chat completion request."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    t0 = time.perf_counter()
    r = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        headers=HEADERS,
        json=payload,
        timeout=300,
    )
    r.raise_for_status()
    data = r.json()
    dt = time.perf_counter() - t0

    return {
        "text": data["choices"][0]["message"]["content"],
        "prompt_tokens": data.get("usage", {}).get("prompt_tokens", 0),
        "completion_tokens": data.get("usage", {}).get("completion_tokens", 0),
        "latency_s": dt,
    }


def get_available_model():
    """Fetch /v1/models and return the first model id."""
    r = requests.get(f"{BASE_URL}/v1/models", headers=HEADERS, timeout=20)
    r.raise_for_status()
    data = r.json()
    return data["data"][0]["id"]


def pct(xs, p):
    """Calculate percentile."""
    if not xs:
        return None
    k = int(round((len(xs) - 1) * p / 100))
    return xs[k]


def run_batch_concurrent(call_fn, n_requests=16, concurrency=4):
    """Run a batch of requests concurrently."""
    t0 = time.perf_counter()
    rows = []

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(call_fn, i) for i in range(n_requests)]
        for fut in as_completed(futures):
            rows.append(fut.result())

    wall_s = time.perf_counter() - t0

    lats = [r["latency_s"] for r in rows]
    lats_sorted = sorted(lats)
    total_tokens = sum(r["completion_tokens"] for r in rows)

    return {
        "wall_s": wall_s,
        "p50_s": pct(lats_sorted, 50),
        "p95_s": pct(lats_sorted, 95),
        "mean_s": statistics.mean(lats) if lats else 0,
        "std_s": statistics.stdev(lats) if len(lats) > 1 else 0,
        "min_s": min(lats) if lats else 0,
        "max_s": max(lats) if lats else 0,
        "total_tokens": total_tokens,
        "tokens_per_second": total_tokens / wall_s if wall_s > 0 else 0,
        "requests_per_second": n_requests / wall_s if wall_s > 0 else 0,
        "rows": rows,
    }


def fetch_metrics():
    """Fetch vLLM metrics."""
    try:
        r = requests.get(f"{BASE_URL}/metrics", timeout=20)
        if r.status_code == 200:
            return r.text
    except:
        pass
    return None


def benchmark_concurrency():
    """Experiment A: Concurrency sweep."""
    print("=" * 60)
    print("Task 4: Concurrency / Throughput Benchmark")
    print("=" * 60)

    os.makedirs("/home/ubuntu/lab_inference/results", exist_ok=True)

    model = get_available_model()
    print(f"Model: {model}")
    print(f"Prompt: {SHORT_PROMPT}")
    print(f"Total requests: {N_REQUESTS}")
    print(f"Concurrency levels: {CONCURRENCY_LEVELS}")
    print()

    results = []

    for concurrency in CONCURRENCY_LEVELS:
        print(f"\n--- Concurrency: {concurrency} ---")

        def make_request(_):
            return chat_once(SHORT_PROMPT, model, temperature=0.0, max_tokens=MAX_TOKENS)

        stats = run_batch_concurrent(make_request, n_requests=N_REQUESTS, concurrency=concurrency)

        print(f"  Wall time: {stats['wall_s']:.2f}s")
        print(f"  p50 latency: {stats['p50_s']:.3f}s")
        print(f"  p95 latency: {stats['p95_s']:.3f}s")
        print(f"  Mean latency: {stats['mean_s']:.3f}s (+/- {stats['std_s']:.3f}s)")
        print(f"  Tokens/s: {stats['tokens_per_second']:.1f}")
        print(f"  Requests/s: {stats['requests_per_second']:.1f}")

        results.append({
            "experiment": "concurrency_sweep",
            "concurrency": concurrency,
            "n_requests": N_REQUESTS,
            "wall_s": round(stats["wall_s"], 4),
            "p50_s": round(stats["p50_s"], 4),
            "p95_s": round(stats["p95_s"], 4),
            "mean_s": round(stats["mean_s"], 4),
            "std_s": round(stats["std_s"], 4),
            "min_s": round(stats["min_s"], 4),
            "max_s": round(stats["max_s"], 4),
            "total_tokens": stats["total_tokens"],
            "tokens_per_second": round(stats["tokens_per_second"], 2),
            "requests_per_second": round(stats["requests_per_second"], 2),
        })

    # Write to CSV
    csv_path = "/home/ubuntu/lab_inference/results/concurrency_results.csv"
    fieldnames = ["experiment", "concurrency", "n_requests", "wall_s", "p50_s", "p95_s", "mean_s", "std_s", "min_s", "max_s", "total_tokens", "tokens_per_second", "requests_per_second"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {csv_path}")

    return results


def benchmark_workload():
    """Experiment B: Workload comparison (short vs long)."""
    print()
    print("=" * 60)
    print("Experiment B: Workload Comparison")
    print("=" * 60)

    model = get_available_model()
    concurrency = 4
    n_requests = 8

    workloads = [
        ("short_answer", SHORT_PROMPT),
        ("long_reasoning", LONG_PROMPT),
    ]

    results = []

    for workload_name, prompt in workloads:
        print(f"\n--- Workload: {workload_name} ---")
        print(f"Prompt: {prompt[:60]}...")

        def make_request(_):
            return chat_once(prompt, model, temperature=0.0, max_tokens=MAX_TOKENS)

        stats = run_batch_concurrent(make_request, n_requests=n_requests, concurrency=concurrency)

        print(f"  Wall time: {stats['wall_s']:.2f}s")
        print(f"  p50 latency: {stats['p50_s']:.3f}s")
        print(f"  p95 latency: {stats['p95_s']:.3f}s")
        print(f"  Mean latency: {stats['mean_s']:.3f}s")
        print(f"  Avg tokens: {statistics.mean([r['completion_tokens'] for r in stats['rows']]):.1f}")
        print(f"  Tokens/s: {stats['tokens_per_second']:.1f}")

        results.append({
            "experiment": "workload_comparison",
            "workload": workload_name,
            "concurrency": concurrency,
            "n_requests": n_requests,
            "wall_s": round(stats["wall_s"], 4),
            "p50_s": round(stats["p50_s"], 4),
            "p95_s": round(stats["p95_s"], 4),
            "mean_s": round(stats["mean_s"], 4),
            "avg_completion_tokens": round(statistics.mean([r["completion_tokens"] for r in stats["rows"]]), 1),
            "tokens_per_second": round(stats["tokens_per_second"], 2),
        })

    print()
    print("=" * 60)
    print("Task 4 Complete!")
    print("=" * 60)

    print("\nKey Questions to Answer:")
    print("1. How do p50/p95 latencies change as concurrency increases?")
    print("2. What is the optimal concurrency level for throughput?")
    print("3. Which workload (short vs long) is more 'system-friendly'? Why?")
    print("4. Why do throughput and single-request latency often conflict?")

    return results


def main():
    results_a = benchmark_concurrency()
    results_b = benchmark_workload()

    # Print summary table
    print()
    print("=" * 60)
    print("Summary Table: Concurrency Sweep")
    print("=" * 60)
    print(f"{'Concurrency':<12} {'p50(s)':<10} {'p95(s)':<10} {'Wall(s)':<10} {'Tok/s':<10} {'Req/s':<10}")
    print("-" * 62)
    for r in results_a:
        print(f"{r['concurrency']:<12} {r['p50_s']:<10.3f} {r['p95_s']:<10.3f} {r['wall_s']:<10.2f} {r['tokens_per_second']:<10.1f} {r['requests_per_second']:<10.1f}")

    print()
    print("=" * 60)
    print("Summary Table: Workload Comparison")
    print("=" * 60)
    print(f"{'Workload':<20} {'p50(s)':<12} {'p95(s)':<12} {'Tok/s':<12} {'Avg Tokens':<12}")
    print("-" * 68)
    for r in results_b:
        print(f"{r['workload']:<20} {r['p50_s']:<12.3f} {r['p95_s']:<12.3f} {r['tokens_per_second']:<12.1f} {r['avg_completion_tokens']:<12.1f}")


if __name__ == "__main__":
    main()
