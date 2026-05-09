#!/usr/bin/env python3
"""
KV Cache / Shared Prefix Experiment.

This script tests whether sharing a long prefix across multiple requests
provides any performance benefit, which would indicate KV cache effectiveness.

We construct a long shared prefix (course material) and then ask different
questions about it. We compare:
1. Requests sharing the same long prefix
2. Requests with different prefixes but similar length
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


# Long shared prefix: course description (simulating a long system prompt or context)
SHARED_PREFIX = """
You are a teaching assistant for an introductory computer science course (CS101).
Below is the complete course material for Week 4 on "Algorithms and Data Structures":

COURSE MATERIAL - WEEK 4: ALGORITHMS AND DATA STRUCTURES

Introduction:
This week covers fundamental concepts in computer science including sorting algorithms,
searching algorithms, and basic data structures. Students will learn about time complexity,
space complexity, and how to choose appropriate algorithms for different problems.

Section 1: Sorting Algorithms
- Bubble Sort: Simple but inefficient O(n^2) algorithm
- Selection Sort: Also O(n^2) but with fewer swaps
- Insertion Sort: O(n^2) but efficient for nearly sorted data
- Merge Sort: Divide and conquer, O(n log n), stable
- Quick Sort: Divide and conquer, average O(n log n), worst case O(n^2)
- Heap Sort: O(n log n), in-place, not stable

Section 2: Searching Algorithms
- Linear Search: O(n), works on any list
- Binary Search: O(log n), requires sorted list
- Hash Table Lookup: O(1) average case

Section 3: Data Structures
- Arrays: Contiguous memory, O(1) access, fixed size
- Linked Lists: Dynamic size, O(n) access, efficient insertions
- Stacks: LIFO, push/pop operations
- Queues: FIFO, enqueue/dequeue operations
- Trees: Hierarchical structure, binary search trees
- Graphs: Nodes and edges, traversal algorithms (BFS, DFS)
- Hash Tables: Key-value pairs, constant time operations

Section 4: Time Complexity Analysis
- Big O notation describes upper bounds
- Common complexities: O(1), O(log n), O(n), O(n log n), O(n^2), O(2^n)
- Best case, average case, and worst case analysis

Section 5: Practical Considerations
- Cache efficiency matters in real systems
- Memory locality affects performance
- Trade-offs between time and space

This course material is designed for beginners with no prior programming experience.
The goal is to build intuition and practical understanding.
"""

# Questions to ask about the shared material
QUESTIONS = [
    "What sorting algorithm is best for nearly sorted data?",
    "What is the time complexity of binary search?",
    "Explain the difference between a stack and a queue.",
    "What data structure provides O(1) average lookup?",
]

# Different prompts without shared prefix (control group)
DIFFERENT_PROMPTS = [
    "In a computer science course, what sorting algorithm works best for nearly sorted lists? Answer briefly.",
    "What is the computational complexity of searching in a sorted list using binary search?",
    "Compare and contrast stacks and queues in programming.",
    "Which programming data structure allows constant-time key-value retrieval?",
]


def chat_once(prompt, model, temperature=0.0, max_tokens=64):
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


def fetch_metrics():
    """Fetch vLLM metrics to check prefix cache hit rate."""
    try:
        r = requests.get(f"{BASE_URL}/metrics", timeout=20)
        if r.status_code == 200:
            return r.text
    except:
        pass
    return None


def parse_prefix_cache_hit_rate(metrics_text):
    """Parse prefix cache hit rate from metrics."""
    if not metrics_text:
        return None
    for line in metrics_text.splitlines():
        if "prefix_cache_hit" in line or "cache_hit" in line.lower():
            try:
                # Try to extract numeric value
                parts = line.strip().split()
                if len(parts) >= 2:
                    return float(parts[-1])
            except:
                pass
    return None


def run_experiment():
    """Run KV cache / shared prefix experiment."""
    print("=" * 60)
    print("Task 3: KV Cache / Shared Prefix Experiment")
    print("=" * 60)

    os.makedirs("/home/ubuntu/lab_inference/results", exist_ok=True)

    model = get_available_model()
    print(f"Model: {model}")
    print()

    results = []
    repeat = 3

    # Experiment A: Shared prefix requests
    print("=" * 60)
    print("Experiment A: Requests with SHARED long prefix")
    print("=" * 60)
    print(f"Prefix length: ~{len(SHARED_PREFIX)} characters")
    print()

    # First, send all shared prefix requests to potentially warm up cache
    shared_prefix_latencies = []
    for i, question in enumerate(QUESTIONS):
        full_prompt = SHARED_PREFIX + "\n\nQuestion: " + question
        print(f"Request {i+1}: {question[:50]}...")

        latencies_for_this = []
        for r in range(repeat):
            result = chat_once(full_prompt, model, temperature=0.0, max_tokens=64)
            latency = result["latency_s"]
            latencies_for_this.append(latency)
            print(f"  Run {r+1}: {latency:.3f}s, {result['usage'].get('prompt_tokens', 0)} prompt tokens")

            results.append({
                "experiment": "shared_prefix",
                "question_id": i + 1,
                "repeat": r + 1,
                "prompt_tokens": result["usage"].get("prompt_tokens", 0),
                "completion_tokens": result["usage"].get("completion_tokens", 0),
                "latency_s": round(latency, 4),
                "text": result["text"][:100].replace("\n", " "),
            })

        shared_prefix_latencies.extend(latencies_for_this)

    print()

    # Experiment B: Different prompts (control group)
    print("=" * 60)
    print("Experiment B: Requests with DIFFERENT prefixes (control)")
    print("=" * 60)

    different_latencies = []
    for i, prompt in enumerate(DIFFERENT_PROMPTS):
        print(f"Request {i+1}: {prompt[:50]}...")

        latencies_for_this = []
        for r in range(repeat):
            result = chat_once(prompt, model, temperature=0.0, max_tokens=64)
            latency = result["latency_s"]
            latencies_for_this.append(latency)
            print(f"  Run {r+1}: {latency:.3f}s, {result['usage'].get('prompt_tokens', 0)} prompt tokens")

            results.append({
                "experiment": "different_prefix",
                "question_id": i + 1,
                "repeat": r + 1,
                "prompt_tokens": result["usage"].get("prompt_tokens", 0),
                "completion_tokens": result["usage"].get("completion_tokens", 0),
                "latency_s": round(latency, 4),
                "text": result["text"][:100].replace("\n", " "),
            })

        different_latencies.extend(latencies_for_this)

    print()

    # Check metrics for prefix cache info
    print("=" * 60)
    print("vLLM Metrics Check")
    print("=" * 60)
    metrics_text = fetch_metrics()
    if metrics_text:
        # Look for relevant cache metrics
        for line in metrics_text.splitlines():
            if "cache" in line.lower() or "prefix" in line.lower():
                if not line.startswith("#"):
                    print(line[:200])
    else:
        print("Could not fetch metrics")

    print()

    # Statistics
    print("=" * 60)
    print("Results Summary")
    print("=" * 60)

    # Write to CSV
    csv_path = "/home/ubuntu/lab_inference/results/prefix_cache_results.csv"
    fieldnames = ["experiment", "question_id", "repeat", "prompt_tokens", "completion_tokens", "latency_s", "text"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to: {csv_path}")

    # Calculate statistics
    shared_avg = statistics.mean(shared_prefix_latencies)
    different_avg = statistics.mean(different_latencies)

    print(f"\nShared prefix requests:")
    print(f"  Count: {len(shared_prefix_latencies)}")
    print(f"  Mean latency: {shared_avg:.3f}s")
    print(f"  Std dev: {statistics.stdev(shared_prefix_latencies):.3f}s" if len(shared_prefix_latencies) > 1 else "")

    print(f"\nDifferent prefix requests:")
    print(f"  Count: {len(different_latencies)}")
    print(f"  Mean latency: {different_avg:.3f}s")
    print(f"  Std dev: {statistics.stdev(different_latencies):.3f}s" if len(different_latencies) > 1 else "")

    # Comparison
    print()
    if shared_avg < different_avg:
        improvement = (different_avg - shared_avg) / different_avg * 100
        print(f"Shared prefix is FASTER by {improvement:.1f}%")
    else:
        slowdown = (shared_avg - different_avg) / different_avg * 100
        print(f"Shared prefix is SLOWER by {slowdown:.1f}%")
        print("(This may be due to longer prompt length requiring more processing)")

    print()
    print("=" * 60)
    print("Task 3 Complete!")
    print("=" * 60)

    print("\nKey Questions to Answer:")
    print("1. Did shared prefix requests show any speedup?")
    print("2. If not, why might that be? (Consider: vLLM version, cache settings, prompt structure)")
    print("3. How does KV cache theoretically help? (Avoiding recomputation of past tokens)")

    return results


if __name__ == "__main__":
    run_experiment()
