#!/usr/bin/env python3
"""
Minimal vLLM client using requests.
- First requests /v1/models and takes the first model id
- Then makes one chat/completions request
- Prints latency, usage, and text
- Saves the result to results/task1_sanity_check.json
"""

import time
import json
import os
import requests

BASE_URL = "http://localhost:8000"
API_KEY = "token-abc123"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}


def get_available_model():
    """Fetch /v1/models and return the first model id."""
    r = requests.get(f"{BASE_URL}/v1/models", headers=HEADERS, timeout=20)
    r.raise_for_status()
    data = r.json()
    model_id = data["data"][0]["id"]
    return model_id


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
        "raw": data,
    }


def main():
    print("=" * 60)
    print("Task 1: vLLM Service Connection Test")
    print("=" * 60)

    # Ensure results directory exists
    os.makedirs("/home/ubuntu/lab_inference/results", exist_ok=True)

    # Step 1: Get available model
    print("\n[1] Fetching /v1/models...")
    try:
        model = get_available_model()
        print(f"    Available model: {model}")
    except requests.exceptions.ConnectionError as e:
        print(f"    ERROR: Cannot connect to vLLM server at {BASE_URL}")
        print(f"    Make sure the server is running: vllm serve /home/ubuntu/models/Qwen3-0.6B --host 0.0.0.0 --port 8000 --api-key token-abc123")
        return

    # Step 2: Send chat completion request
    print("\n[2] Sending chat completion request...")
    prompt = "What is 2 + 2?"
    print(f"    Prompt: {prompt}")

    result = chat_once(prompt, model, temperature=0.0, max_tokens=64)

    print(f"\n[3] Response:")
    print(f"    Text: {result['text']}")
    print(f"    Usage: {result['usage']}")
    print(f"    Latency: {result['latency_s']:.3f} seconds")

    # Step 3: Save to file
    output_path = "/home/ubuntu/lab_inference/results/task1_sanity_check.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[4] Result saved to: {output_path}")

    print("\n" + "=" * 60)
    print("Task 1 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
