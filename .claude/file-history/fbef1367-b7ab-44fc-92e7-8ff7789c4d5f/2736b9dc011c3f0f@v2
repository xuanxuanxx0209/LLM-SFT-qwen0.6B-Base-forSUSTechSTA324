#!/usr/bin/env python3
"""
Hand-written manual decoder for educational purposes.
Shows how autoregressive generation works token by token.
"""

import time
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class DecodeConfig:
    max_new_tokens: int = 50
    strategy: str = "greedy"  # "greedy" or "sample"
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    stop_on_eos: bool = True


def top_k_filter(logits, top_k):
    """Filter logits to keep only top-k tokens."""
    if top_k <= 0:
        return logits
    values, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
    cutoff = values[..., -1, None]
    return torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)


def top_p_filter(logits, top_p):
    """Filter logits using nucleus sampling (top-p)."""
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    remove_mask = cumulative_probs > top_p
    remove_mask[..., 1:] = remove_mask[..., :-1].clone()
    remove_mask[..., 0] = False
    filtered_sorted = sorted_logits.masked_fill(remove_mask, float("-inf"))
    filtered_logits = torch.full_like(logits, float("-inf"))
    filtered_logits.scatter_(dim=-1, index=sorted_indices, src=filtered_sorted)
    return filtered_logits


def select_next_token(logits, cfg: DecodeConfig):
    """Select next token based on decoding strategy."""
    if cfg.strategy == "greedy":
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / cfg.temperature
    logits = top_k_filter(logits, cfg.top_k)
    logits = top_p_filter(logits, cfg.top_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def manual_decode(model, tokenizer, prompt, cfg: DecodeConfig, device="cuda"):
    """
    Manual autoregressive decoding.

    This is intentionally simple and inefficient to show how generation works.
    At each step, we pass ALL generated tokens so far, which is wasteful.
    KV cache would avoid recomputing past token representations.
    """
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    generated = input_ids.clone()
    new_ids = []

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(cfg.max_new_tokens):
            # Forward pass through all tokens (inefficient without KV cache!)
            outputs = model(input_ids=generated)
            logits = outputs.logits[:, -1, :]  # Get logits for last position
            next_token = select_next_token(logits, cfg)
            token_id = next_token.item()
            new_ids.append(token_id)
            generated = torch.cat([generated, next_token], dim=-1)
            if cfg.stop_on_eos and token_id == tokenizer.eos_token_id:
                break
    dt = time.perf_counter() - t0

    return {
        "full_text": tokenizer.decode(generated[0], skip_special_tokens=True),
        "num_new_tokens": len(new_ids),
        "latency_s": dt,
    }


def main():
    print("=" * 60)
    print("Task 2/5: Manual Decoder Demo")
    print("=" * 60)

    # Load model
    model_path = "/home/ubuntu/models/Qwen3-0.6B"
    print(f"Loading model from {model_path}...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    print(f"Model loaded on {device}")

    # Test prompts
    prompts = [
        ("Math (greedy)", "What is 2 + 2?", DecodeConfig(max_new_tokens=20, strategy="greedy")),
        ("Math (sample t=0.7)", "What is 2 + 2?", DecodeConfig(max_new_tokens=20, strategy="sample", temperature=0.7)),
        ("Creative (greedy)", "Write a short slogan", DecodeConfig(max_new_tokens=30, strategy="greedy")),
        ("Creative (sample)", "Write a short slogan", DecodeConfig(max_new_tokens=30, strategy="sample", temperature=0.8, top_k=20, top_p=0.9)),
    ]

    results = []
    for name, prompt, cfg in prompts:
        print(f"\n--- {name} ---")
        print(f"Prompt: {prompt}")
        print(f"Config: strategy={cfg.strategy}, temp={cfg.temperature}, top_k={cfg.top_k}, top_p={cfg.top_p}")

        result = manual_decode(model, tokenizer, prompt, cfg, device)
        results.append((name, result))

        print(f"Generated ({result['num_new_tokens']} tokens in {result['latency_s']:.3f}s):")
        print(f"  {result['full_text']}")

    print("\n" + "=" * 60)
    print("Comparison of generation paths:")
    print("=" * 60)
    for name, result in results:
        print(f"{name}: {result['num_new_tokens']} tokens, {result['latency_s']:.3f}s, {result['num_new_tokens']/result['latency_s']:.1f} tok/s")

    print("\n" + "=" * 60)
    print("Manual Decode Demo Complete!")
    print("=" * 60)
    print("\nKey observations:")
    print("- This implementation is INEFFICIENT: it recomputes all past tokens at each step")
    print("- KV cache would store past key/value states to avoid redundant computation")
    print("- This is why vLLM and other serving frameworks use KV cache")


if __name__ == "__main__":
    main()
