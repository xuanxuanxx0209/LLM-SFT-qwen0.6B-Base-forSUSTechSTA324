# Experiment Log: Decoding / KV Cache / vLLM

## 1. Environment Information

| Item | Value |
|------|-------|
| Date | 2026-03-27 |
| Machine | Linux 6.8.0-78-generic |
| GPU | NVIDIA GeForce RTX 5090 |
| Python version | 3.12.12 |
| PyTorch version | 2.9.1+cu130 |
| Transformers version | 4.57.6 |
| vLLM version | 0.15.0 |
| Model | Qwen3-0.6B (/home/ubuntu/models/Qwen3-0.6B) |

---

## 2. Task 1: Service Bring-up

### What I did

1. Created project directory structure: `lab_inference/scripts/` and `lab_inference/results/`
2. Started vLLM server with command:
   ```bash
   vllm serve /home/ubuntu/models/Qwen3-0.6B --dtype auto --host 0.0.0.0 --port 8000 --api-key token-abc123
   ```
3. Created `scripts/01_call_vllm.py` to test connection
4. Verified `/v1/models` endpoint returns model list
5. Sent first chat completion request and saved results

### Output Summary

```
Model: /home/ubuntu/models/Qwen3-0.6B
Prompt: What is 2 + 2?
Response: <think>
Okay, so the question is asking, "What is 2 + 2?" Hmm, let's see...
Usage: {'prompt_tokens': 16, 'total_tokens': 80, 'completion_tokens': 64}
Latency: 0.271 seconds
```

### Problems Encountered

- No major issues. Server started successfully after ~40 seconds (model loading + CUDA graph capture)
- vLLM logged: `Detected the chat template content format to be 'string'`

### How I Solved Them

- Server startup takes time due to model loading and CUDA graph capture (~19 seconds for engine init, ~5 seconds for graph capture)
- This is expected behavior for first-time startup

---

## 3. Task 2: Decoding Strategy Comparison

### Prompt Design

| Prompt Type | Prompt |
|-------------|--------|
| stable_math | "What is 15 + 27? Only output the number." |
| stable_fact | "What is the capital of France? Answer in one word." |
| open_creative | "Write a short slogan for a coffee shop." |
| open_story | "Write the opening sentence of a mystery story." |

### Configuration Table

| Config | Temperature | top_p | top_k | Note |
|--------|-------------|-------|-------|------|
| A_greedy | 0.0 | 1.0 | None | Greedy decoding |
| B_low_temp | 0.2 | 1.0 | 20 | Slightly conservative |
| C_medium | 0.7 | 1.0 | 20 | Medium randomness |
| D_high | 1.0 | 0.9 | 50 | More open-ended |

### Result Summary

**Stable Math Prompt:**
| Config | Avg Latency | Unique Outputs / 3 |
|--------|-------------|-------------------|
| A_greedy | 0.158s | 1/3 (100% stable) |
| B_low_temp | 0.308s | 3/3 (100% diverse) |
| C_medium | 0.222s | 3/3 |
| D_high | 0.194s | 3/3 |

**Open Creative Prompt:**
| Config | Avg Latency | Unique Outputs / 3 |
|--------|-------------|-------------------|
| A_greedy | 0.144s | 1/3 (100% stable) |
| B_low_temp | 0.234s | 3/3 |
| C_medium | 0.223s | 3/3 |
| D_high | 0.162s | 3/3 |

### My Observations

1. **Greedy decoding (temp=0.0) produces identical outputs every time** - For both stable and open-ended prompts, config A produced the exact same text in all 3 runs.

2. **Higher temperature increases diversity** - Configs B, C, D all produced 3 unique outputs per prompt.

3. **Greedy is fastest** - Lower latency because no sampling computation needed.

4. **Interesting pattern**: The model outputs "<think>" reasoning blocks before answering, showing internal monologue behavior.

---

## 4. Task 3: KV Cache / Shared Prefix

### Experiment Design

- **Shared prefix**: ~1929 character course description (~472 tokens)
- **4 questions** about the course material
- **3 repeats** per question
- **Control group**: Same questions without shared prefix (~17-27 tokens)

### Result Summary

| Experiment | Mean Latency | Std Dev |
|------------|--------------|---------|
| Shared prefix (472 tokens) | 0.158s | 0.009s |
| Different prefix (17-27 tokens) | 0.145s | 0.014s |

**vLLM Metrics showed:**
- `prefix_cache_queries_total`: 6885
- `prefix_cache_hits_total`: 5904
- `prefix_cache_hit_rate`: ~85.7%

### My Explanation

**Why shared prefix was slower:**
The shared prefix requests were **8.7% slower** not because KV cache didn't work, but because:
1. **Prompt length difference**: 472 tokens vs 17-27 tokens
2. Processing longer prompts requires more initial computation regardless of caching
3. KV cache benefits are most visible when the **same exact prefix** is reused across requests

**Did I observe caching benefit?**
- The metrics confirm prefix caching IS working (85%+ hit rate)
- But the experiment design had a confounding factor: different prompt lengths
- A better experiment would compare same-length prompts with vs without shared history

---

## 5. Task 4: Concurrency / Throughput / Latency

### Experiment Setup

- **Prompt**: "What is 2 + 2? Answer with just the number."
- **Total requests**: 16 per run
- **Concurrency levels**: 1, 2, 4, 8
- **Max tokens**: 64

### Result Table

| Concurrency | p50 (s) | p95 (s) | Wall Time (s) | Tokens/s | Requests/s |
|-------------|---------|---------|---------------|----------|------------|
| 1 | 0.145 | 0.179 | 2.33 | 438.8 | 6.9 |
| 2 | 0.164 | 0.189 | 1.32 | 773.9 | 12.1 |
| 4 | 0.181 | 0.190 | 0.71 | 1439.3 | 22.5 |
| 8 | 0.205 | 0.217 | 0.43 | 2379.4 | 37.2 |

### My Explanation

1. **p50/p95 latency increases with concurrency** - From 0.145s to 0.205s (p50), ~41% increase. This is because requests compete for GPU resources.

2. **Throughput dramatically improves** - Tokens/s increases from 438.8 to 2379.4 (5.4x improvement). The system processes more work in parallel.

3. **Optimal operating point depends on goal**:
   - For lowest latency: concurrency=1
   - For highest throughput: concurrency=8 (or higher)
   - Trade-off is inherent: batching improves efficiency but adds queuing delay

4. **Why throughput and latency conflict**:
   - Higher concurrency = more batching = better GPU utilization
   - But each request waits longer in queue before processing
   - This is a fundamental systems trade-off

---

## 6. Summary of Three Generation Paths

| Method | Latency | Best For |
|--------|---------|----------|
| vLLM API | 0.111s | Production serving |
| transformers.generate() | 1.343s | Local prototyping |
| manual_decode | 0.706s | Education/debugging |

**manual_decode** is best for **understanding how generation works at the token level**, but not suitable for production due to lack of KV cache optimization.

**transformers.generate()** is best for **local development and prototyping** because it's easy to use with built-in KV cache, but lacks serving features.

**vLLM** is best for **production serving and high-throughput applications**, especially with concurrent requests, thanks to PagedAttention and optimized batching.

---

## 7. Failed Attempts

| Failed Experiment | What Went Wrong | How I Revised |
|-------------------|-----------------|---------------|
| Initial decoding script | KeyError on `output_tokens` | Fixed by using `.get()` with default value |
| KV cache experiment interpretation | Initially thought cache wasn't working because shared prefix was slower | Checked vLLM metrics and realized prompt length was the confounding factor |

---

## 8. Final Conclusion

### About Decoding

I learned that **temperature=0 (greedy) produces deterministic outputs** - the same prompt always yields the same response. Higher temperatures introduce randomness, which is useful for creative tasks but harmful for factual questions. The `top-p` (nucleus sampling) is more adaptive than fixed `top-k` because it dynamically adjusts the candidate pool based on probability distribution.

### About KV Cache

KV cache is an **inference optimization that avoids recomputing past token representations**. The key insight: when generating token N, you don't want to recompute attention for tokens 0 to N-1. vLLM's metrics showed 85%+ prefix cache hit rate, confirming the cache works. The cache benefit is most visible when:
1. Same system prompt is reused across many requests
2. Long conversation history is shared
3. Multi-turn conversations build on previous context

### About vLLM

vLLM shines in **serving scenarios with concurrent requests**. Key advantages:
1. **PagedAttention**: Efficient memory management for KV cache
2. **OpenAI-compatible API**: Drop-in replacement for existing workflows
3. **Built-in metrics**: `/metrics` endpoint for monitoring
4. **CUDA graph capture**: Reduces kernel launch overhead

The benchmark showed 5.4x throughput improvement from concurrency=1 to concurrency=8.

### If I Were to Give 3 Suggestions to the Next Class

1. **Start the server early** - Model loading and CUDA graph capture take ~40 seconds. Start it before you begin coding.

2. **Check vLLM metrics** - The `/metrics` endpoint reveals what's actually happening (cache hits, queue lengths, throughput).

3. **Design experiments carefully** - My KV cache experiment had a confounding variable (prompt length). Control your variables!

---

## Appendix: Files Created

```
lab_inference/
├── README.md
├── scripts/
│   ├── 00_check_env.py
│   ├── 01_call_vllm.py
│   ├── 02_manual_decode.py
│   ├── 03_decoding_compare.py
│   ├── 04_prefix_cache_test.py
│   ├── 05_benchmark.py
│   └── 06_three_paths_compare.py
├── results/
│   ├── task1_sanity_check.json
│   ├── decoding_compare.csv
│   ├── prefix_cache_results.csv
│   └── concurrency_results.csv
└── exp_log.md
```
