# Week 4 Lab: vLLM Inference

This lab explores the complete pipeline: **logits → decoding → autoregressive generation → KV cache → vLLM serving → benchmarking → experiment log**

## Project Structure

```
lab_inference/
├─ README.md
├─ scripts/
│  ├─ 00_check_env.py          # Environment check
│  ├─ 01_call_vllm.py          # Minimal vLLM client
│  ├─ 02_manual_decode.py      # Hand-written decoder
│  ├─ 03_decoding_compare.py   # Decoding strategy comparison
│  ├─ 04_prefix_cache_test.py  # KV cache / shared prefix test
│  └─ 05_benchmark.py          # Concurrency / throughput benchmark
├─ results/
│  ├─ decoding_compare.csv
│  ├─ prefix_cache_results.csv
│  ├─ concurrency_results.csv
│  └─ notes.md
└─ exp_log.md
```

## Scripts Overview

| Script | Purpose |
|--------|---------|
| `00_check_env.py` | Verify environment setup (GPU, packages) |
| `01_call_vllm.py` | Start vLLM server, complete first inference |
| `02_manual_decode.py` | Manual token-by-token decoding |
| `03_decoding_compare.py` | Compare greedy, sampling, top-k, top-p |
| `04_prefix_cache_test.py` | Observe KV cache benefits |
| `05_benchmark.py` | Throughput, latency, concurrency tests |

## Results

All experiment outputs are saved in `results/` directory.

## Experiment Log

See `exp_log.md` for observations, conclusions, and reflections.
