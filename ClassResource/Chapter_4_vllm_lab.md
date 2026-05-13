# Week 4 Lab Handout: vLLM

In this class, your goal is to truly understand and run through the following pipeline in one session:

> **logits → decoding → autoregressive generation → KV cache → vLLM serving → benchmarking → experiment log**

The goal of this lab is not just to “call an API,” but to connect **generation mechanics, caching, system throughput, and experiment design** into one complete picture.

------

## 1. Learning Objectives

By the end of this lab, you should be able to:

1. Explain in your own words the relationship between **logits, probabilities, and sampled tokens**.
2. Understand the behavioral differences between greedy decoding, sampling, top-k, and top-p.
3. Explain why autoregressive inference is a process of **appending one token at a time**.
4. Explain the role of **KV cache** during inference and why it reduces repeated computation.
5. Deploy a minimal usable **vLLM OpenAI-compatible server**.
6. Write a minimal client to complete one local inference request.
7. Design and run 3 groups of experiments:
   - decoding strategy experiments
   - KV cache / shared prefix experiments
   - throughput / latency / concurrency experiments
8. Write a proper **experiment log** that includes observations, data, explanations, failed attempts, and reflections.

------

## 2. How This Class Works

### 2.1 Your relationship with Claude Code

In this lab, Claude Code is:

- your pair programmer
- your shell / scripting assistant
- your debugging assistant
- your code refactoring assistant
- but **not** your ghostwriter for experimental conclusions

### 2.2 What Claude Code is allowed to do

Claude Code may:

- create project directories
- generate Python scripts as requested
- help you fix bugs
- help complete command-line arguments
- help write outputs into csv / md / json
- explain error messages
- make your benchmark scripts more robust

### 2.3 What Claude Code is not allowed to do for you

Claude Code must not:

- write your final experimental conclusions for you
- directly write “what I learned” for you
- fabricate experimental results when you have not actually run the code
- skip the observation process and produce polished but empty analysis

### 2.4 Recommended collaboration format

Whenever you ask Claude Code for help, try to follow this format:

```text
Goal:
Current file:
What you should modify:
What you should not modify:
After finishing, please explain:
```

For example:

```text
Goal: Implement a minimal vLLM client.
Current file: scripts/01_call_vllm.py
What you should modify: Only write a requests-based chat/completions call.
What you should not modify: Do not introduce extra frameworks, and do not split it into many files.
After finishing, please explain: What each parameter does, and how to verify that the request actually succeeded.
```

------

## 3. Suggested Time Plan for This Lab

| Stage  | Time     | Goal                                                         |
| ------ | -------- | ------------------------------------------------------------ |
| Part A | 3–5 min  | Check environment, start the server, complete the first inference |
| Part B | 5–10 min | Understand decoding behavior with a minimal script           |
| Part C | 5–10 min | Run KV cache / shared prefix experiments                     |
| Part D | 5–15 min | Run throughput / latency / concurrency experiments           |
| Part E | 5–10 min | Write the experiment log and conclusions                     |

------

## 4. What You Need to Submit

In the end, you should submit these 4 things:

1. the core scripts under `scripts/`
2. the experiment outputs under `results/` (csv/json/md are all acceptable)
3. `exp_log.md`
4. a short conclusion answering:
   - which settings are more stable?
   - which settings are faster?
   - what does KV cache bring?
   - what are the advantages of vLLM over a hand-written decoder or plain `transformers.generate()`?

------

## 5. Suggested Project Structure

Do not put everything into one huge script. A recommended structure is:

```text
lab_inference/
├─ README.md
├─ scripts/
│  ├─ 00_check_env.py
│  ├─ 01_call_vllm.py
│  ├─ 02_manual_decode.py
│  ├─ 03_decoding_compare.py
│  ├─ 04_prefix_cache_test.py
│  └─ 05_benchmark.py
├─ results/
│  ├─ decoding_compare.csv
│  ├─ prefix_cache_results.csv
│  ├─ concurrency_results.csv
│  └─ notes.md
└─ exp_log.md
```

------

## 6. Minimal Background: Connect the Concepts First

### 6.1 From logits to tokens

At each step, a language model does not “output a whole sentence directly.” Instead, it:

1. reads the current context
2. computes the **logits** for all vocabulary tokens at the next position
3. turns logits into a probability distribution through softmax
4. chooses the next token according to some strategy
5. appends that token to the sequence
6. repeats the process

This is **autoregressive decoding**.

### 6.2 Greedy, sampling, top-k, top-p

- **greedy**: choose the token with the highest probability at every step
- **sampling**: randomly sample according to the probability distribution
- **top-k**: only sample from the top `k` most probable tokens
- **top-p**: keep the smallest candidate set whose cumulative probability reaches `p`, then sample from it

As a rule of thumb:

- stable tasks are better suited to low temperature and low randomness
- creative tasks are better suited to moderate sampling
- `top-p` is often more adaptive than fixed `top-k`

### 6.3 Why KV cache is needed

Inference generates tokens one step at a time.

If, for every new token, you recompute all attention-related calculations for all previous tokens, the cost becomes very high.

So during inference, we usually cache the historical **key/value states** of previous tokens. This is the **KV cache**. The intuition is:

> If the attention key/value for past tokens has already been computed, do not compute it again.

Remember these two sentences:

1. **KV cache is an inference optimization, not a training optimization.**
2. **Its main benefit comes from avoiding repeated computation over the same historical prefix.**

### 6.4 Why vLLM matters

When you only write a local `transformers.generate()` script, you are learning **how a model generates**.

When you use **vLLM**, you begin to understand **how a model is served efficiently**:

- OpenAI-compatible API
- better throughput
- more systematic cache management
- better support for benchmarking and multi-request concurrency

------

## 7. Key Code (You Should Learn to Use the Following)

### 7.1 Minimal vLLM server startup

Below is a minimal working example. Replace the model name with one that is actually available to you.

```bash
mkdir -p /home/ubuntu/models/Qwen3-0.6B

modelscope download \
  --model Qwen/Qwen3-0.6B \
  --local_dir /home/ubuntu/models/Qwen3-0.6B

vllm serve /home/ubuntu/models/Qwen3-0.6B-Base --dtype auto --host 0.0.0.0 --port 8888 --api-key token-abc123
```

If you are running a local single-machine experiment, the OpenAI-compatible API is usually exposed at `http://localhost:8000` by default.

#### You need to verify 3 things

1. the server has actually started
2. `/v1/models` returns a model list
3. `/v1/chat/completions` successfully returns text

------

### 7.2 Minimal client

```python
# scripts/01_call_vllm.py
import time
import requests

BASE_URL = "http://localhost:8000"
API_KEY = "token-abc123"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}


def chat_once(prompt, model, temperature=0.0, top_p=1.0, top_k=None, max_tokens=128):
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
```

------

### 7.3 The core loop of a hand-written decoder

This code is not meant to replace vLLM. It is meant to let you really see what generation is doing.

```python
# scripts/02_manual_decode.py
import time
import torch
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class DecodeConfig:
    max_new_tokens: int = 50
    strategy: str = "greedy"   # "greedy" or "sample"
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    stop_on_eos: bool = True


def top_k_filter(logits, top_k):
    if top_k <= 0:
        return logits
    values, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
    cutoff = values[..., -1, None]
    return torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)


def top_p_filter(logits, top_p):
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
    if cfg.strategy == "greedy":
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / cfg.temperature
    logits = top_k_filter(logits, cfg.top_k)
    logits = top_p_filter(logits, cfg.top_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def manual_decode(model, tokenizer, prompt, cfg: DecodeConfig, device="cuda"):
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    generated = input_ids.clone()
    new_ids = []

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(cfg.max_new_tokens):
            outputs = model(input_ids=generated)
            logits = outputs.logits[:, -1, :]
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
```

### What you should observe

- Why can this version “run,” but is not suitable for efficient serving?
- Why is calling `model(input_ids=generated)` at every step expensive?
- Without caching, where is the wasted computation?

------

### 7.4 Fetching `/metrics`

```python
# reusable inside scripts/05_benchmark.py
import re
import time
import requests

METRIC_RE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)\{[^}]*\}\s+([0-9eE+\-.]+)$|^([a-zA-Z_:][a-zA-Z0-9_:]*)\s+([0-9eE+\-.]+)$"
)


def fetch_metrics_text(base_url="http://localhost:8000"):
    r = requests.get(f"{base_url}/metrics", timeout=20)
    r.raise_for_status()
    return r.text


def parse_metrics(metrics_text):
    out = {}
    for line in metrics_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = METRIC_RE.match(line)
        if not m:
            continue
        name = m.group(1) or m.group(3)
        val = m.group(2) or m.group(4)
        try:
            out[name] = float(val)
        except ValueError:
            pass
    return out


def metrics_snapshot(base_url="http://localhost:8000"):
    return {
        "t": time.perf_counter(),
        "metrics": parse_metrics(fetch_metrics_text(base_url)),
    }
```

### Metrics you should pay attention to

Focus on:

- total request count
- prompt token count
- generation token count
- p50 / p95 latency under concurrency
- estimated tokens per second

------

## 8. Formal Task Design

This set of tasks is not “all-or-nothing.” It is a layered progression. You must complete at least **Tasks 1, 2, and 4**.

------

## Task 1: Start the service and complete the first inference

### Goal

You need to prove that you have connected the full pipeline:

> local/server model → vLLM service → HTTP request → text output

### Requirements

1. start the vLLM service
2. use a script to fetch `/v1/models`
3. send one request to `/v1/chat/completions`
4. print the returned `usage`, output text, and latency
5. save the result to `results/task1_sanity_check.json`

### Passing criteria

In your log, you must answer:

- which model did you request?
- roughly how high was the latency for a single request?
- what fields appeared in `usage`?
- if it failed, how did you troubleshoot it?

### Recommended prompt to Claude Code

```text
Please help me create scripts/01_call_vllm.py.
Requirements:
1. First request /v1/models and automatically take the first model id;
2. Then send one chat/completions request;
3. Print latency, usage, and text;
4. Save the result to results/task1_sanity_check.json;
5. Keep the code short and avoid unnecessary abstraction.
```

------

## Task 2: Compare different decoding strategies

### Goal

You need to answer one of the most important questions:

> **Why are some settings more stable, while others are more divergent?**

### Experiment design

Prepare at least two kinds of prompts:

1. **stable task**
   - example: simple arithmetic
   - example: only output a single number
2. **open-ended task**
   - example: write one slogan
   - example: write the opening of a short story

### Compare at least these configurations

| config id | temperature | top_p | top_k     | note                  |
| --------- | ----------- | ----- | --------- | --------------------- |
| A         | 0.0         | 1.0   | 0 or None | close to greedy       |
| B         | 0.2         | 1.0   | 20        | slightly conservative |
| C         | 0.7         | 1.0   | 20        | medium randomness     |
| D         | 1.0         | 0.9   | 50        | more open-ended       |

### At minimum, you should measure

- the number of **unique outputs** across 3 repeated runs
- average output length
- average latency
- your subjective judgment of “stability / diversity”

### Questions you need to answer

1. Why are stable tasks not suitable for high randomness?
2. Why are creative tasks not always well served by greedy decoding?
3. Why can `top-p` be more flexible than fixed `top-k`?

### Deliverables

- `results/decoding_compare.csv`
- the Task 2 analysis section in `exp_log.md`

------

## Task 3 (Intermediate): Observe the benefit of KV cache / shared prefix

### Goal

Move from conceptual understanding to observable phenomena.

### Design idea

Construct a **long shared prefix**, then pair it with multiple different questions. For example:

- a long course description
- a long table or FAQ
- a long article
- a long system prompt

Then compare two kinds of requests:

1. **requests sharing the same long prefix**
2. **requests with different prefixes but similar length**

### Metrics to observe

- single-request latency
- average latency
- if supported by your version, prefix-cache-related metrics
- whether you observe any sign that “later requests become faster”

### Note

Different vLLM versions may differ in how prefix caching is enabled and what the default behavior is. You need to:

1. check the local vLLM version first
2. confirm related arguments via `vllm serve --help` or local docs
3. avoid making strong claims before confirming version behavior

### Recommended experimental framework

```python
shared_prefix = """
You are a teaching assistant for a course. Below is a long piece of course material:
...
...(at least a few hundred to a few thousand tokens)
"""

questions = [
    "Summarize the theme of this material in one sentence.",
    "Which part of this material is most suitable for beginners?",
    "If this were turned into a classroom lab, what should the first step be?",
]
```

### Questions you need to answer

1. Did you observe any benefit from the shared prefix?
2. If not, what might be the reason?
3. How does this relate to the intuition behind KV cache?

------

## Task 4 (Intermediate to challenging): Run throughput / latency / concurrency benchmarks

### Goal

For the first time, think like a systems engineer:

> **What settings make a single request faster? What settings improve overall throughput? Why can the two conflict?**

### Experiment A: concurrency sweep

Keep fixed:

- one short prompt
- one model
- `temperature=0.0`
- fixed `max_tokens`
- total request count `n_requests=8` or `16`

Vary:

- `concurrency in [1, 2, 4, 8]`

You should record:

- wall time
- p50 latency
- p95 latency
- total completion tokens
- estimated tokens/s

### Experiment B: workload comparison

Compare two workloads:

1. **short-answer workload**: one sentence or one number is enough
2. **reasoning workload**: requires generating more tokens

You need to answer:

- which workload is more “system-friendly”?
- why are reasoning-style requests more likely to increase latency?
- why are throughput and single-request latency often not optimized at the same operating point?

### Recommended statistics helper

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics
import time


def run_batch_concurrent(call_fn, n_requests=8, concurrency=4):
    t0 = time.perf_counter()
    rows = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(call_fn, i) for i in range(n_requests)]
        for fut in as_completed(futures):
            rows.append(fut.result())
    wall_s = time.perf_counter() - t0

    lats = [r["latency_s"] for r in rows]
    lats_sorted = sorted(lats)

    def pct(xs, p):
        if not xs:
            return None
        k = int(round((len(xs) - 1) * p / 100))
        return xs[k]

    return {
        "wall_s": wall_s,
        "p50_s": pct(lats_sorted, 50),
        "p95_s": pct(lats_sorted, 95),
        "mean_s": statistics.mean(lats),
        "rows": rows,
    }
```

------

## Task 5: Compare three generation paths

### Goal

Tie the whole course together:

1. `manual_decode`
2. `transformers.generate(...)`
3. `vLLM API`

### You are not comparing “whose code is shorter”

You are comparing:

- which is the most transparent?
- which is best for teaching?
- which is best for production serving?
- which has stronger benchmarking advantages?

### Judgments you should write

Suggested sentence patterns:

- `manual_decode` is best for understanding ________, but not suitable for ________.
- `transformers.generate()` is best for ________, because ________.
- `vLLM` is best for ________, especially in ________ scenarios.

------

## 9. Experiment Log Template (Required Submission)

Please write `exp_log.md` using the following template:

```markdown
# Experiment Log: Decoding / KV Cache / vLLM

## 1. Environment Information
- Date:
- Machine:
- GPU:
- Python version:
- transformers version:
- vLLM version:
- Model:

## 2. Task 1: Service Bring-up
### What I did
### Screenshot / output summary
### Problems encountered
### How I solved them

## 3. Task 2: Decoding Strategy Comparison
### Prompt design
### Configuration table
### Result summary
### My observations
- Which configuration was the most stable?
- Which configuration was the most divergent?
- Why?

## 4. Task 3: KV cache / shared prefix
### Experiment design
### Result summary
### My explanation
- Did I observe any caching benefit?
- If not, what might be the reason?

## 5. Task 4: Concurrency / Throughput / Latency
### Experiment setup
### Result table
### My explanation
- How did p50 / p95 change?
- How did tokens/s change?
- What is the most practical operating point?

## 6. Summary of the Three Generation Paths
- manual decode:
- transformers.generate:
- vLLM:

## 7. Failed Attempts
- Which experiments failed?
- Which assumptions were wrong?
- How did I revise them?

## 8. Final Conclusion (must be written by yourself)
- About decoding, I learned:
- About KV cache, I learned:
- About vLLM, I learned:
- If I were to give 3 suggestions to the next class, I would say:
```

------

## 10. Optional Appendix: Task Prompts You Can Directly Give to Claude Code

### Task A: Help me set up the environment first

```text
Please inspect the current directory and create the lab_inference project structure.
Then generate a minimal README explaining the roles of scripts, results, and exp_log.md.
Do not write too much extra explanation; only keep what is truly needed for this lab.

The Qwen3-0.6B is in the /home/models/Qwen3-0.6B.
```

### Task B: Help me write the minimal vLLM client

```text
Please create scripts/01_call_vllm.py.
Requirements:
- first request /v1/models
- automatically take the first model id
- then make one chat/completions request
- print latency, usage, and text
- save the result to results/task1_sanity_check.json
- do not use the openai SDK, only requests
```

### Task C: Help me write the decoding comparison script

```text
Please create scripts/03_decoding_compare.py.
Requirements:
- include two prompts: one stable task and one open-ended task
- compare 4 parameter groups
- repeat each group 3 times
- write csv output to results/decoding_compare.csv
- each record should include at least prompt_name, temperature, top_p, top_k, latency_s, and text
- use the transformers to load model
```

### Task D: Help me write the concurrency benchmark

```text
Please create scripts/05_benchmark.py.
Requirements:
- support concurrency in [1,2,4,8]
- keep the total number of requests fixed
- output p50, p95, mean, and wall time
- if possible, also estimate generation tokens/s from /metrics and write it into the results
```

### Task E: Do not write conclusions for me, only organize the results

```text
Please read the csv/json files under results/ and help me organize a “result summary outline.”
Important: do not write the final conclusions for me, and do not fabricate explanations;
you may only organize the data into an outline that helps me write exp_log.md myself.
```

