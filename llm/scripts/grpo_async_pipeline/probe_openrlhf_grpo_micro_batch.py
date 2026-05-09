#!/usr/bin/env python3
"""Probe the largest GRPO micro-train batch size that fits on one GPU."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil
from pynvml import (
    nvmlDeviceGetComputeRunningProcesses,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlInit,
    nvmlShutdown,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
COMMON_SCRIPT_DIR = PROJECT_ROOT / "scripts"
if str(COMMON_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_SCRIPT_DIR))

from path_utils import infer_default_model_path, resolve_llm_env_executable, runtime_data_root, runtime_result_root

MODEL_CANDIDATE_NAMES = ("Qwen3-0.6B-Base", "Qwen3-0.6B-BASE", "Qwen3-0.6B")


def alive_process_tree(root_pid: int) -> list[psutil.Process]:
    try:
        root = psutil.Process(root_pid)
    except psutil.Error:
        return []

    processes = [root]
    try:
        processes.extend(root.children(recursive=True))
    except psutil.Error:
        pass

    alive = []
    for proc in processes:
        try:
            if proc.is_running():
                alive.append(proc)
        except psutil.Error:
            continue
    return alive


def cpu_rss_bytes(processes: list[psutil.Process]) -> int:
    total = 0
    for proc in processes:
        try:
            total += proc.memory_info().rss
        except psutil.Error:
            continue
    return total


def gpu_process_bytes(handle, pids: set[int]) -> int:
    total = 0
    try:
        for proc in nvmlDeviceGetComputeRunningProcesses(handle):
            if proc.pid in pids:
                total += proc.usedGpuMemory
    except Exception:
        return 0
    return total


def stop_ray(python_bin: str) -> None:
    ray_bin = str(Path(python_bin).resolve().with_name("ray"))
    subprocess.run([ray_bin, "stop", "--force"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


def run_probe_once(
    *,
    python_bin: str,
    model_path: Path,
    prompt_data: Path,
    reward_script: Path,
    chat_template_file: Path,
    output_root: Path,
    micro_train_batch_size: int,
    gradient_accumulation: int,
    n_samples_per_prompt: int,
    temperature: float,
    top_p: float,
    prompt_max_len: int,
    generate_max_len: int,
    max_len: int,
    vllm_gpu_memory_utilization: float,
    vllm_attention_backend: str,
) -> dict:
    run_dir = output_root / f"micro_train_{micro_train_batch_size}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    train_batch_size = micro_train_batch_size * gradient_accumulation
    rollout_batch_size = max(1, train_batch_size // n_samples_per_prompt)
    chat_template_text = chat_template_file.read_text(encoding="utf-8")

    cmd = [
        python_bin,
        "-m",
        "openrlhf.cli.train_ppo_ray",
        "--pretrain",
        str(model_path),
        "--remote_rm_url",
        str(reward_script),
        "--prompt_data",
        str(prompt_data),
        "--input_key",
        "messages",
        "--label_key",
        "label",
        "--apply_chat_template",
        "--tokenizer_chat_template",
        chat_template_text,
        "--save_path",
        str(run_dir / "training_final_model"),
        "--ckpt_path",
        str(run_dir / "checkpoints_grpo"),
        "--save_hf_ckpt",
        "--disable_ds_ckpt",
        "--save_steps",
        "1000000",
        "--max_ckpt_num",
        "2",
        "--logging_steps",
        "1",
        "--actor_num_nodes",
        "1",
        "--actor_num_gpus_per_node",
        "1",
        "--vllm_num_engines",
        "1",
        "--vllm_tensor_parallel_size",
        "1",
        "--colocate_all_models",
        "--vllm_enable_sleep",
        "--deepspeed_enable_sleep",
        "--enforce_eager",
        "--zero_stage",
        "2",
        "--adam_offload",
        "--gradient_checkpointing",
        "--attn_implementation",
        "eager",
        "--advantage_estimator",
        "group_norm",
        "--kl_estimator",
        "k3",
        "--init_kl_coef",
        "0",
        "--actor_learning_rate",
        "5e-7",
        "--temperature",
        str(temperature),
        "--top_p",
        str(top_p),
        "--num_episodes",
        "1",
        "--max_epochs",
        "1",
        "--n_samples_per_prompt",
        str(n_samples_per_prompt),
        "--micro_rollout_batch_size",
        "1",
        "--micro_train_batch_size",
        str(micro_train_batch_size),
        "--train_batch_size",
        str(train_batch_size),
        "--rollout_batch_size",
        str(rollout_batch_size),
        "--prompt_max_len",
        str(prompt_max_len),
        "--generate_max_len",
        str(generate_max_len),
        "--max_len",
        str(max_len),
        "--max_samples",
        "16",
        "--use_tensorboard",
        str(run_dir / "tensorboard"),
        "--vllm_gpu_memory_utilization",
        str(vllm_gpu_memory_utilization),
        "--wandb_run_name",
        f"grpo_probe_micro_{micro_train_batch_size}",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    env["TOKENIZERS_PARALLELISM"] = "true"
    env["OPENRLHF_VLLM_ATTENTION_BACKEND"] = vllm_attention_backend

    stop_ray(python_bin)
    log_path = run_dir / "probe.log"
    start = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    max_cpu_rss = 0
    max_gpu_proc = 0
    max_gpu_total = 0
    output_chunks: list[str] = []
    try:
        while True:
            line = proc.stdout.readline() if proc.stdout else ""
            if line:
                output_chunks.append(line)

            tree = alive_process_tree(proc.pid)
            pids = {p.pid for p in tree}
            max_cpu_rss = max(max_cpu_rss, cpu_rss_bytes(tree))
            max_gpu_proc = max(max_gpu_proc, gpu_process_bytes(handle, pids))
            try:
                max_gpu_total = max(max_gpu_total, nvmlDeviceGetMemoryInfo(handle).used)
            except Exception:
                pass

            if proc.poll() is not None:
                if proc.stdout:
                    output_chunks.extend(proc.stdout.readlines())
                break
            time.sleep(0.5)
    finally:
        nvmlShutdown()
        stop_ray(python_bin)

    elapsed = time.perf_counter() - start
    output_text = "".join(output_chunks)
    log_path.write_text(output_text, encoding="utf-8")

    failure_reason = None
    if proc.returncode != 0:
        lowered = output_text.lower()
        if "outofmemory" in lowered or "cuda out of memory" in lowered:
            failure_reason = "oom"
        else:
            failure_reason = "other"

    result = {
        "micro_train_batch_size": micro_train_batch_size,
        "gradient_accumulation": gradient_accumulation,
        "train_batch_size": train_batch_size,
        "rollout_batch_size": rollout_batch_size,
        "n_samples_per_prompt": n_samples_per_prompt,
        "temperature": temperature,
        "top_p": top_p,
        "returncode": proc.returncode,
        "success": proc.returncode == 0,
        "failure_reason": failure_reason,
        "elapsed_seconds": elapsed,
        "max_cpu_rss_bytes": max_cpu_rss,
        "max_cpu_rss_gb": max_cpu_rss / (1024**3),
        "max_gpu_process_bytes": max_gpu_proc,
        "max_gpu_process_gb": max_gpu_proc / (1024**3),
        "max_gpu_total_bytes": max_gpu_total,
        "max_gpu_total_gb": max_gpu_total / (1024**3),
        "log_path": str(log_path),
    }
    (run_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    data_root = runtime_data_root(PROJECT_ROOT)
    result_root = runtime_result_root(PROJECT_ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default=str(infer_default_model_path(PROJECT_ROOT, MODEL_CANDIDATE_NAMES)),
    )
    parser.add_argument(
        "--prompt-data",
        default=str(data_root / "deepseek-v3.2-speciale-openr1-math-3k.grpo_prompt_train.jsonl"),
    )
    parser.add_argument(
        "--reward-script",
        default=str(SCRIPT_DIR / "math_exact_match_reward.py"),
    )
    parser.add_argument(
        "--chat-template-file",
        default=str(SCRIPT_DIR / "templates" / "qwen3_06b_base_eot_chat_template.jinja"),
    )
    parser.add_argument(
        "--output-root",
        default=str(result_root / "grpo_micro_batch_probe"),
    )
    parser.add_argument("--min-micro-train-batch", type=int, default=1)
    parser.add_argument("--max-micro-train-batch", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--n-samples-per-prompt", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--prompt-max-len", type=int, default=2048)
    parser.add_argument("--generate-max-len", type=int, default=2048)
    parser.add_argument("--max-len", type=int, default=4096)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.25)
    parser.add_argument(
        "--vllm-attention-backend",
        default="TRITON_ATTN",
        choices=["TRITON_ATTN", "FLEX_ATTENTION", "FLASHINFER", "FLASH_ATTN"],
    )
    parser.add_argument("--keep-going-after-failure", action="store_true", default=False)
    args = parser.parse_args()

    python_bin = resolve_llm_env_executable("python")
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    results = []
    for micro_train_batch_size in range(args.min_micro_train_batch, args.max_micro_train_batch + 1):
        result = run_probe_once(
            python_bin=python_bin,
            model_path=Path(args.model_path).resolve(),
            prompt_data=Path(args.prompt_data).resolve(),
            reward_script=Path(args.reward_script).resolve(),
            chat_template_file=Path(args.chat_template_file).resolve(),
            output_root=output_root,
            micro_train_batch_size=micro_train_batch_size,
            gradient_accumulation=args.gradient_accumulation,
            n_samples_per_prompt=args.n_samples_per_prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            prompt_max_len=args.prompt_max_len,
            generate_max_len=args.generate_max_len,
            max_len=args.max_len,
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            vllm_attention_backend=args.vllm_attention_backend,
        )
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))
        if (not args.keep_going_after_failure) and (not result["success"]):
            break

    summary = {
        "model_path": args.model_path,
        "prompt_data": args.prompt_data,
        "reward_script": args.reward_script,
        "gradient_accumulation": args.gradient_accumulation,
        "n_samples_per_prompt": args.n_samples_per_prompt,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "prompt_max_len": args.prompt_max_len,
        "generate_max_len": args.generate_max_len,
        "max_len": args.max_len,
        "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "vllm_attention_backend": args.vllm_attention_backend,
        "results": results,
        "max_successful_micro_train_batch_size": max(
            (item["micro_train_batch_size"] for item in results if item["success"]),
            default=0,
        ),
    }
    (output_root / "probe_results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("FINAL_SUMMARY=" + json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
