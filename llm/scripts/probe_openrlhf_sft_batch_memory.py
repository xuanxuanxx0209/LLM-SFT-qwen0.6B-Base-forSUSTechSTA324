#!/usr/bin/env python3
"""Probe the largest OpenRLHF SFT micro batch size that fits on one GPU.

The probe runs short single-GPU OpenRLHF SFT jobs against a stress dataset and
records peak CPU RSS and GPU memory usage for each tested micro batch size.
"""

from __future__ import annotations

import argparse
import json
import os
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
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from path_utils import infer_default_model_path, runtime_data_root, runtime_result_root

MODEL_CANDIDATE_NAMES = ("Qwen3-0.6B-Base", "Qwen3-0.6B-BASE", "Qwen3-0.6B", "Qwen2.5-Math-1.5B")


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


def run_probe_once(
    micro_batch_size: int,
    dataset: Path,
    model_path: Path,
    run_dir: Path,
    master_port: int,
    attn_implementation: str,
    adam_offload: bool,
    zero_stage: int,
) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "deepspeed",
        "--master_port",
        str(master_port),
        "--module",
        "openrlhf.cli.train_sft",
        "--pretrain",
        str(model_path),
        "--dataset",
        str(dataset),
        "--input_key",
        "messages",
        "--apply_chat_template",
        "--multiturn",
        "--max_len",
        "4096",
        "--max_epochs",
        "1",
        "--micro_train_batch_size",
        str(micro_batch_size),
        "--train_batch_size",
        str(micro_batch_size),
        "--max_samples",
        "8",
        "--save_path",
        str(run_dir / "final_model"),
        "--ckpt_path",
        str(run_dir / "checkpoints_sft"),
        "--save_steps",
        "1000000",
        "--save_hf_ckpt",
        "--max_ckpt_num",
        "2",
        "--logging_steps",
        "1",
        "--zero_stage",
        str(zero_stage),
        "--learning_rate",
        "5e-6",
        "--gradient_checkpointing",
        "--attn_implementation",
        attn_implementation,
    ]
    if adam_offload:
        cmd.append("--adam_offload")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    env["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

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
            time.sleep(0.2)
    finally:
        nvmlShutdown()

    elapsed = time.perf_counter() - start
    output_text = "".join(output_chunks)
    log_path.write_text(output_text, encoding="utf-8")

    failure_reason = None
    if proc.returncode != 0:
        if "CUDA out of memory" in output_text or "OutOfMemoryError" in output_text:
            failure_reason = "oom"
        else:
            failure_reason = "other"

    result = {
        "micro_train_batch_size": micro_batch_size,
        "train_batch_size": micro_batch_size,
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
        "--dataset",
        default=str(data_root / "deepseek-v3.2-speciale-openr1-math-3k.sft_train.longest8.jsonl"),
    )
    parser.add_argument("--model-path", default=str(infer_default_model_path(PROJECT_ROOT, MODEL_CANDIDATE_NAMES)))
    parser.add_argument("--output-root", default=str(result_root / "openrlhf_batch_probe"))
    parser.add_argument("--min-micro-batch", type=int, default=1)
    parser.add_argument("--max-micro-batch", type=int, default=8)
    parser.add_argument("--keep-going-after-failure", action="store_true", default=False)
    parser.add_argument("--master-port-base", type=int, default=29700)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--adam-offload", action="store_true", default=False)
    parser.add_argument("--zero-stage", type=int, default=2)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    results = []
    for micro in range(args.min_micro_batch, args.max_micro_batch + 1):
        result = run_probe_once(
            micro_batch_size=micro,
            dataset=Path(args.dataset),
            model_path=Path(args.model_path),
            run_dir=output_root / f"micro_{micro}",
            master_port=args.master_port_base + micro,
            attn_implementation=args.attn_implementation,
            adam_offload=args.adam_offload,
            zero_stage=args.zero_stage,
        )
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))
        if (not args.keep_going_after_failure) and (not result["success"]):
            break

    summary = {
        "dataset": args.dataset,
        "model_path": args.model_path,
        "attn_implementation": args.attn_implementation,
        "adam_offload": args.adam_offload,
        "zero_stage": args.zero_stage,
        "results": results,
        "max_successful_micro_batch": max((r["micro_train_batch_size"] for r in results if r["success"]), default=0),
    }
    (output_root / "probe_results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("FINAL_SUMMARY=" + json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
