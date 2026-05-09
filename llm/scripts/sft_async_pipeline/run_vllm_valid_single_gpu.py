#!/usr/bin/env python3
"""Run single-GPU vLLM validation with optional async local scheduling.

Design goals:
- single-GPU default behavior
- deterministic prompt formatting
- reproducible chunking
- merged JSONL output named after the model directory

The local Qwen2.5-Math-1.5B config reports max_position_embeddings=4096,
so this script clamps generation to fit inside that real context window.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import socket
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

from transformers import AutoTokenizer


SYSTEM_PROMPT = (
    "You are a careful math solver. Solve the problem step by step. "
    "End your response with a final line in the format: #### <answer>"
)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
COMMON_SCRIPT_DIR = PROJECT_ROOT / "scripts"
if str(COMMON_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_SCRIPT_DIR))

from path_utils import infer_default_model_path, runtime_data_root, runtime_result_root

BLACKWELL_MAJOR = 12
MODEL_CANDIDATE_NAMES = ("Qwen3-0.6B", "Qwen3-0.6B-Base", "Qwen3-0.6B-BASE")


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_model_max_len(model_path: Path) -> int:
    config_path = model_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    max_len = int(config.get("max_position_embeddings", 4096))
    return max_len


def build_prompts(records: list[dict], tokenizer) -> tuple[list[dict], int]:
    prompt_token_lengths = []
    updated = []
    for record in records:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": record["question"]},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        item = dict(record)
        item["prompt"] = prompt
        item["prompt_token_length"] = len(prompt_ids)
        updated.append(item)
        prompt_token_lengths.append(len(prompt_ids))
    return updated, max(prompt_token_lengths)


def chunk_records(records: list[dict], num_chunks: int) -> list[list[dict]]:
    chunk_size = math.ceil(len(records) / num_chunks)
    return [records[i : i + chunk_size] for i in range(0, len(records), chunk_size)]


def maybe_override_chat_template(tokenizer, chat_template_file: str | None) -> None:
    if chat_template_file is None:
        return
    tokenizer.chat_template = Path(chat_template_file).read_text(encoding="utf-8")


def resolve_tokenizer_path(model_path: Path, tokenizer_path_arg: str | None) -> Path:
    if tokenizer_path_arg is None:
        return model_path
    return Path(tokenizer_path_arg)


def build_output_record(record: dict, text: str, model_name: str, worker_index: int) -> dict:
    return {
        "source_index": record["source_index"],
        "split": record.get("split", "valid"),
        "question": record["question"],
        "answer": record["answer"],
        "solution": record.get("solution"),
        "perturbation_type": record.get("perturbation_type"),
        "seed_question": record.get("seed_question"),
        "seed_solution": record.get("seed_solution"),
        "seed_answer": record.get("seed_answer"),
        "model_name": model_name,
        "worker_index": worker_index,
        "prompt_token_length": record.get("prompt_token_length"),
        "generated_text": text,
    }


def parse_device_ids(device_ids_arg: str | None, num_workers: int) -> list[str]:
    if device_ids_arg is None:
        return [str(i) for i in range(num_workers)]

    device_ids = [item.strip() for item in device_ids_arg.split(",") if item.strip()]
    if len(device_ids) < num_workers:
        raise ValueError(
            f"--device-ids provides {len(device_ids)} devices, but num_workers={num_workers}."
        )
    return device_ids[:num_workers]


def resolve_worker_runtime(
    attention_backend: str,
    enforce_eager: bool,
    disable_compilation: bool,
) -> tuple[str | None, bool, int | None]:
    import torch

    major, _minor = torch.cuda.get_device_capability(0)
    auto_blackwell_safe_mode = attention_backend == "auto" and major >= BLACKWELL_MAJOR

    resolved_backend = attention_backend
    if auto_blackwell_safe_mode:
        resolved_backend = "TRITON_ATTN"

    if resolved_backend == "auto":
        resolved_backend = None

    resolved_enforce_eager = enforce_eager or auto_blackwell_safe_mode
    resolved_compilation_config = 0 if (disable_compilation or auto_blackwell_safe_mode) else None
    return resolved_backend, resolved_enforce_eager, resolved_compilation_config


def pick_server_port(server_port_base: int, worker_index: int) -> int:
    preferred_port = server_port_base + worker_index
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", preferred_port))
            return preferred_port
        except OSError:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]


def wait_for_server_ready(base_url: str, process: subprocess.Popen, timeout_seconds: int) -> None:
    import httpx

    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    with httpx.Client(timeout=5.0) as client:
        while time.time() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"vLLM server exited early with return code {process.returncode} before becoming ready."
                )
            try:
                response = client.get(f"{base_url}/v1/models")
                if response.status_code == 200:
                    return
            except Exception as exc:  # pragma: no cover - best effort polling
                last_error = exc
            time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for vLLM server at {base_url}. Last error: {last_error}")


def terminate_process(process: subprocess.Popen, timeout_seconds: int = 10) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout_seconds)


async def generate_with_server_async(
    *,
    model_path: str,
    tokenizer_path: str,
    prompts: list[str],
    concurrency: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    resolved_backend: str | None,
    resolved_enforce_eager: bool,
    resolved_compilation_config: int | None,
) -> list[str]:
    from vllm import SamplingParams
    from vllm.config import CompilationConfig, CompilationMode
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    engine_kwargs = {
        "model": model_path,
        "tokenizer": tokenizer_path,
        "tensor_parallel_size": 1,
        "dtype": "bfloat16",
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enforce_eager": resolved_enforce_eager,
        "disable_log_stats": True,
        "generation_config": "vllm",
        "max_num_seqs": concurrency,
        "attention_backend": getattr(AttentionBackendEnum, resolved_backend) if resolved_backend is not None else None,
    }
    if resolved_compilation_config is not None:
        engine_kwargs["compilation_config"] = CompilationConfig(mode=CompilationMode.NONE)

    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**engine_kwargs))
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    results = [""] * len(prompts)
    total = len(prompts)
    progress_interval = max(1, total // 10)
    prompt_queue: asyncio.Queue[tuple[int, str]] = asyncio.Queue()
    for index, prompt in enumerate(prompts):
        prompt_queue.put_nowait((index, prompt))

    completed = 0
    completed_lock = asyncio.Lock()

    async def worker(worker_index: int) -> None:
        nonlocal completed
        while True:
            try:
                prompt_index, prompt = prompt_queue.get_nowait()
            except asyncio.QueueEmpty:
                return

            request_id = f"worker{worker_index}-prompt{prompt_index}"
            final_output = None
            try:
                async for output in engine.generate(prompt, sampling_params, request_id=request_id):
                    final_output = output
            except Exception as exc:  # pragma: no cover - runtime path
                raise RuntimeError(f"Async engine generation failed for prompt_index={prompt_index}: {exc}") from exc
            finally:
                prompt_queue.task_done()

            text = final_output.outputs[0].text if final_output and final_output.outputs else ""
            results[prompt_index] = text
            async with completed_lock:
                completed += 1
                if completed == total or completed % progress_interval == 0:
                    print(
                        json.dumps(
                            {
                                "event": "server_async_progress",
                                "completed": completed,
                                "total": total,
                                "concurrency": concurrency,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )

    worker_count = min(concurrency, total)
    tasks = [asyncio.create_task(worker(worker_index)) for worker_index in range(worker_count)]
    try:
        await asyncio.gather(*tasks)
        await prompt_queue.join()
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        with suppress(Exception):
            engine.shutdown()

    return results


def launch_vllm_server(
    *,
    args: argparse.Namespace,
    resolved_backend: str | None,
    resolved_enforce_eager: bool,
    resolved_compilation_config: int | None,
) -> tuple[subprocess.Popen, str, str, int]:
    port = pick_server_port(args.server_port_base, args.worker_index)
    base_url = f"http://127.0.0.1:{port}"
    served_model_name = Path(args.model_path).name
    vllm_bin = Path(sys.executable).resolve().with_name("vllm")

    cmd = [
        str(vllm_bin),
        "serve",
        args.model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        served_model_name,
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-num-seqs",
        str(args.batch_size),
        "--generation-config",
        "vllm",
    ]
    if args.tokenizer_path is not None:
        cmd.extend(["--tokenizer", args.tokenizer_path])
    if resolved_backend is not None:
        cmd.extend(["--attention-backend", resolved_backend])
    if resolved_enforce_eager:
        cmd.append("--enforce-eager")
    if resolved_compilation_config is not None:
        cmd.extend(["--compilation-config", json.dumps({"mode": resolved_compilation_config})])

    process = subprocess.Popen(cmd)
    wait_for_server_ready(base_url, process, args.server_start_timeout)
    return process, base_url, served_model_name, port


def run_controller(args: argparse.Namespace) -> None:
    model_path = Path(args.model_path)
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = model_path.name
    work_dir = output_dir / f"{model_name}_valid_1000_workdir"
    chunk_input_dir = work_dir / "inputs"
    chunk_output_dir = work_dir / "outputs"
    chunk_log_dir = work_dir / "logs"
    chunk_input_dir.mkdir(parents=True, exist_ok=True)
    chunk_output_dir.mkdir(parents=True, exist_ok=True)
    chunk_log_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(input_path)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise ValueError("No records found for inference.")

    tokenizer_path = resolve_tokenizer_path(model_path, args.tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)
    maybe_override_chat_template(tokenizer, args.chat_template_file)
    prepared_records, max_prompt_tokens = build_prompts(records, tokenizer)

    model_context_limit = get_model_max_len(model_path)
    prompt_buffer = 32
    effective_max_new_tokens = min(
        args.requested_max_new_tokens,
        model_context_limit - max_prompt_tokens - prompt_buffer,
    )
    if effective_max_new_tokens <= 0:
        raise ValueError(
            f"No room left for generation: model_context_limit={model_context_limit}, "
            f"max_prompt_tokens={max_prompt_tokens}, prompt_buffer={prompt_buffer}"
        )
    engine_max_model_len = min(
        model_context_limit,
        max_prompt_tokens + effective_max_new_tokens + prompt_buffer,
    )

    device_ids = parse_device_ids(args.device_ids, args.num_workers)
    chunks = chunk_records(prepared_records, args.num_workers)
    chunk_input_paths = []
    for worker_index, chunk in enumerate(chunks):
        chunk_path = chunk_input_dir / f"valid_chunk_{worker_index:02d}.jsonl"
        write_jsonl(chunk_path, chunk)
        chunk_input_paths.append(chunk_path)

    processes: list[subprocess.Popen] = []
    for worker_index, chunk_path in enumerate(chunk_input_paths):
        chunk_output_path = chunk_output_dir / f"valid_chunk_{worker_index:02d}.jsonl"
        log_path = chunk_log_dir / f"worker_{worker_index:02d}.log"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--worker-index",
            str(worker_index),
            "--model-path",
            str(model_path),
            "--chunk-input",
            str(chunk_path),
            "--chunk-output",
            str(chunk_output_path),
            "--batch-size",
            str(args.batch_size),
            "--max-model-len",
            str(engine_max_model_len),
            "--max-new-tokens",
            str(effective_max_new_tokens),
            "--temperature",
            str(args.temperature),
            "--top-p",
            str(args.top_p),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--attention-backend",
            args.attention_backend,
            "--scheduler-mode",
            args.scheduler_mode,
            "--server-port-base",
            str(args.server_port_base),
            "--server-start-timeout",
            str(args.server_start_timeout),
        ]
        if args.tokenizer_path is not None:
            cmd.extend(["--tokenizer-path", args.tokenizer_path])
        if args.enforce_eager:
            cmd.append("--enforce-eager")
        if args.disable_compilation:
            cmd.append("--disable-compilation")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = device_ids[worker_index]
        log_file = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        process.log_file = log_file  # type: ignore[attr-defined]
        processes.append(process)

    failed = []
    for process in processes:
        return_code = process.wait()
        process.log_file.close()  # type: ignore[attr-defined]
        if return_code != 0:
            failed.append(return_code)

    if failed:
        raise RuntimeError(f"One or more workers failed: {failed}")

    merged = []
    for worker_index in range(len(chunk_input_paths)):
        chunk_output_path = chunk_output_dir / f"valid_chunk_{worker_index:02d}.jsonl"
        merged.extend(load_jsonl(chunk_output_path))

    merged.sort(key=lambda x: x["source_index"])
    if len(merged) != len(records):
        raise RuntimeError(f"Merged record count mismatch: expected {len(records)}, got {len(merged)}.")

    final_output_path = output_dir / f"{model_name}.valid_1000.outputs.jsonl"
    write_jsonl(final_output_path, merged)

    metadata = {
        "model_name": model_name,
        "model_path": str(model_path),
        "tokenizer_path": str(tokenizer_path),
        "input_path": str(input_path),
        "output_path": str(final_output_path),
        "num_records": len(merged),
        "num_workers": len(chunk_input_paths),
        "batch_size": args.batch_size,
        "model_context_limit": model_context_limit,
        "engine_max_model_len": engine_max_model_len,
        "max_prompt_tokens": max_prompt_tokens,
        "requested_max_new_tokens": args.requested_max_new_tokens,
        "effective_max_new_tokens": effective_max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "device_ids": device_ids,
        "chat_template_file": args.chat_template_file,
        "attention_backend": args.attention_backend,
        "scheduler_mode": args.scheduler_mode,
        "enforce_eager": args.enforce_eager,
        "disable_compilation": args.disable_compilation,
    }
    metadata_path = output_dir / f"{model_name}.valid_1000.metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def run_worker(args: argparse.Namespace) -> None:
    from vllm import LLM, SamplingParams
    from vllm.config.attention import AttentionConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    chunk_input = Path(args.chunk_input)
    chunk_output = Path(args.chunk_output)
    model_name = Path(args.model_path).name

    records = load_jsonl(chunk_input)
    prompts = [record["prompt"] for record in records]

    resolved_backend, resolved_enforce_eager, resolved_compilation_config = resolve_worker_runtime(
        attention_backend=args.attention_backend,
        enforce_eager=args.enforce_eager,
        disable_compilation=args.disable_compilation,
    )

    outputs = []
    server_port = None
    if args.scheduler_mode == "static_batch":
        llm_kwargs = {
            "model": args.model_path,
            "tokenizer": args.tokenizer_path or args.model_path,
            "tensor_parallel_size": 1,
            "dtype": "bfloat16",
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": resolved_enforce_eager,
        }
        if resolved_backend is not None:
            llm_kwargs["attention_config"] = AttentionConfig(backend=getattr(AttentionBackendEnum, resolved_backend))
        if resolved_compilation_config is not None:
            llm_kwargs["compilation_config"] = resolved_compilation_config
        llm = LLM(**llm_kwargs)
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
        )
        for start in range(0, len(prompts), args.batch_size):
            batch_records = records[start : start + args.batch_size]
            batch_prompts = prompts[start : start + args.batch_size]
            generations = llm.generate(batch_prompts, sampling_params, use_tqdm=False)
            for record, generation in zip(batch_records, generations):
                text = generation.outputs[0].text if generation.outputs else ""
                outputs.append(build_output_record(record, text, model_name, args.worker_index))
    elif args.scheduler_mode == "full_queue":
        llm_kwargs = {
            "model": args.model_path,
            "tokenizer": args.tokenizer_path or args.model_path,
            "tensor_parallel_size": 1,
            "dtype": "bfloat16",
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": resolved_enforce_eager,
        }
        if resolved_backend is not None:
            llm_kwargs["attention_config"] = AttentionConfig(backend=getattr(AttentionBackendEnum, resolved_backend))
        if resolved_compilation_config is not None:
            llm_kwargs["compilation_config"] = resolved_compilation_config
        llm = LLM(**llm_kwargs)
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
        )
        generations = llm.generate(prompts, sampling_params, use_tqdm=False)
        for record, generation in zip(records, generations):
            text = generation.outputs[0].text if generation.outputs else ""
            outputs.append(build_output_record(record, text, model_name, args.worker_index))
    else:
        texts = asyncio.run(
            generate_with_server_async(
                model_path=args.model_path,
                tokenizer_path=args.tokenizer_path or args.model_path,
                prompts=prompts,
                concurrency=args.batch_size,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_new_tokens,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                resolved_backend=resolved_backend,
                resolved_enforce_eager=resolved_enforce_eager,
                resolved_compilation_config=resolved_compilation_config,
            )
        )
        outputs = [
            build_output_record(record, text, model_name, args.worker_index)
            for record, text in zip(records, texts)
        ]

    write_jsonl(chunk_output, outputs)
    print(
        json.dumps(
                {
                    "worker_index": args.worker_index,
                    "chunk_input": str(chunk_input),
                    "chunk_output": str(chunk_output),
                    "num_records": len(outputs),
                    "resolved_attention_backend": resolved_backend,
                    "resolved_enforce_eager": resolved_enforce_eager,
                    "resolved_compilation_config": resolved_compilation_config,
                    "scheduler_mode": args.scheduler_mode,
                    "tokenizer_path": args.tokenizer_path or args.model_path,
                    "server_port_base": args.server_port_base,
                    "server_port": None,
                },
            ensure_ascii=False,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    data_root = runtime_data_root(PROJECT_ROOT)
    result_root = runtime_result_root(PROJECT_ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help="Run one local worker process.")
    parser.add_argument("--worker-index", type=int, default=-1)
    parser.add_argument("--model-path", default=str(infer_default_model_path(PROJECT_ROOT, MODEL_CANDIDATE_NAMES)))
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Optional Hugging Face tokenizer path. If omitted, the model path tokenizer is used.",
    )
    parser.add_argument("--input", default=str(data_root / "valid_1000.jsonl"))
    parser.add_argument("--output-dir", default=str(result_root))
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--device-ids",
        default=None,
        help="Comma-separated physical GPU ids to assign to workers, e.g. '0'.",
    )
    parser.add_argument(
        "--chat-template-file",
        default=None,
        help="Optional path to a Jinja chat template that overrides tokenizer.chat_template.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--scheduler-mode",
        default="full_queue",
        choices=["static_batch", "full_queue", "server_async"],
        help="Request scheduling mode. full_queue lets vLLM roll the queue internally; server_async keeps a fixed in-flight concurrency window with the local async engine.",
    )
    parser.add_argument(
        "--requested-max-new-tokens",
        type=int,
        default=4096,
        help="Requested generation length. The controller clamps this to the model's true context limit.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument(
        "--attention-backend",
        default="auto",
        choices=["auto", "FLASH_ATTN", "TRITON_ATTN", "FLASHINFER", "FLEX_ATTENTION"],
        help="vLLM attention backend. 'auto' keeps default behavior except on Blackwell GPUs, where it switches to TRITON_ATTN.",
    )
    parser.add_argument("--enforce-eager", action="store_true", default=False)
    parser.add_argument("--disable-compilation", action="store_true", default=False)
    parser.add_argument("--server-port-base", type=int, default=18000)
    parser.add_argument("--server-start-timeout", type=int, default=300)

    # Worker-only parameters.
    parser.add_argument("--chunk-input", default=None)
    parser.add_argument("--chunk-output", default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.worker:
        if args.chunk_input is None or args.chunk_output is None:
            raise ValueError("Worker mode requires --chunk-input and --chunk-output.")
        run_worker(args)
    else:
        run_controller(args)


if __name__ == "__main__":
    main()
