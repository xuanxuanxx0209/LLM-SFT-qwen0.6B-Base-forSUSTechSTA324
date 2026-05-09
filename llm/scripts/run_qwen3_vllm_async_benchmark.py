#!/usr/bin/env python3
"""Launch a local vLLM server and benchmark async generation on 1000 samples."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SCRIPT_DIR / "grpo_async_pipeline") not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR / "grpo_async_pipeline"))

from answer_utils import answers_match, clean_answer, extract_prediction
from path_utils import infer_default_model_path, runtime_data_root, runtime_result_root


SYSTEM_PROMPT = (
    "You are a careful math solver. Solve the problem step by step. "
    "End your response with a final line in the format: #### <answer>"
)
MODEL_CANDIDATE_NAMES = ("Qwen3-0.6B", "Qwen3-0.6B-Base", "Qwen3-0.6B-BASE")
console = Console()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record.setdefault("source_index", index)
            records.append(record)
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def tail_text(path: Path, chars: int = 6000) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")[-chars:]


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * q))
    return ordered[index]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default=str(infer_default_model_path(PROJECT_ROOT, MODEL_CANDIDATE_NAMES)),
    )
    parser.add_argument(
        "--input",
        default=str(runtime_data_root(PROJECT_ROOT) / "valid_1000.jsonl"),
    )
    parser.add_argument(
        "--output-root",
        default=str(runtime_result_root(PROJECT_ROOT)),
    )
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--display-samples", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--served-model-name", default="Qwen3-0.6B-local")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--server-ready-timeout", type=int, default=420)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--keep-server", action="store_true", default=False)
    return parser


def launch_server(args: argparse.Namespace, run_dir: Path) -> tuple[subprocess.Popen[str], Path]:
    server_log_path = run_dir / "vllm_server.log"
    cmd = [
        "vllm",
        "serve",
        args.model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--served-model-name",
        args.served_model_name,
        "--dtype",
        "bfloat16",
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--trust-remote-code",
        "--enforce-eager",
    ]
    env = os.environ.copy()
    env.setdefault("LLM_RUNTIME_ROOT", "/dev/shm/llm")
    env.setdefault("VLLM_NO_USAGE_STATS", "1")
    process = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=server_log_path.open("w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    return process, server_log_path


def terminate_server(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + 15
    while time.time() < deadline:
        if process.poll() is not None:
            return
        time.sleep(0.5)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def wait_for_server(base_url: str, process: subprocess.Popen[str], log_path: Path, timeout_seconds: int) -> None:
    deadline = time.time() + timeout_seconds
    last_error: str | None = None
    with httpx.Client(timeout=10.0) as client:
        while time.time() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"vLLM server exited early with return code {process.returncode}.\n{tail_text(log_path)}"
                )
            try:
                response = client.get(f"{base_url}/v1/models")
                if response.status_code == 200:
                    return
                last_error = f"status={response.status_code} body={response.text[:200]}"
            except Exception as exc:
                last_error = repr(exc)
            time.sleep(2)
    raise TimeoutError(f"Timed out waiting for vLLM server. Last error: {last_error}\n{tail_text(log_path)}")


async def request_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    record: dict,
    args: argparse.Namespace,
    base_url: str,
) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": record["question"]},
    ]
    payload = {
        "model": args.served_model_name,
        "messages": messages,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }

    async with semaphore:
        last_error: str | None = None
        for attempt in range(1, args.retries + 1):
            started_at = time.perf_counter()
            try:
                response = await client.post(f"{base_url}/v1/chat/completions", json=payload)
                response.raise_for_status()
                data = response.json()
                latency_seconds = time.perf_counter() - started_at
                generated_text = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                prediction = extract_prediction(generated_text)
                gold_answer = clean_answer(record.get("answer", ""))
                is_correct = answers_match(prediction, gold_answer)
                return {
                    "source_index": record.get("source_index"),
                    "split": record.get("split", "valid"),
                    "question": record["question"],
                    "solution": record.get("solution"),
                    "answer": record.get("answer"),
                    "system_prompt": SYSTEM_PROMPT,
                    "generated_text": generated_text,
                    "predicted_answer": prediction,
                    "final_answer": gold_answer,
                    "is_correct": bool(is_correct),
                    "latency_seconds": latency_seconds,
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    "finish_reason": data["choices"][0].get("finish_reason"),
                    "attempt": attempt,
                }
            except Exception as exc:
                last_error = repr(exc)
                await asyncio.sleep(min(2**attempt, 5))

        raise RuntimeError(f"Request failed after {args.retries} attempts: {last_error}")


async def benchmark_async(records: list[dict], args: argparse.Namespace, base_url: str) -> list[dict]:
    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = httpx.Timeout(connect=10.0, read=args.request_timeout, write=30.0, pool=30.0)
    limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)
    results: list[dict] = []

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        tasks = [asyncio.create_task(request_one(client, semaphore, record, args, base_url)) for record in records]
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress:
            task_id = progress.add_task("Running async chat completions", total=len(tasks))
            for future in asyncio.as_completed(tasks):
                result = await future
                results.append(result)
                progress.update(task_id, advance=1)

    results.sort(key=lambda item: item.get("source_index", 0))
    return results


def build_summary(results: list[dict], total_elapsed: float, args: argparse.Namespace, run_dir: Path) -> dict:
    latencies = [item["latency_seconds"] for item in results]
    prompt_tokens = sum(item.get("prompt_tokens", 0) for item in results)
    completion_tokens = sum(item.get("completion_tokens", 0) for item in results)
    total_tokens = sum(item.get("total_tokens", 0) for item in results)
    exact_matches = sum(1 for item in results if item.get("is_correct"))
    return {
        "run_dir": str(run_dir),
        "model_path": args.model_path,
        "served_model_name": args.served_model_name,
        "input": args.input,
        "total_requests": len(results),
        "concurrency": args.concurrency,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "total_elapsed_seconds": total_elapsed,
        "requests_per_second": 0.0 if total_elapsed <= 0 else len(results) / total_elapsed,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "completion_tokens_per_second": 0.0 if total_elapsed <= 0 else completion_tokens / total_elapsed,
        "total_tokens_per_second": 0.0 if total_elapsed <= 0 else total_tokens / total_elapsed,
        "exact_match_count": exact_matches,
        "exact_match_rate": 0.0 if not results else exact_matches / len(results),
        "latency_avg_seconds": statistics.mean(latencies) if latencies else 0.0,
        "latency_p50_seconds": percentile(latencies, 0.50),
        "latency_p95_seconds": percentile(latencies, 0.95),
        "latency_p99_seconds": percentile(latencies, 0.99),
    }


def render_config(args: argparse.Namespace, run_dir: Path) -> None:
    table = Table(title="Qwen3-0.6B Async vLLM Benchmark", box=box.SIMPLE_HEAVY)
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Model", args.model_path)
    table.add_row("Input", args.input)
    table.add_row("Run Dir", str(run_dir))
    table.add_row("Samples", str(args.max_samples))
    table.add_row("Concurrency", str(args.concurrency))
    table.add_row("Max Tokens", str(args.max_tokens))
    table.add_row("GPU Memory Util", str(args.gpu_memory_utilization))
    table.add_row("Port", str(args.port))
    console.print(table)


def render_summary(summary: dict) -> None:
    table = Table(title="Performance Summary", box=box.SIMPLE_HEAVY)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    rows = [
        ("Requests", str(summary["total_requests"])),
        ("Elapsed (s)", f"{summary['total_elapsed_seconds']:.2f}"),
        ("Req/s", f"{summary['requests_per_second']:.2f}"),
        ("Prompt Tokens", str(summary["prompt_tokens"])),
        ("Completion Tokens", str(summary["completion_tokens"])),
        ("Completion Tok/s", f"{summary['completion_tokens_per_second']:.2f}"),
        ("Exact Match", f"{summary['exact_match_count']} / {summary['total_requests']}"),
        ("Exact Match Rate", f"{summary['exact_match_rate']:.2%}"),
        ("Latency Avg (s)", f"{summary['latency_avg_seconds']:.2f}"),
        ("Latency P50 (s)", f"{summary['latency_p50_seconds']:.2f}"),
        ("Latency P95 (s)", f"{summary['latency_p95_seconds']:.2f}"),
        ("Latency P99 (s)", f"{summary['latency_p99_seconds']:.2f}"),
    ]
    for metric, value in rows:
        table.add_row(metric, value)
    console.print(table)


def render_samples(results: list[dict], display_count: int) -> None:
    if not results or display_count <= 0:
        return
    console.print(Panel.fit(f"Rendering {min(display_count, len(results))} sample outputs", style="green"))
    for index, record in enumerate(results[:display_count], start=1):
        correctness = "MATCH" if record.get("is_correct") else "MISS"
        title = f"Sample {index} | source_index={record.get('source_index')} | {correctness}"
        body = (
            f"[bold cyan]System Prompt[/bold cyan]\n{record['system_prompt']}\n\n"
            f"[bold cyan]Input[/bold cyan]\n{record['question']}\n\n"
            f"[bold cyan]Output[/bold cyan]\n{record['generated_text']}\n\n"
            f"[bold cyan]Detected Answer[/bold cyan]\n{record['predicted_answer'] or '<empty>'}\n\n"
            f"[bold cyan]Final Answer[/bold cyan]\n{record['final_answer'] or '<empty>'}\n\n"
            f"[bold cyan]Latency[/bold cyan] {record['latency_seconds']:.2f}s | "
            f"[bold cyan]Completion Tokens[/bold cyan] {record.get('completion_tokens', 0)}"
        )
        style = "green" if record.get("is_correct") else "yellow"
        console.print(Panel(body, title=title, border_style=style))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    model_path = Path(args.model_path).resolve()
    input_path = Path(args.input).resolve()
    output_root = Path(args.output_root).resolve()
    run_dir = output_root / f"vllm_qwen3_06b_async_{utc_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(input_path)[: args.max_samples]
    if not records:
        raise ValueError(f"No input records found: {input_path}")

    outputs_path = run_dir / "outputs.jsonl"
    summary_path = run_dir / "summary.json"
    render_config(args, run_dir)

    base_url = f"http://127.0.0.1:{args.port}"
    server_process, server_log_path = launch_server(args, run_dir)
    started_at = time.perf_counter()
    try:
        wait_for_server(base_url, server_process, server_log_path, args.server_ready_timeout)
        results = asyncio.run(benchmark_async(records, args, base_url))
        total_elapsed = time.perf_counter() - started_at
        write_jsonl(outputs_path, results)
        summary = build_summary(results, total_elapsed, args, run_dir)
        summary["server_log"] = str(server_log_path)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        render_summary(summary)
        render_samples(results, args.display_samples)
        console.print(Panel.fit(f"Saved outputs to {outputs_path}\nSaved summary to {summary_path}", style="blue"))
    finally:
        if not args.keep_server:
            terminate_server(server_process)


if __name__ == "__main__":
    main()
