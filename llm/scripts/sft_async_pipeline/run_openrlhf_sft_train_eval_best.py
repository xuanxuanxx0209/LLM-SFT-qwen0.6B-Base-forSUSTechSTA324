#!/usr/bin/env python3
"""Train OpenRLHF SFT, evaluate checkpoints, and promote the best model.

This script is designed to be practical for single-GPU workflows:
- train a model with OpenRLHF SFT
- save periodic HF checkpoints
- evaluate checkpoints plus the raw training-final model on valid_1000
- use async vLLM scheduling by default
- generate a Markdown report in English
- copy the best validation model into a final model directory

It also supports `--skip-training` so an interrupted evaluation run can be
continued on an existing training output directory.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
COMMON_SCRIPT_DIR = PROJECT_ROOT / "scripts"
if str(COMMON_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_SCRIPT_DIR))

from path_utils import infer_default_model_path, resolve_llm_env_executable, runtime_cache_root, runtime_data_root

STEP_PATTERN = re.compile(r"global_step(\d+)_hf$")
MODEL_CANDIDATE_NAMES = ("Qwen3-0.6B-Base", "Qwen3-0.6B-BASE", "Qwen3-0.6B")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def read_chat_template(path: str | None) -> str | None:
    if path is None:
        return None
    return Path(path).read_text(encoding="utf-8")


def resolve_python_bin() -> str:
    return resolve_llm_env_executable("python")


def resolve_curve_python_bin() -> str:
    candidates: list[str] = []
    system_python = shutil.which("python3")
    if system_python:
        candidates.append(system_python)
    candidates.append(resolve_python_bin())

    checked = set()
    for candidate in candidates:
        resolved = str(Path(candidate).resolve()) if Path(candidate).exists() else candidate
        if resolved in checked:
            continue
        checked.add(resolved)
        probe = subprocess.run(
            [candidate, "-c", "import matplotlib; import tensorboard"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if probe.returncode == 0:
            return candidate
    return resolve_python_bin()


def resolve_deepspeed_bin() -> str:
    return resolve_llm_env_executable("deepspeed")


def discover_checkpoints(ckpt_dir: Path) -> list[tuple[int, Path]]:
    checkpoints = []
    if not ckpt_dir.exists():
        return checkpoints
    for path in ckpt_dir.iterdir():
        if not path.is_dir():
            continue
        match = STEP_PATTERN.match(path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))
    checkpoints.sort(key=lambda item: item[0])
    return checkpoints


def compute_save_steps(
    num_train_records: int,
    micro_train_batch_size: int,
    gradient_accumulation: int,
    max_epochs: int,
    save_per_epoch: bool,
    target_mid_checkpoints: int,
) -> tuple[int, int, int, int]:
    train_batch_size = micro_train_batch_size * gradient_accumulation
    optimizer_steps_per_epoch = max(1, num_train_records // train_batch_size)
    total_optimizer_steps = optimizer_steps_per_epoch * max_epochs
    if save_per_epoch:
        save_steps = optimizer_steps_per_epoch
        max_ckpt_num = max_epochs
    elif target_mid_checkpoints <= 0:
        save_steps = total_optimizer_steps + 1
        max_ckpt_num = 1
    else:
        save_steps = max(1, total_optimizer_steps // target_mid_checkpoints)
        max_ckpt_num = max(target_mid_checkpoints, 1)
    return train_batch_size, optimizer_steps_per_epoch, total_optimizer_steps, save_steps, max_ckpt_num


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_command(
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    description: str,
) -> None:
    log(f"Starting: {description}")
    log(f"Command: {' '.join(cmd)}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return_code = proc.wait()
    if return_code != 0:
        raise RuntimeError(f"{description} failed with return code {return_code}. See {log_path}.")
    log(f"Completed: {description}")


def generate_curve_artifacts(run_root: Path, eval_root: Path) -> None:
    python_bin = resolve_curve_python_bin()
    curve_script = SCRIPT_DIR / "generate_sft_curves.py"
    if not curve_script.exists():
        log(f"Curve helper script was not found, skipping curve generation: {curve_script}")
        return

    curve_cmd = [
        python_bin,
        str(curve_script),
        "--run-root",
        str(run_root),
        "--eval-root",
        str(eval_root),
    ]
    curve_log_path = run_root / "generate_curves.log"
    curve_env = os.environ.copy()
    run_command(
        cmd=curve_cmd,
        cwd=PROJECT_ROOT,
        env=curve_env,
        log_path=curve_log_path,
        description="curve generation from TensorBoard and validation summaries",
    )


def run_training(args: argparse.Namespace, run_root: Path, manifest: dict) -> Path:
    deepspeed_bin = resolve_deepspeed_bin()
    tensorboard_root = run_root / "tensorboard"
    tensorboard_root.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_root / "checkpoints_sft"
    raw_final_dir = run_root / "training_final_model"
    train_log_path = run_root / "train.log"

    chat_template_text = read_chat_template(args.chat_template_file)
    train_cmd = [
        deepspeed_bin,
        "--master_port",
        str(args.master_port),
        "--module",
        "openrlhf.cli.train_sft",
        "--pretrain",
        args.model_path,
        "--dataset",
        args.train_dataset,
        "--input_key",
        "messages",
        "--apply_chat_template",
        "--multiturn",
        "--max_len",
        str(args.max_len),
        "--max_epochs",
        str(args.max_epochs),
        "--micro_train_batch_size",
        str(args.micro_train_batch_size),
        "--train_batch_size",
        str(manifest["train_batch_size"]),
        "--save_path",
        str(raw_final_dir),
        "--ckpt_path",
        str(ckpt_dir),
        "--save_steps",
        str(manifest["save_steps"]),
        "--save_hf_ckpt",
        "--disable_ds_ckpt",
        "--max_ckpt_num",
        str(manifest["max_ckpt_num"]),
        "--logging_steps",
        "1",
        "--zero_stage",
        str(args.zero_stage),
        "--learning_rate",
        str(args.learning_rate),
        "--gradient_checkpointing",
        "--attn_implementation",
        args.attn_implementation,
        "--use_tensorboard",
        str(tensorboard_root),
    ]
    if args.adam_offload:
        train_cmd.append("--adam_offload")
    if args.tb_run_name:
        train_cmd.extend(["--wandb_run_name", args.tb_run_name])
    if chat_template_text is not None:
        train_cmd.extend(["--tokenizer_chat_template", chat_template_text])

    train_env = os.environ.copy()
    train_env["CUDA_VISIBLE_DEVICES"] = args.train_gpu
    train_env["PYTHONPATH"] = str(PROJECT_ROOT) + (
        os.pathsep + train_env["PYTHONPATH"] if train_env.get("PYTHONPATH") else ""
    )
    cache_root = runtime_cache_root(PROJECT_ROOT)
    train_env.setdefault("TMPDIR", str(cache_root / "tmp"))
    train_env.setdefault("HF_HOME", str(cache_root / "hf"))
    train_env.setdefault("HF_DATASETS_CACHE", str(cache_root / "datasets"))
    train_env.setdefault("TRANSFORMERS_CACHE", str(cache_root / "transformers"))
    train_env.setdefault("TORCH_HOME", str(cache_root / "torch"))
    train_env.setdefault("TRITON_CACHE_DIR", str(cache_root / "triton"))
    train_env.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    for key in ["TMPDIR", "HF_HOME", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE", "TORCH_HOME", "TRITON_CACHE_DIR", "XDG_CACHE_HOME"]:
        Path(train_env[key]).mkdir(parents=True, exist_ok=True)
    train_env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    run_command(
        cmd=train_cmd,
        cwd=PROJECT_ROOT,
        env=train_env,
        log_path=train_log_path,
        description="OpenRLHF SFT training",
    )
    return raw_final_dir


def evaluate_one_model(
    model_path: Path,
    label: str,
    eval_root: Path,
    args: argparse.Namespace,
) -> dict:
    python_bin = resolve_python_bin()
    eval_cmd = [
        python_bin,
        str(SCRIPT_DIR / "eval_sft_checkpoint.py"),
        "--model-path",
        str(model_path),
        "--validation-input",
        args.validation_input,
        "--output-root",
        str(eval_root),
        "--label",
        label,
        "--engine",
        "vllm",
        "--num-workers",
        "1",
        "--device-ids",
        args.eval_device_id,
        "--batch-size",
        str(args.eval_batch_size),
        "--limit",
        str(args.validation_limit),
        "--requested-max-new-tokens",
        str(args.eval_max_new_tokens),
        "--chat-template-file",
        args.chat_template_file,
        "--vllm-attention-backend",
        args.eval_vllm_attention_backend,
        "--vllm-scheduler-mode",
        args.eval_vllm_scheduler_mode,
    ]
    eval_log_path = eval_root / f"{label}.launcher.log"
    eval_env = os.environ.copy()
    eval_env.setdefault("TMPDIR", str(runtime_cache_root(PROJECT_ROOT) / "tmp"))
    Path(eval_env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    run_command(
        cmd=eval_cmd,
        cwd=PROJECT_ROOT,
        env=eval_env,
        log_path=eval_log_path,
        description=f"validation evaluation for {label}",
    )
    summary_path = eval_root / label / "summary.json"
    summary = load_json(summary_path)
    summary["label"] = label
    summary["source_model_path"] = str(model_path)
    return summary


def copy_best_model(source_dir: Path, target_dir: Path) -> None:
    if source_dir.resolve() == target_dir.resolve():
        return
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir)


def build_report_markdown(manifest: dict, evaluation_records: list[dict], best_record: dict, final_model_dir: Path) -> str:
    lines = [
        "# OpenRLHF SFT Training and Validation Report",
        "",
        "## Run Configuration",
        "",
        f"- model_path: `{manifest['model_path']}`",
        f"- train_dataset: `{manifest['train_dataset']}`",
        f"- validation_input: `{manifest['validation_input']}`",
        f"- max_epochs: `{manifest['max_epochs']}`",
        f"- micro_train_batch_size: `{manifest['micro_train_batch_size']}`",
        f"- gradient_accumulation: `{manifest['gradient_accumulation']}`",
        f"- train_batch_size: `{manifest['train_batch_size']}`",
        f"- max_len: `{manifest['max_len']}`",
        f"- learning_rate: `{manifest['learning_rate']}`",
        f"- checkpoint_strategy: `{manifest['checkpoint_strategy']}`",
        f"- target_mid_checkpoints: `{manifest['target_mid_checkpoints']}`",
        f"- save_steps: `{manifest['save_steps']}`",
        f"- eval_batch_size: `{manifest['eval_batch_size']}`",
        f"- eval_max_new_tokens: `{manifest['eval_max_new_tokens']}`",
        f"- eval_scheduler_mode: `{manifest['eval_vllm_scheduler_mode']}`",
        "",
        "## Validation Results",
        "",
        "| label | global_step | accuracy | correct | total | elapsed_seconds | output_toks_per_s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    records = sorted(
        evaluation_records,
        key=lambda item: (item.get("global_step") is None, item.get("global_step") or 0, item["label"]),
    )
    for record in records:
        step_value = "" if record.get("global_step") is None else str(record["global_step"])
        lines.append(
            "| {label} | {step} | {acc:.4%} | {correct} | {total} | {elapsed:.2f} | {tps:.2f} |".format(
                label=record["label"],
                step=step_value,
                acc=record["accuracy"],
                correct=record["correct"],
                total=record["total"],
                elapsed=record["elapsed_seconds"],
                tps=record["output_tokens_per_second"],
            )
        )
    lines.extend(
        [
            "",
            "## Best Validation Model",
            "",
            f"- selected_label: `{best_record['label']}`",
            f"- selected_source_path: `{best_record['source_model_path']}`",
            f"- selected_accuracy: `{best_record['accuracy']:.4%}`",
            f"- promoted_final_model_dir: `{final_model_dir}`",
            "",
            "## Notes",
            "",
            "- Validation was executed with async local scheduling (`server_async`).",
            "- The evaluation request concurrency was set from `eval_batch_size`.",
            f"- `{final_model_dir.name}` contains the best validation-performing HF checkpoint or raw training-final model.",
            "",
        ]
    )
    return "\n".join(lines)


def pick_best_record(evaluation_records: list[dict]) -> dict:
    def sort_key(record: dict) -> tuple[float, int]:
        step = record.get("global_step")
        normalized_step = -1 if step is None else step
        return (record["accuracy"], normalized_step)

    return max(evaluation_records, key=sort_key)


def build_parser() -> argparse.ArgumentParser:
    data_root = runtime_data_root(PROJECT_ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=str(infer_default_model_path(PROJECT_ROOT, MODEL_CANDIDATE_NAMES)))
    parser.add_argument(
        "--train-dataset",
        default=str(
            data_root / "deepseek-v3.2-speciale-openr1-math-3k.plus_distilled_corpus_400k_with_cot-filtered.system_prompt.sft_train.jsonl"
        ),
    )
    parser.add_argument("--validation-input", default=str(data_root / "valid_1000.jsonl"))
    parser.add_argument(
        "--chat-template-file",
        default=str(SCRIPT_DIR / "templates" / "qwen3_06b_base_eot_chat_template.jinja"),
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--skip-training", action="store_true", default=False)
    parser.add_argument("--max-len", type=int, default=4096)
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--micro-train-batch-size", type=int, default=3)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--save-per-epoch", action="store_true", default=True)
    parser.add_argument("--no-save-per-epoch", dest="save_per_epoch", action="store_false")
    parser.add_argument("--target-mid-checkpoints", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-7)
    parser.add_argument("--train-gpu", default="0")
    parser.add_argument("--eval-device-id", default="0")
    parser.add_argument("--master-port", type=int, default=29663)
    parser.add_argument("--validation-limit", type=int, default=1000)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--eval-max-new-tokens", type=int, default=4096)
    parser.add_argument(
        "--eval-vllm-attention-backend",
        default="auto",
        choices=["auto", "FLASH_ATTN", "TRITON_ATTN", "FLASHINFER", "FLEX_ATTENTION"],
    )
    parser.add_argument(
        "--eval-vllm-scheduler-mode",
        default="server_async",
        choices=["static_batch", "full_queue", "server_async"],
    )
    parser.add_argument("--tb-run-name", default="openrlhf_sft_train_eval_best")
    parser.add_argument("--zero-stage", type=int, default=2)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--adam-offload", action="store_true", default=True)
    parser.add_argument("--no-adam-offload", dest="adam_offload", action="store_false")
    parser.add_argument("--eval-output-dir-name", default="eval_valid1000_async_b64")
    parser.add_argument("--final-model-dir-name", default="best_final_model")
    parser.add_argument("--skip-curve-generation", action="store_true", default=False)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    eval_root = run_root / args.eval_output_dir_name
    eval_root.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model_path).resolve()
    train_dataset = Path(args.train_dataset).resolve()
    validation_input = Path(args.validation_input).resolve()

    num_train_records = count_lines(train_dataset)
    train_batch_size, optimizer_steps_per_epoch, total_optimizer_steps, save_steps, max_ckpt_num = compute_save_steps(
        num_train_records=num_train_records,
        micro_train_batch_size=args.micro_train_batch_size,
        gradient_accumulation=args.gradient_accumulation,
        max_epochs=args.max_epochs,
        save_per_epoch=args.save_per_epoch,
        target_mid_checkpoints=args.target_mid_checkpoints,
    )

    manifest = {
        "model_path": str(model_path),
        "train_dataset": str(train_dataset),
        "validation_input": str(validation_input),
        "chat_template_file": args.chat_template_file,
        "max_len": args.max_len,
        "max_epochs": args.max_epochs,
        "micro_train_batch_size": args.micro_train_batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "train_batch_size": train_batch_size,
        "learning_rate": args.learning_rate,
        "checkpoint_strategy": "per_epoch" if args.save_per_epoch else "target_mid_checkpoints",
        "target_mid_checkpoints": args.target_mid_checkpoints,
        "num_train_records": num_train_records,
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "total_optimizer_steps": total_optimizer_steps,
        "save_steps": save_steps,
        "max_ckpt_num": max_ckpt_num,
        "eval_batch_size": args.eval_batch_size,
        "eval_max_new_tokens": args.eval_max_new_tokens,
        "eval_vllm_attention_backend": args.eval_vllm_attention_backend,
        "eval_vllm_scheduler_mode": args.eval_vllm_scheduler_mode,
        "skip_training": args.skip_training,
        "run_root": str(run_root),
        "eval_root": str(eval_root),
        "final_model_dir_name": args.final_model_dir_name,
    }
    manifest_path = run_root / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    log("OpenRLHF train/eval orchestrator started.")
    log(f"Run root: {run_root}")
    log(f"Model path: {model_path}")
    log(f"Training dataset: {train_dataset}")
    log(f"Validation input: {validation_input}")
    log(
        "Evaluation defaults: engine=vllm, scheduler_mode="
        f"{args.eval_vllm_scheduler_mode}, batch_size={args.eval_batch_size}, max_new_tokens={args.eval_max_new_tokens}."
    )

    if args.skip_training:
        log("Skip-training mode is enabled. Existing checkpoints and model directories will be evaluated.")
        raw_final_dir = run_root / "training_final_model"
        raw_final_label = "training_final_model"
        if not raw_final_dir.exists():
            raw_final_dir = run_root / "final_model"
            raw_final_label = "final_model"
        if not raw_final_dir.exists():
            raw_final_dir = run_root / args.final_model_dir_name
            raw_final_label = args.final_model_dir_name
    else:
        raw_final_dir = run_training(args=args, run_root=run_root, manifest=manifest)
        raw_final_label = "training_final_model"

    ckpt_dir = run_root / "checkpoints_sft"
    checkpoints = discover_checkpoints(ckpt_dir)
    if not checkpoints:
        log("No periodic checkpoints were found.")
    else:
        log(f"Discovered {len(checkpoints)} checkpoint directories for evaluation.")

    evaluation_targets: list[tuple[str, Path, int | None]] = []
    for step, path in checkpoints:
        evaluation_targets.append((path.name, path, step))
    checkpoint_steps = {step for step, _path in checkpoints}

    if raw_final_dir.exists():
        final_step = total_optimizer_steps if not args.skip_training else total_optimizer_steps
        if final_step in checkpoint_steps and raw_final_label == "training_final_model":
            log(
                "The raw training-final model matches the last saved checkpoint step. "
                "Skipping duplicate validation for training_final_model."
            )
        else:
            evaluation_targets.append((raw_final_label, raw_final_dir, final_step))
    else:
        log(f"Raw training-final model directory not found: {raw_final_dir}")

    if not evaluation_targets:
        raise RuntimeError("No model targets were available for validation evaluation.")

    evaluation_records = []
    for label, path, step in evaluation_targets:
        summary = evaluate_one_model(model_path=path, label=label, eval_root=eval_root, args=args)
        summary["global_step"] = step
        evaluation_records.append(summary)
        log(
            f"Validation completed for {label}: accuracy={summary['accuracy']:.4%}, "
            f"correct={summary['correct']}/{summary['total']}."
        )

    best_record = pick_best_record(evaluation_records)
    best_source_path = Path(best_record["source_model_path"])
    selected_final_dir = run_root / args.final_model_dir_name
    copy_best_model(best_source_path, selected_final_dir)

    selection_metadata = {
        "selected_label": best_record["label"],
        "selected_source_model_path": best_record["source_model_path"],
        "selected_accuracy": best_record["accuracy"],
        "selected_global_step": best_record.get("global_step"),
        "selected_final_model_dir": str(selected_final_dir),
    }
    (run_root / "best_model_selection.json").write_text(
        json.dumps(selection_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    (run_root / "evaluation_summary.json").write_text(
        json.dumps(evaluation_records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path = run_root / "evaluation_report.md"
    report_path.write_text(
        build_report_markdown(
            manifest=manifest,
            evaluation_records=evaluation_records,
            best_record=best_record,
            final_model_dir=selected_final_dir,
        ),
        encoding="utf-8",
    )

    if args.skip_curve_generation:
        log("Curve generation was skipped by configuration.")
    else:
        generate_curve_artifacts(run_root=run_root, eval_root=eval_root)

    log("All evaluations are complete.")
    log(
        f"Best validation model: {best_record['label']} with accuracy {best_record['accuracy']:.4%}. "
        f"Promoted directory: {selected_final_dir}"
    )
    log(f"English report written to: {report_path}")
    if not args.skip_curve_generation:
        log(f"Curve artifacts were generated under: {run_root}")


if __name__ == "__main__":
    main()
