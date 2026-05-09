#!/usr/bin/env python3
"""Generate SFT loss and validation accuracy artifacts from run outputs.

This helper parses:
- TensorBoard scalar events for training loss / learning rate
- validation `summary.json` files for checkpoint accuracy

Outputs:
- JSONL history files
- Markdown tables
- PDF and PNG line charts
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


STEP_PATTERN = re.compile(r"global_step(\d+)_hf$")


def find_tensorboard_run_dir(run_root: Path) -> Path | None:
    tensorboard_root = run_root / "tensorboard"
    if not tensorboard_root.exists():
        return None

    event_files = sorted(tensorboard_root.rglob("events.out.tfevents.*"))
    if not event_files:
        return None
    return event_files[-1].parent


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_run_manifest(run_root: Path) -> dict:
    manifest_path = run_root / "run_manifest.json"
    if not manifest_path.exists():
        return {}
    return load_json(manifest_path)


def read_loss_history(tb_run_dir: Path) -> list[dict]:
    accumulator = EventAccumulator(str(tb_run_dir))
    accumulator.Reload()

    tags = set(accumulator.Tags().get("scalars", []))
    if "train/loss_mean" not in tags:
        return []

    loss_mean_events = accumulator.Scalars("train/loss_mean")
    gpt_loss_events = (
        {event.step: event.value for event in accumulator.Scalars("train/gpt_loss")}
        if "train/gpt_loss" in tags
        else {}
    )
    lr_events = (
        {event.step: event.value for event in accumulator.Scalars("train/lr")}
        if "train/lr" in tags
        else {}
    )

    records = []
    for event in loss_mean_events:
        records.append(
            {
                "step": int(event.step),
                "loss_mean": float(event.value),
                "gpt_loss": float(gpt_loss_events.get(event.step, event.value)),
                "lr": float(lr_events.get(event.step, 0.0)),
            }
        )
    return records


def discover_accuracy_history(eval_root: Path, manifest: dict) -> list[dict]:
    records = []
    if not eval_root.exists():
        return records

    total_optimizer_steps = manifest.get("total_optimizer_steps")
    for summary_path in sorted(eval_root.glob("*/summary.json")):
        summary = load_json(summary_path)
        label = summary["label"]
        match = STEP_PATTERN.match(label)
        if match:
            global_step = int(match.group(1))
        elif total_optimizer_steps is not None:
            global_step = int(total_optimizer_steps)
        else:
            global_step = None
        summary["global_step"] = global_step
        records.append(summary)

    records.sort(key=lambda item: (item["global_step"] is None, item["global_step"] or 0, item["label"]))
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_loss_markdown(records: list[dict]) -> str:
    lines = [
        "# Training Loss History",
        "",
        "| step | loss_mean | gpt_loss | learning_rate |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        lines.append(
            "| {step} | {loss_mean:.6f} | {gpt_loss:.6f} | {lr:.8f} |".format(
                step=record["step"],
                loss_mean=record["loss_mean"],
                gpt_loss=record["gpt_loss"],
                lr=record["lr"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def build_accuracy_markdown(records: list[dict]) -> str:
    lines = [
        "# Validation Accuracy History",
        "",
        "| label | global_step | accuracy | correct | total | elapsed_seconds | output_toks_per_s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        step = "" if record["global_step"] is None else record["global_step"]
        lines.append(
            "| {label} | {step} | {acc:.4%} | {correct} | {total} | {elapsed:.2f} | {tps:.2f} |".format(
                label=record["label"],
                step=step,
                acc=record["accuracy"],
                correct=record["correct"],
                total=record["total"],
                elapsed=record["elapsed_seconds"],
                tps=record["output_tokens_per_second"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def save_line_plot(
    x_values: list[int],
    y_values: list[float],
    title: str,
    ylabel: str,
    output_base: Path,
) -> None:
    if plt is None or not x_values:
        return

    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, marker="o", linewidth=2, markersize=5, color="#1f77b4")
    plt.title(title)
    plt.xlabel("Global Step")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_base.with_suffix(".pdf"), format="pdf")
    plt.savefig(output_base.with_suffix(".png"), format="png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--eval-root", default=None)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    eval_root = Path(args.eval_root).resolve() if args.eval_root else run_root / "eval_valid1000_async_b64"

    manifest = load_run_manifest(run_root)
    tb_run_dir = find_tensorboard_run_dir(run_root)

    if tb_run_dir is None:
        raise FileNotFoundError(f"No TensorBoard event file was found under {run_root / 'tensorboard'}.")

    loss_records = read_loss_history(tb_run_dir)
    accuracy_records = discover_accuracy_history(eval_root, manifest)

    loss_jsonl = run_root / "training_loss_history.jsonl"
    loss_md = run_root / "training_loss_history.md"
    loss_plot = run_root / "training_loss_curve"
    write_jsonl(loss_jsonl, loss_records)
    loss_md.write_text(build_loss_markdown(loss_records), encoding="utf-8")
    save_line_plot(
        x_values=[record["step"] for record in loss_records],
        y_values=[record["loss_mean"] for record in loss_records],
        title="OpenRLHF SFT Training Loss",
        ylabel="Loss Mean",
        output_base=loss_plot,
    )

    accuracy_jsonl = run_root / "validation_accuracy_history.jsonl"
    accuracy_md = run_root / "validation_accuracy_history.md"
    accuracy_plot = run_root / "validation_accuracy_curve"
    write_jsonl(accuracy_jsonl, accuracy_records)
    accuracy_md.write_text(build_accuracy_markdown(accuracy_records), encoding="utf-8")
    save_line_plot(
        x_values=[record["global_step"] for record in accuracy_records if record["global_step"] is not None],
        y_values=[record["accuracy"] * 100 for record in accuracy_records if record["global_step"] is not None],
        title="OpenRLHF SFT Validation Accuracy",
        ylabel="Validation Accuracy (%)",
        output_base=accuracy_plot,
    )

    print(json.dumps(
        {
            "tensorboard_run_dir": str(tb_run_dir),
            "eval_root": str(eval_root),
            "loss_records": len(loss_records),
            "accuracy_records": len(accuracy_records),
            "loss_jsonl": str(loss_jsonl),
            "loss_md": str(loss_md),
            "loss_pdf": str(loss_plot.with_suffix(".pdf")),
            "loss_png": str(loss_plot.with_suffix(".png")),
            "accuracy_jsonl": str(accuracy_jsonl),
            "accuracy_md": str(accuracy_md),
            "accuracy_pdf": str(accuracy_plot.with_suffix(".pdf")),
            "accuracy_png": str(accuracy_plot.with_suffix(".png")),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
