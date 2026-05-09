#!/usr/bin/env python3
"""Generate GRPO loss/reward/validation-accuracy artifacts from run outputs."""

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
LOSS_TAG_CANDIDATES = ["train/policy_loss", "train/actor_loss", "train/loss_mean"]
REWARD_TAG_CANDIDATES = ["train/exact_match", "train/score", "train/reward"]
LR_TAG_CANDIDATES = ["train/actor_lr", "train/lr"]


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


def infer_epoch(global_step: int | None, optimizer_steps_per_epoch: int) -> int | None:
    if global_step is None or optimizer_steps_per_epoch <= 0:
        return None
    if global_step <= 0:
        return None
    return max(1, (global_step + optimizer_steps_per_epoch - 1) // optimizer_steps_per_epoch)


def first_available_tag(tags: set[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in tags:
            return candidate
    return None


def read_scalar_history(tb_run_dir: Path, tag: str) -> list[dict]:
    accumulator = EventAccumulator(str(tb_run_dir))
    accumulator.Reload()
    scalar_events = accumulator.Scalars(tag)
    return [{"step": int(event.step), "value": float(event.value)} for event in scalar_events]


def read_training_history(tb_run_dir: Path) -> tuple[list[dict], dict]:
    accumulator = EventAccumulator(str(tb_run_dir))
    accumulator.Reload()
    scalar_tags = set(accumulator.Tags().get("scalars", []))

    loss_tag = first_available_tag(scalar_tags, LOSS_TAG_CANDIDATES)
    reward_tag = first_available_tag(scalar_tags, REWARD_TAG_CANDIDATES)
    lr_tag = first_available_tag(scalar_tags, LR_TAG_CANDIDATES)

    loss_events = accumulator.Scalars(loss_tag) if loss_tag else []
    reward_map = {event.step: event.value for event in accumulator.Scalars(reward_tag)} if reward_tag else {}
    lr_map = {event.step: event.value for event in accumulator.Scalars(lr_tag)} if lr_tag else {}

    records = []
    for event in loss_events:
        records.append(
            {
                "step": int(event.step),
                "loss": float(event.value),
                "reward": float(reward_map.get(event.step, 0.0)),
                "lr": float(lr_map.get(event.step, 0.0)),
            }
        )

    metadata = {
        "loss_tag": loss_tag,
        "reward_tag": reward_tag,
        "lr_tag": lr_tag,
        "scalar_tags": sorted(scalar_tags),
    }
    return records, metadata


def discover_accuracy_history(eval_root: Path, manifest: dict) -> list[dict]:
    records = []
    if not eval_root.exists():
        return records

    total_optimizer_steps = manifest.get("total_optimizer_steps")
    optimizer_steps_per_epoch = int(manifest.get("optimizer_steps_per_epoch", 0) or 0)
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
        summary["epoch"] = infer_epoch(global_step, optimizer_steps_per_epoch)
        records.append(summary)

    records.sort(key=lambda item: (item["global_step"] is None, item["global_step"] or 0, item["label"]))
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_training_markdown(records: list[dict], metadata: dict) -> str:
    lines = [
        "# GRPO Training History",
        "",
        f"- loss_tag: `{metadata.get('loss_tag')}`",
        f"- reward_tag: `{metadata.get('reward_tag')}`",
        f"- lr_tag: `{metadata.get('lr_tag')}`",
        "",
        "| step | loss | reward | learning_rate |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        lines.append(
            "| {step} | {loss:.6f} | {reward:.6f} | {lr:.8f} |".format(
                step=record["step"],
                loss=record["loss"],
                reward=record["reward"],
                lr=record["lr"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def build_accuracy_markdown(records: list[dict]) -> str:
    lines = [
        "# Validation Accuracy History",
        "",
        "| label | epoch | global_step | accuracy | correct | total | elapsed_seconds | output_toks_per_s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        epoch = "" if record.get("epoch") is None else record["epoch"]
        step = "" if record["global_step"] is None else record["global_step"]
        lines.append(
            "| {label} | {epoch} | {step} | {acc:.4%} | {correct} | {total} | {elapsed:.2f} | {tps:.2f} |".format(
                label=record["label"],
                epoch=epoch,
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

    training_records, metadata = read_training_history(tb_run_dir)
    accuracy_records = discover_accuracy_history(eval_root, manifest)

    training_jsonl = run_root / "training_loss_history.jsonl"
    training_md = run_root / "training_loss_history.md"
    training_plot = run_root / "training_loss_curve"
    write_jsonl(training_jsonl, training_records)
    training_md.write_text(build_training_markdown(training_records, metadata), encoding="utf-8")
    save_line_plot(
        x_values=[record["step"] for record in training_records],
        y_values=[record["loss"] for record in training_records],
        title="OpenRLHF GRPO Policy Loss",
        ylabel="Policy Loss",
        output_base=training_plot,
    )

    reward_jsonl = run_root / "training_reward_history.jsonl"
    reward_md = run_root / "training_reward_history.md"
    reward_plot = run_root / "training_reward_curve"
    reward_records = [{"step": record["step"], "reward": record["reward"]} for record in training_records]
    write_jsonl(reward_jsonl, reward_records)
    reward_md.write_text(
        "\n".join(
            [
                "# GRPO Reward History",
                "",
                f"- reward_tag: `{metadata.get('reward_tag')}`",
                "",
                "| step | reward |",
                "| ---: | ---: |",
                *[f"| {record['step']} | {record['reward']:.6f} |" for record in reward_records],
                "",
            ]
        ),
        encoding="utf-8",
    )
    save_line_plot(
        x_values=[record["step"] for record in reward_records],
        y_values=[record["reward"] for record in reward_records],
        title="OpenRLHF GRPO Training Reward",
        ylabel="Reward / Exact Match",
        output_base=reward_plot,
    )

    accuracy_jsonl = run_root / "validation_accuracy_history.jsonl"
    accuracy_md = run_root / "validation_accuracy_history.md"
    accuracy_plot = run_root / "validation_accuracy_curve"
    write_jsonl(accuracy_jsonl, accuracy_records)
    accuracy_md.write_text(build_accuracy_markdown(accuracy_records), encoding="utf-8")
    save_line_plot(
        x_values=[record["global_step"] for record in accuracy_records if record["global_step"] is not None],
        y_values=[record["accuracy"] * 100 for record in accuracy_records if record["global_step"] is not None],
        title="OpenRLHF GRPO Validation Accuracy",
        ylabel="Validation Accuracy (%)",
        output_base=accuracy_plot,
    )

    metadata_path = run_root / "training_scalar_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "tensorboard_run_dir": str(tb_run_dir),
                "eval_root": str(eval_root),
                "training_records": len(training_records),
                "accuracy_records": len(accuracy_records),
                "training_jsonl": str(training_jsonl),
                "training_md": str(training_md),
                "training_pdf": str(training_plot.with_suffix(".pdf")),
                "training_png": str(training_plot.with_suffix(".png")),
                "reward_jsonl": str(reward_jsonl),
                "reward_md": str(reward_md),
                "reward_pdf": str(reward_plot.with_suffix(".pdf")),
                "reward_png": str(reward_plot.with_suffix(".png")),
                "accuracy_jsonl": str(accuracy_jsonl),
                "accuracy_md": str(accuracy_md),
                "accuracy_pdf": str(accuracy_plot.with_suffix(".pdf")),
                "accuracy_png": str(accuracy_plot.with_suffix(".png")),
                "metadata_json": str(metadata_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
