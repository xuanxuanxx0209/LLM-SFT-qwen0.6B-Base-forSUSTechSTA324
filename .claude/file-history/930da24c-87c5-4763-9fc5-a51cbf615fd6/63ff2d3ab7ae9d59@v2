#!/usr/bin/env python3
"""Launch OpenRLHF SFT for Qwen3-0.6B math fine-tuning."""

import os
import subprocess
import sys
from pathlib import Path

PYTHON_BIN = "/home/ubuntu/miniconda3/envs/llm/bin/python"
DEEPSPEED_BIN = "/home/ubuntu/miniconda3/envs/llm/bin/deepspeed"

MODEL_PATH = "/home/ubuntu/models/Qwen3-0.6B-Base"
DATASET_PATH = "/home/ubuntu/jgy-dataset/math_sft_think_boxed.jsonl"
SAVE_PATH = "/home/ubuntu/Qwen3-0.6B-Math-SFT"

# Custom chat template that preserves <think> tags in assistant output
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message.role == 'system' %}<|im_start|>system\n{{ message.content }}<|im_end|>\n"
    "{% elif message.role == 'user' %}<|im_start|>user\n{{ message.content }}<|im_end|>\n"
    "{% elif message.role == 'assistant' %}<|im_start|>assistant\n{{ message.content }}<|im_end|>\n"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

# Hyperparameters
LR = 5e-6
TRAIN_BATCH_SIZE = 16
MICRO_BATCH_SIZE = 8
GRADIENT_ACCUMULATION = 2  # 8 * 2 = 16
MAX_EPOCHS = 3
MAX_LEN = 4096
ZERO_STAGE = 2
MASTER_PORT = 29663

def main():
    os.makedirs(SAVE_PATH, exist_ok=True)

    cmd = [
        DEEPSPEED_BIN,
        "--master_port", str(MASTER_PORT),
        "--module", "openrlhf.cli.train_sft",
        "--pretrain", MODEL_PATH,
        "--dataset", DATASET_PATH,
        "--input_key", "messages",
        "--apply_chat_template",
        "--multiturn",
        "--max_len", str(MAX_LEN),
        "--max_epochs", str(MAX_EPOCHS),
        "--micro_train_batch_size", str(MICRO_BATCH_SIZE),
        "--train_batch_size", str(TRAIN_BATCH_SIZE),
        "--save_path", SAVE_PATH,
        "--save_steps", "-1",           # no intermediate checkpoints
        "--disable_ds_ckpt",             # disable deepspeed checkpointing
        "--logging_steps", "1",
        "--zero_stage", str(ZERO_STAGE),
        "--learning_rate", str(LR),
        "--gradient_checkpointing",
        "--attn_implementation", "eager",
        "--use_tensorboard", os.path.join(SAVE_PATH, "tensorboard"),
        "--tokenizer_chat_template", CHAT_TEMPLATE,
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    # Ensure ninja is available for DeepSpeed JIT compilation
    conda_bin = "/home/ubuntu/miniconda3/envs/llm/bin"
    env["PATH"] = conda_bin + os.pathsep + env.get("PATH", "")
    # Cache dirs
    cache_dir = Path("/tmp/openrlhf_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    env.setdefault("TMPDIR", str(cache_dir / "tmp"))
    env.setdefault("HF_HOME", str(cache_dir / "hf"))
    env.setdefault("HF_DATASETS_CACHE", str(cache_dir / "datasets"))
    env.setdefault("TRANSFORMERS_CACHE", str(cache_dir / "transformers"))
    env.setdefault("TORCH_HOME", str(cache_dir / "torch"))

    print("=" * 60)
    print("OpenRLHF SFT Math Fine-tuning")
    print("=" * 60)
    print(f"Model:      {MODEL_PATH}")
    print(f"Dataset:    {DATASET_PATH}")
    print(f"Save path:  {SAVE_PATH}")
    print(f"LR:         {LR}")
    print(f"Batch size: {TRAIN_BATCH_SIZE} (micro={MICRO_BATCH_SIZE}, grad_accum={GRADIENT_ACCUMULATION})")
    print(f"Epochs:     {MAX_EPOCHS}")
    print(f"Max len:    {MAX_LEN}")
    print("=" * 60)
    print(f"Command:\n{' '.join(cmd)}")
    print("=" * 60)

    proc = subprocess.Popen(cmd, env=env)
    return_code = proc.wait()
    sys.exit(return_code)


if __name__ == "__main__":
    main()
