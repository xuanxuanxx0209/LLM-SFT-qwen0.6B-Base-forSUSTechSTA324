#!/usr/bin/env python3
"""LoRA fine-tune Qwen3-0.6B-mathsft-V3 on critical-thinking weakness data.

Output: /home/ubuntu/qwen3-0.6B-mathsft-V3-lora-r16/  (LoRA adapter)
Then run merge_lora.py to produce the standalone safetensors model.
"""
import os
import subprocess
import sys
from pathlib import Path

PYTHON_BIN = "/home/ubuntu/miniconda3/envs/llm/bin/python"
DEEPSPEED_BIN = "/home/ubuntu/miniconda3/envs/llm/bin/deepspeed"

MODEL_PATH = "/home/ubuntu/qwen3-0.6B-mathsft-V3"
DATASET_PATH = "/home/ubuntu/jgy-dataset/lora_weakness_train_v2.jsonl"
SAVE_PATH = "/home/ubuntu/qwen3-0.6B-mathsft-V3-lora-r8-lr2.5e-5"

# Same chat template as run_math_sft.py (preserves <think> tags)
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message.role == 'system' %}<|im_start|>system\n{{ message.content }}<|im_end|>\n"
    "{% elif message.role == 'user' %}<|im_start|>user\n{{ message.content }}<|im_end|>\n"
    "{% elif message.role == 'assistant' %}<|im_start|>assistant\n{{ message.content }}<|im_end|>\n"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

# Hyperparameters (LoRA v2: milder than v1 — preserves non-target categories)
LR = 2.5e-5                  # v6: half of v2 (5e-5); LR ablation
TRAIN_BATCH_SIZE = 16
MICRO_BATCH_SIZE = 4         # smaller than full SFT (8) — adapter backward adds memory
GRADIENT_ACCUMULATION = 4    # 4 * 4 = 16
MAX_EPOCHS = 2               # v5: same as v2 (rank ablation)
MAX_LEN = 4096
ZERO_STAGE = 2
MASTER_PORT = 29669          # v6: bump from v5 (29668)

# LoRA config (smaller capacity than v1)
LORA_RANK = 8
LORA_ALPHA = 16              # alpha = 2 × rank
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",      # attention
    "gate_proj", "up_proj", "down_proj",          # MLP
]  # explicitly excludes lm_head and embed_tokens


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
        "--save_steps", "-1",
        "--disable_ds_ckpt",
        "--logging_steps", "1",
        "--zero_stage", str(ZERO_STAGE),
        "--learning_rate", str(LR),
        "--lr_warmup_ratio", "0.05",
        "--gradient_checkpointing",
        "--attn_implementation", "eager",
        "--use_tensorboard", os.path.join(SAVE_PATH, "tensorboard"),
        "--tokenizer_chat_template", CHAT_TEMPLATE,
        # LoRA flags
        "--lora_rank", str(LORA_RANK),
        "--lora_alpha", str(LORA_ALPHA),
        "--lora_dropout", str(LORA_DROPOUT),
        "--target_modules", *TARGET_MODULES,
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    conda_bin = "/home/ubuntu/miniconda3/envs/llm/bin"
    env["PATH"] = conda_bin + os.pathsep + env.get("PATH", "")
    cache_dir = Path("/tmp/openrlhf_cache_lora")
    cache_dir.mkdir(parents=True, exist_ok=True)
    env.setdefault("TMPDIR", str(cache_dir / "tmp"))
    env.setdefault("HF_HOME", str(cache_dir / "hf"))
    env.setdefault("HF_DATASETS_CACHE", str(cache_dir / "datasets"))
    env.setdefault("TRANSFORMERS_CACHE", str(cache_dir / "transformers"))
    env.setdefault("TORCH_HOME", str(cache_dir / "torch"))

    print("=" * 60)
    print("OpenRLHF LoRA Fine-tuning (continue from V3)")
    print("=" * 60)
    print(f"Base model: {MODEL_PATH}")
    print(f"Dataset:    {DATASET_PATH}")
    print(f"Save path:  {SAVE_PATH}")
    print(f"LR:         {LR}")
    print(f"Batch:      {TRAIN_BATCH_SIZE} (micro={MICRO_BATCH_SIZE}, accum={GRADIENT_ACCUMULATION})")
    print(f"Epochs:     {MAX_EPOCHS}")
    print(f"LoRA:       rank={LORA_RANK}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
    print(f"Targets:    {' '.join(TARGET_MODULES)}")
    print("=" * 60)

    proc = subprocess.Popen(cmd, env=env)
    sys.exit(proc.wait())


if __name__ == "__main__":
    main()
