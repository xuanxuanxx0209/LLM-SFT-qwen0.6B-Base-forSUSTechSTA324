#!/bin/bash
set -euo pipefail

MODEL_PATH="/home/ubuntu/Qwen3-0.6B-Base-Math-SFT-v2/checkpoints_sft/global_step1000_hf"
DATASET_PATH="/home/ubuntu/jgy-dataset/combined_7460_merged.jsonl"
OUTPUT_PATH="/home/ubuntu/Qwen3-0.6B-Base-Math-SFT-v2"
TEMPLATE_PATH="/home/ubuntu/models/Qwen3-0.6B-Base/chat_template.jinja"
TB_LOG_DIR="${OUTPUT_PATH}/tensorboard"
LOG_FILE="${OUTPUT_PATH}/training.log"

mkdir -p "${OUTPUT_PATH}"
mkdir -p "${TB_LOG_DIR}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm

CHAT_TEMPLATE=$(cat "${TEMPLATE_PATH}")

export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29501
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

exec python -m openrlhf.cli.train_sft \
    --pretrain "${MODEL_PATH}" \
    --dataset "${DATASET_PATH}" \
    --apply_chat_template \
    --input_key messages \
    --tokenizer_chat_template "${CHAT_TEMPLATE}" \
    --max_len 4096 \
    --micro_train_batch_size 2 \
    --train_batch_size 16 \
    --learning_rate 1e-7 \
    --max_epochs 3 \
    --save_path "${OUTPUT_PATH}" \
    --save_steps 500 \
    --save_hf_ckpt \
    --disable_ds_ckpt \
    --max_ckpt_num 2 \
    --ckpt_path "${OUTPUT_PATH}/checkpoints_sft" \
    --logging_steps 1 \
    --gradient_checkpointing \
    --zero_stage 2 \
    --param_dtype bf16 \
    --attn_implementation flash_attention_2 \
    --use_tensorboard "${TB_LOG_DIR}" \
    >> "${LOG_FILE}" 2>&1
