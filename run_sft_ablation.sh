#!/bin/bash
set -e

MODEL_PATH="/home/ubuntu/models/Qwen3-0.6B-Base"
DATASET="/home/ubuntu/jgy-dataset/sftdata_plus_general100_plus_critical_thinking100.jsonl"
TB_ROOT="/home/ubuntu/tb_combined"

DEEPSPEED_BIN="/home/ubuntu/miniconda3/envs/llm/bin/deepspeed"

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/home/ubuntu/llm:${PYTHONPATH}"
export PATH="/home/ubuntu/miniconda3/envs/llm/bin:${PATH}"
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export TMPDIR="/tmp"
export HF_HOME="/home/ubuntu/.cache/hf"
export HF_DATASETS_CACHE="/home/ubuntu/.cache/datasets"

# 公共训练参数（基于实测：所有样本 < 1743 tokens，4096 足够且更快）
# 尝试 micro=8 + grad_acc=2 => global=16，每 epoch 约 148 步，比 micro=4 快约 2 倍
COMMON_ARGS=(
    --pretrain "$MODEL_PATH"
    --dataset "$DATASET"
    --input_key question
    --output_key solution
    --apply_chat_template
    --max_len 4096
    --max_epochs 3
    --micro_train_batch_size 4
    --train_batch_size 16
    --save_steps -1
    --logging_steps 1
    --zero_stage 0
    --gradient_checkpointing
    --attn_implementation flash_attention_2
    --lr_warmup_ratio 0.03
)

run_exp() {
    local LR=$1
    local NAME=$2
    local OUT_DIR="/home/ubuntu/models/${NAME}"
    local TB_DIR="${TB_ROOT}/${NAME}"
    local LOG="${OUT_DIR}/training.log"

    mkdir -p "$OUT_DIR"
    mkdir -p "$TB_DIR"

    echo "========================================"
    echo "Starting experiment: ${NAME}"
    echo "LR: ${LR}"
    echo "Output: ${OUT_DIR}"
    echo "TensorBoard: ${TB_DIR}"
    echo "========================================"

    # 启动 TensorBoard（如果 6006 未占用）
    if ! ss -tlnp | grep -q ':6006 '; then
        echo "Starting TensorBoard on port 6006..."
        nohup /home/ubuntu/miniconda3/envs/llm/bin/tensorboard --logdir="$TB_ROOT" --port=6006 --host=0.0.0.0 > "${OUT_DIR}/tensorboard.log" 2>&1 &
        sleep 2
    fi

    "$DEEPSPEED_BIN" \
        --master_port 29663 \
        --module openrlhf.cli.train_sft \
        "${COMMON_ARGS[@]}" \
        --learning_rate "$LR" \
        --save_path "$OUT_DIR" \
        --use_tensorboard "$TB_DIR" \
        --wandb_run_name "$NAME" \
        2>&1 | tee "$LOG"

    echo "Experiment ${NAME} completed. Model saved to ${OUT_DIR}"
    echo ""
}

# 实验 1: lr = 5e-6
run_exp "5e-6" "Qwen3-0.6B-SFT-Math-lr5e-6"

# 实验 2: lr = 1e-6
run_exp "1e-6" "Qwen3-0.6B-SFT-Math-lr1e-6"

echo "========================================"
echo "All experiments completed."
echo "========================================"
