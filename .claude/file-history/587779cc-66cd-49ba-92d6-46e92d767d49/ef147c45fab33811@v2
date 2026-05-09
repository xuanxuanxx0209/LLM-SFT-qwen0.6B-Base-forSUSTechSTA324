#!/bin/bash
set -euo pipefail

OUTPUT_PATH="/home/ubuntu/Qwen3-0.6B-Base-Math-SFT"
TB_LOG_DIR="${OUTPUT_PATH}/tensorboard"

mkdir -p "${OUTPUT_PATH}"
mkdir -p "${TB_LOG_DIR}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm

echo "========================================"
echo "Qwen3-0.6B-Base Math SFT Training"
echo "========================================"
echo "Dataset:   /home/ubuntu/jgy-dataset/7460.jsonl"
echo "Output:    ${OUTPUT_PATH}"
echo "LR:        1e-6"
echo "Micro BS:  4"
echo "Grad Acc:  4"
echo "Epochs:    3"
echo "Max Len:   4096"
echo "TensorBoard: http://localhost:6006"
echo "========================================"

# Start TensorBoard
echo "Starting TensorBoard on port 6006..."
nohup tensorboard --logdir "${TB_LOG_DIR}" --port 6006 --bind_all > "${OUTPUT_PATH}/tensorboard.log" 2>&1 &
TB_PID=$!
echo $TB_PID > "${OUTPUT_PATH}/tensorboard.pid"
sleep 2

# Verify TensorBoard started
if ! kill -0 ${TB_PID} 2>/dev/null; then
    echo "WARNING: TensorBoard failed to start. Check ${OUTPUT_PATH}/tensorboard.log"
else
    echo "TensorBoard running at PID ${TB_PID}"
fi

# Run training
echo "Starting training..."
python /home/ubuntu/sft_7460_launcher.py

# Cleanup TensorBoard after training
kill ${TB_PID} 2>/dev/null || true
wait ${TB_PID} 2>/dev/null || true

echo "Done."
