#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PIPELINE_DIR="${SCRIPT_DIR}"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
CALLER_DIR="$(pwd -P)"
RUNTIME_ROOT="${LLM_RUNTIME_ROOT:-/dev/shm/llm}"
MODEL_ROOT="${LLM_MODEL_ROOT:-${RUNTIME_ROOT}/models}"
DATA_ROOT="${LLM_DATA_ROOT:-${RUNTIME_ROOT}/data}"
RESULT_ROOT="${LLM_RESULT_ROOT:-${RUNTIME_ROOT}/result}"
CACHE_ROOT="${LLM_CACHE_ROOT:-${RUNTIME_ROOT}/cache}"

timestamp() {
  date -u +"%Y-%m-%d %H:%M:%S UTC"
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    log "Required file is missing: ${path}"
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    log "Required directory is missing: ${path}"
    exit 1
  fi
}

has_model_files() {
  local path="$1"
  [[ -f "${path}/config.json" && -f "${path}/tokenizer.json" ]]
}

pick_python_bin() {
  if [[ -n "${PYTHON_BIN:-}" && -x "${PYTHON_BIN}" ]]; then
    printf '%s\n' "${PYTHON_BIN}"
    return
  fi
  if [[ -x "/home/ubuntu/miniconda3/envs/llm/bin/python" ]]; then
    printf '%s\n' "/home/ubuntu/miniconda3/envs/llm/bin/python"
    return
  fi
  if [[ -x "/root/miniconda3/envs/llm/bin/python" ]]; then
    printf '%s\n' "/root/miniconda3/envs/llm/bin/python"
    return
  fi
  command -v python3
}

find_model_candidate() {
  local candidate
  for candidate in \
    "${MODEL_ROOT}/Qwen3-0.6B-Base" \
    "${MODEL_ROOT}/Qwen3-0.6B-BASE" \
    "${MODEL_ROOT}/Qwen3-0.6B" \
    "/dev/shm/models/Qwen3-0.6B-Base" \
    "/dev/shm/models/Qwen3-0.6B-BASE" \
    "/dev/shm/models/Qwen3-0.6B" \
    "${HOME}/models/Qwen3-0.6B-Base" \
    "${HOME}/models/Qwen3-0.6B-BASE" \
    "${HOME}/models/Qwen3-0.6B" \
    "${PROJECT_ROOT}/models/Qwen3-0.6B-Base" \
    "${PROJECT_ROOT}/models/Qwen3-0.6B-BASE" \
    "${PROJECT_ROOT}/models/Qwen3-0.6B"; do
    if [[ -d "${candidate}" ]] && has_model_files "${candidate}"; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

infer_model_path() {
  if [[ -n "${MODEL_PATH:-}" ]]; then
    printf '%s\n' "${MODEL_PATH}"
    return
  fi
  if [[ -f "${CALLER_DIR}/config.json" && -f "${CALLER_DIR}/tokenizer.json" ]]; then
    printf '%s\n' "${CALLER_DIR}"
    return
  fi
  if find_model_candidate; then
    return
  fi
  printf '%s\n' "${MODEL_ROOT}/Qwen3-0.6B-Base"
}

infer_chat_template_file() {
  if [[ -n "${CHAT_TEMPLATE_FILE:-}" ]]; then
    printf '%s\n' "${CHAT_TEMPLATE_FILE}"
    return
  fi
  if [[ -f "${MODEL_PATH}/chat_template.jinja" ]]; then
    printf '%s\n' "${MODEL_PATH}/chat_template.jinja"
    return
  fi
  printf '%s\n' "${PIPELINE_DIR}/templates/qwen3_06b_base_eot_chat_template.jinja"
}

PYTHON_BIN="$(pick_python_bin)"
MODEL_PATH="$(infer_model_path)"
RAW_RLHF_DATASET="${RAW_RLHF_DATASET:-${DATA_ROOT}/deepseek-v3.2-speciale-openr1-math-3k.rlhf_train.jsonl}"
TRAIN_DATASET="${TRAIN_DATASET:-${DATA_ROOT}/deepseek-v3.2-speciale-openr1-math-3k.grpo_prompt_train.jsonl}"
PREP_METADATA_OUTPUT="${PREP_METADATA_OUTPUT:-${DATA_ROOT}/deepseek-v3.2-speciale-openr1-math-3k.grpo_prompt_train.metadata.json}"
VALIDATION_INPUT="${VALIDATION_INPUT:-${DATA_ROOT}/valid_1000.jsonl}"
CHAT_TEMPLATE_FILE="$(infer_chat_template_file)"
REWARD_SCRIPT="${REWARD_SCRIPT:-${PIPELINE_DIR}/math_exact_match_reward.py}"
TRAIN_GPU="${TRAIN_GPU:-0}"
EVAL_DEVICE_ID="${EVAL_DEVICE_ID:-0}"
MAX_LABEL_CHARS="${MAX_LABEL_CHARS:-120}"
MAX_LEN="${MAX_LEN:-4096}"
PROMPT_MAX_LEN="${PROMPT_MAX_LEN:-1024}"
GENERATE_MAX_LEN="${GENERATE_MAX_LEN:-3072}"
MAX_EPOCHS="${MAX_EPOCHS:-4}"
MICRO_TRAIN_BATCH_SIZE="${MICRO_TRAIN_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-15}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-6}"
MICRO_ROLLOUT_BATCH_SIZE="${MICRO_ROLLOUT_BATCH_SIZE:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
ACTOR_LEARNING_RATE="${ACTOR_LEARNING_RATE:-5e-7}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.6}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
SAVE_PER_EPOCH="${SAVE_PER_EPOCH:-1}"
TARGET_MID_CHECKPOINTS="${TARGET_MID_CHECKPOINTS:-4}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.15}"
VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}"
VALIDATION_LIMIT="${VALIDATION_LIMIT:-1000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-4096}"
EVAL_VLLM_ATTENTION_BACKEND="${EVAL_VLLM_ATTENTION_BACKEND:-auto}"
EVAL_VLLM_SCHEDULER_MODE="${EVAL_VLLM_SCHEDULER_MODE:-server_async}"
ZERO_STAGE="${ZERO_STAGE:-2}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
TB_RUN_NAME="${TB_RUN_NAME:-openrlhf_grpo_async_eval_pipeline}"
EVAL_OUTPUT_DIR_NAME="${EVAL_OUTPUT_DIR_NAME:-eval_valid1000_async_b64}"
FINAL_MODEL_DIR_NAME="${FINAL_MODEL_DIR_NAME:-best_final_model}"
SKIP_TRAINING="${SKIP_TRAINING:-0}"
SKIP_CURVE_GENERATION="${SKIP_CURVE_GENERATION:-0}"
RUN_STAMP="$(date -u +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-openrlhf_single_gpu_grpo_async_eval_e4_mb2_ga15_ns6_${RUN_STAMP}}"
RUN_ROOT="${RUN_ROOT:-${RESULT_ROOT}/${RUN_NAME}}"

mkdir -p "${RESULT_ROOT}"

require_file "${RAW_RLHF_DATASET}"
require_file "${VALIDATION_INPUT}"
require_file "${CHAT_TEMPLATE_FILE}"
require_file "${REWARD_SCRIPT}"
require_dir "${PIPELINE_DIR}"
require_dir "${DATA_ROOT}"
require_dir "${RESULT_ROOT}"
require_dir "${MODEL_PATH}"
require_file "${MODEL_PATH}/config.json"
require_file "${MODEL_PATH}/tokenizer.json"

log "Starting the one-click OpenRLHF GRPO + async evaluation pipeline."
log "Pipeline directory: ${PIPELINE_DIR}"
log "Project root: ${PROJECT_ROOT}"
log "Runtime root: ${RUNTIME_ROOT}"
log "Python interpreter: ${PYTHON_BIN}"
log "Model path: ${MODEL_PATH}"
log "Raw RLHF dataset: ${RAW_RLHF_DATASET}"
log "Prepared GRPO dataset: ${TRAIN_DATASET}"
log "Validation input: ${VALIDATION_INPUT}"
log "Run root: ${RUN_ROOT}"
log "Training config: dataset_epochs=${MAX_EPOCHS}, ppo_max_epochs_per_rollout=1, micro_batch=${MICRO_TRAIN_BATCH_SIZE}, gradient_accumulation=${GRADIENT_ACCUMULATION}, n_samples_per_prompt=${N_SAMPLES_PER_PROMPT}, actor_learning_rate=${ACTOR_LEARNING_RATE}, rollout_temperature=${ROLLOUT_TEMPERATURE}, max_len=${MAX_LEN}"
log "Sample budget: max_samples=${MAX_SAMPLES} (0 means full prepared dataset)"
log "Evaluation config: scheduler_mode=${EVAL_VLLM_SCHEDULER_MODE}, batch_size=${EVAL_BATCH_SIZE}, validation_limit=${VALIDATION_LIMIT}, requested_max_new_tokens=${EVAL_MAX_NEW_TOKENS}"
log "vLLM backend override inside OpenRLHF: ${VLLM_ATTENTION_BACKEND}"

mkdir -p "${RUN_ROOT}"
export TMPDIR="${TMPDIR:-/dev/shm}"
export TMPDIR="${TMPDIR:-${CACHE_ROOT}/tmp}"
export HOME="${HOME:-${CACHE_ROOT}/home}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${CACHE_ROOT}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${CACHE_ROOT}/transformers}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-${CACHE_ROOT}/xdg_config}"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
mkdir -p \
  "${TMPDIR}" \
  "${HOME}" \
  "${HF_HOME}" \
  "${HF_DATASETS_CACHE}" \
  "${TRANSFORMERS_CACHE}" \
  "${TORCH_HOME}" \
  "${TRITON_CACHE_DIR}" \
  "${XDG_CACHE_HOME}" \
  "${XDG_CONFIG_HOME}"

log "Preparing the GRPO prompt-only dataset with concise exact-match targets."
"${PYTHON_BIN}" "${PIPELINE_DIR}/prepare_grpo_prompt_dataset.py" \
  --input "${RAW_RLHF_DATASET}" \
  --output "${TRAIN_DATASET}" \
  --metadata-output "${PREP_METADATA_OUTPUT}" \
  --model-path "${MODEL_PATH}" \
  --chat-template-file "${CHAT_TEMPLATE_FILE}" \
  --max-label-chars "${MAX_LABEL_CHARS}"

CMD=(
  "${PYTHON_BIN}"
  "${PIPELINE_DIR}/run_openrlhf_grpo_train_eval_best.py"
  "--model-path" "${MODEL_PATH}"
  "--train-dataset" "${TRAIN_DATASET}"
  "--reward-script" "${REWARD_SCRIPT}"
  "--validation-input" "${VALIDATION_INPUT}"
  "--chat-template-file" "${CHAT_TEMPLATE_FILE}"
  "--run-root" "${RUN_ROOT}"
  "--max-len" "${MAX_LEN}"
  "--prompt-max-len" "${PROMPT_MAX_LEN}"
  "--generate-max-len" "${GENERATE_MAX_LEN}"
  "--max-epochs" "${MAX_EPOCHS}"
  "--micro-train-batch-size" "${MICRO_TRAIN_BATCH_SIZE}"
  "--gradient-accumulation" "${GRADIENT_ACCUMULATION}"
  "--n-samples-per-prompt" "${N_SAMPLES_PER_PROMPT}"
  "--micro-rollout-batch-size" "${MICRO_ROLLOUT_BATCH_SIZE}"
  "--actor-learning-rate" "${ACTOR_LEARNING_RATE}"
  "--temperature" "${ROLLOUT_TEMPERATURE}"
  "--top-p" "${ROLLOUT_TOP_P}"
  "--max-samples" "${MAX_SAMPLES}"
  "--train-gpu" "${TRAIN_GPU}"
  "--eval-device-id" "${EVAL_DEVICE_ID}"
  "--validation-limit" "${VALIDATION_LIMIT}"
  "--eval-batch-size" "${EVAL_BATCH_SIZE}"
  "--eval-max-new-tokens" "${EVAL_MAX_NEW_TOKENS}"
  "--eval-vllm-attention-backend" "${EVAL_VLLM_ATTENTION_BACKEND}"
  "--eval-vllm-scheduler-mode" "${EVAL_VLLM_SCHEDULER_MODE}"
  "--tb-run-name" "${TB_RUN_NAME}"
  "--zero-stage" "${ZERO_STAGE}"
  "--attn-implementation" "${ATTN_IMPLEMENTATION}"
  "--vllm-gpu-memory-utilization" "${VLLM_GPU_MEMORY_UTILIZATION}"
  "--vllm-attention-backend" "${VLLM_ATTENTION_BACKEND}"
  "--eval-output-dir-name" "${EVAL_OUTPUT_DIR_NAME}"
  "--final-model-dir-name" "${FINAL_MODEL_DIR_NAME}"
)

if [[ "${SAVE_PER_EPOCH}" == "1" ]]; then
  CMD+=("--save-per-epoch")
else
  CMD+=("--no-save-per-epoch" "--target-mid-checkpoints" "${TARGET_MID_CHECKPOINTS}")
fi

if [[ "${SKIP_TRAINING}" == "1" ]]; then
  CMD+=("--skip-training")
  log "Skip-training mode is enabled. The script will only evaluate existing checkpoints/models."
fi

if [[ "${SKIP_CURVE_GENERATION}" == "1" ]]; then
  CMD+=("--skip-curve-generation")
  log "Curve generation is disabled for this run."
fi

log "Launching the Python orchestrator."
"${CMD[@]}"

log "Pipeline finished successfully."
log "Best model: ${RUN_ROOT}/${FINAL_MODEL_DIR_NAME}"
log "English report: ${RUN_ROOT}/evaluation_report.md"
log "Evaluation summary: ${RUN_ROOT}/evaluation_summary.json"
log "Best-model selection: ${RUN_ROOT}/best_model_selection.json"
log "Training loss table: ${RUN_ROOT}/training_loss_history.md"
log "Training loss plot: ${RUN_ROOT}/training_loss_curve.png"
log "Training reward table: ${RUN_ROOT}/training_reward_history.md"
log "Training reward plot: ${RUN_ROOT}/training_reward_curve.png"
log "Validation accuracy table: ${RUN_ROOT}/validation_accuracy_history.md"
log "Validation accuracy plot: ${RUN_ROOT}/validation_accuracy_curve.png"
