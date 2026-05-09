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

cleanup_non_best_weights() {
  local run_root="$1"
  local removed_any=0

  for path in \
    "${run_root}/checkpoints_sft" \
    "${run_root}/training_final_model" \
    "${run_root}/final_model"; do
    if [[ -e "${path}" ]]; then
      rm -rf "${path}"
      removed_any=1
      log "Removed intermediate weight directory: ${path}"
    fi
  done

  if [[ "${removed_any}" -eq 0 ]]; then
    log "No intermediate weight directories needed cleanup."
  fi
}

cleanup_other_results() {
  local keep_dir="$1"
  local result_root="${PROJECT_ROOT}/result"
  require_dir "${result_root}"

  while IFS= read -r candidate; do
    [[ -z "${candidate}" ]] && continue
    if [[ "${candidate}" == "${keep_dir}" ]]; then
      continue
    fi
    rm -rf "${candidate}"
    log "Removed old result directory: ${candidate}"
  done < <(find "${result_root}" -mindepth 1 -maxdepth 1 -type d | sort)
}

PYTHON_BIN="$(pick_python_bin)"
MODEL_PATH="$(infer_model_path)"
TRAIN_DATASET="${TRAIN_DATASET:-${DATA_ROOT}/deepseek-v3.2-speciale-openr1-math-3k.plus_distilled_corpus_400k_with_cot-filtered.system_prompt.sft_train.jsonl}"
VALIDATION_INPUT="${VALIDATION_INPUT:-${DATA_ROOT}/valid_1000.jsonl}"
CHAT_TEMPLATE_FILE="${CHAT_TEMPLATE_FILE:-${PIPELINE_DIR}/templates/qwen3_06b_base_eot_chat_template.jinja}"
TRAIN_GPU="${TRAIN_GPU:-0}"
EVAL_DEVICE_ID="${EVAL_DEVICE_ID:-0}"
MASTER_PORT="${MASTER_PORT:-29663}"
MAX_LEN="${MAX_LEN:-4096}"
MAX_EPOCHS="${MAX_EPOCHS:-3}"
MICRO_TRAIN_BATCH_SIZE="${MICRO_TRAIN_BATCH_SIZE:-3}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-8}"
SAVE_PER_EPOCH="${SAVE_PER_EPOCH:-1}"
TARGET_MID_CHECKPOINTS="${TARGET_MID_CHECKPOINTS:-4}"
LEARNING_RATE="${LEARNING_RATE:-3e-7}"
VALIDATION_LIMIT="${VALIDATION_LIMIT:-1000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-4096}"
EVAL_VLLM_ATTENTION_BACKEND="${EVAL_VLLM_ATTENTION_BACKEND:-auto}"
EVAL_VLLM_SCHEDULER_MODE="${EVAL_VLLM_SCHEDULER_MODE:-server_async}"
ZERO_STAGE="${ZERO_STAGE:-2}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
TB_RUN_NAME="${TB_RUN_NAME:-openrlhf_sft_async_eval_pipeline}"
EVAL_OUTPUT_DIR_NAME="${EVAL_OUTPUT_DIR_NAME:-eval_valid1000_async_b64}"
FINAL_MODEL_DIR_NAME="${FINAL_MODEL_DIR_NAME:-best_final_model}"
CLEANUP_NON_BEST_WEIGHTS="${CLEANUP_NON_BEST_WEIGHTS:-1}"
CLEANUP_OTHER_RESULTS="${CLEANUP_OTHER_RESULTS:-0}"
SKIP_TRAINING="${SKIP_TRAINING:-0}"
RUN_STAMP="$(date -u +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-openrlhf_single_gpu_sft_async_eval_e3_mb3_ga8_lr3e7_${RUN_STAMP}}"
RUN_ROOT="${RUN_ROOT:-${RESULT_ROOT}/${RUN_NAME}}"

mkdir -p "${RESULT_ROOT}"

require_file "${TRAIN_DATASET}"
require_file "${VALIDATION_INPUT}"
require_file "${CHAT_TEMPLATE_FILE}"
require_dir "${PIPELINE_DIR}"
require_dir "${DATA_ROOT}"
require_dir "${RESULT_ROOT}"
require_dir "${MODEL_PATH}"
require_file "${MODEL_PATH}/config.json"
require_file "${MODEL_PATH}/tokenizer.json"

log "Starting the one-click OpenRLHF SFT + async evaluation pipeline."
log "Pipeline directory: ${PIPELINE_DIR}"
log "Project root: ${PROJECT_ROOT}"
log "Runtime root: ${RUNTIME_ROOT}"
log "Python interpreter: ${PYTHON_BIN}"
log "Model path: ${MODEL_PATH}"
log "Training dataset: ${TRAIN_DATASET}"
log "Validation input: ${VALIDATION_INPUT}"
log "Run root: ${RUN_ROOT}"
log "Training config: max_epochs=${MAX_EPOCHS}, micro_batch=${MICRO_TRAIN_BATCH_SIZE}, gradient_accumulation=${GRADIENT_ACCUMULATION}, learning_rate=${LEARNING_RATE}, max_len=${MAX_LEN}, save_per_epoch=${SAVE_PER_EPOCH}"
log "Evaluation config: scheduler_mode=${EVAL_VLLM_SCHEDULER_MODE}, batch_size=${EVAL_BATCH_SIZE}, validation_limit=${VALIDATION_LIMIT}, requested_max_new_tokens=${EVAL_MAX_NEW_TOKENS}"
log "Final model directory name: ${FINAL_MODEL_DIR_NAME}"

mkdir -p "${RUN_ROOT}"

CMD=(
  "${PYTHON_BIN}"
  "${PIPELINE_DIR}/run_openrlhf_sft_train_eval_best.py"
  "--model-path" "${MODEL_PATH}"
  "--train-dataset" "${TRAIN_DATASET}"
  "--validation-input" "${VALIDATION_INPUT}"
  "--chat-template-file" "${CHAT_TEMPLATE_FILE}"
  "--run-root" "${RUN_ROOT}"
  "--max-len" "${MAX_LEN}"
  "--max-epochs" "${MAX_EPOCHS}"
  "--micro-train-batch-size" "${MICRO_TRAIN_BATCH_SIZE}"
  "--gradient-accumulation" "${GRADIENT_ACCUMULATION}"
  "--learning-rate" "${LEARNING_RATE}"
  "--train-gpu" "${TRAIN_GPU}"
  "--eval-device-id" "${EVAL_DEVICE_ID}"
  "--master-port" "${MASTER_PORT}"
  "--validation-limit" "${VALIDATION_LIMIT}"
  "--eval-batch-size" "${EVAL_BATCH_SIZE}"
  "--eval-max-new-tokens" "${EVAL_MAX_NEW_TOKENS}"
  "--eval-vllm-attention-backend" "${EVAL_VLLM_ATTENTION_BACKEND}"
  "--eval-vllm-scheduler-mode" "${EVAL_VLLM_SCHEDULER_MODE}"
  "--tb-run-name" "${TB_RUN_NAME}"
  "--zero-stage" "${ZERO_STAGE}"
  "--attn-implementation" "${ATTN_IMPLEMENTATION}"
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

log "Launching the Python orchestrator."
"${CMD[@]}"

if [[ "${CLEANUP_NON_BEST_WEIGHTS}" == "1" ]]; then
  log "Cleaning intermediate checkpoints and raw training weights."
  cleanup_non_best_weights "${RUN_ROOT}"
else
  log "Intermediate checkpoints are kept because CLEANUP_NON_BEST_WEIGHTS=${CLEANUP_NON_BEST_WEIGHTS}."
fi

if [[ "${CLEANUP_OTHER_RESULTS}" == "1" ]]; then
  log "Cleaning other result directories and keeping only the current SFT run."
  cleanup_other_results "${RUN_ROOT}"
else
  log "Other result directories are kept because CLEANUP_OTHER_RESULTS=${CLEANUP_OTHER_RESULTS}."
fi

log "Pipeline finished successfully."
log "Best model: ${RUN_ROOT}/${FINAL_MODEL_DIR_NAME}"
log "English report: ${RUN_ROOT}/evaluation_report.md"
log "Evaluation summary: ${RUN_ROOT}/evaluation_summary.json"
log "Best-model selection: ${RUN_ROOT}/best_model_selection.json"
log "Training loss table: ${RUN_ROOT}/training_loss_history.md"
log "Training loss plot: ${RUN_ROOT}/training_loss_curve.png"
log "Validation accuracy table: ${RUN_ROOT}/validation_accuracy_history.md"
log "Validation accuracy plot: ${RUN_ROOT}/validation_accuracy_curve.png"
