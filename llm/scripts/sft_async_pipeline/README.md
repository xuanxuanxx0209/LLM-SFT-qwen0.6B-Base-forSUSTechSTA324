# SFT Async Eval Pipeline

This folder contains the complete one-click single-GPU SFT + async validation pipeline.

## Entry Point

Run the whole workflow with:

```bash
bash /dev/shm/llm/scripts/sft_async_pipeline/run_sft_async_eval_pipeline.sh
```

If you are already inside a model directory that contains `config.json` and
`tokenizer.json`, the shell script will use the current directory as
`MODEL_PATH`. Otherwise, set `MODEL_PATH` explicitly:

```bash
MODEL_PATH=/dev/shm/llm/models/Qwen3-0.6B-Base bash /dev/shm/llm/scripts/sft_async_pipeline/run_sft_async_eval_pipeline.sh
```

## What This Pipeline Does

1. Loads the model from the current directory or `MODEL_PATH`.
2. Runs OpenRLHF SFT training.
3. Saves one HF checkpoint at the end of each epoch by default.
4. Evaluates checkpoints plus the training-final model on `valid_1000`.
5. Uses async local scheduling with `server_async`.
6. Selects the best validation-performing model.
7. Promotes that model into `best_final_model`.
8. Generates an English report.
9. Generates loss and validation-accuracy curve files.
10. Cleans non-best weights by default.

## Default Configuration

- `max_epochs=3`
- `micro_train_batch_size=3`
- `gradient_accumulation=8`
- `learning_rate=3e-7`
- `save_per_epoch=1`
- `eval_batch_size=64`
- `requested_max_new_tokens=4096`
- `eval_vllm_scheduler_mode=server_async`

## Files In This Folder

- `run_sft_async_eval_pipeline.sh`
  Main bash entry point.
- `run_openrlhf_sft_train_eval_best.py`
  Training + checkpoint evaluation + best-model promotion.
- `eval_sft_checkpoint.py`
  Evaluates a single checkpoint or final model.
- `run_vllm_valid_single_gpu.py`
  Single-GPU async or batched vLLM evaluation backend.
- `run_hf_valid_single_gpu.py`
  Transformers fallback evaluator.
- `score_valid_outputs.py`
  Accuracy scorer for generated outputs.
- `generate_sft_curves.py`
  Generates loss / accuracy history files and plots.
- `templates/qwen3_06b_base_eot_chat_template.jinja`
  Chat template used by training and evaluation.

## Important Environment Variables

- `MODEL_PATH`
- `TRAIN_DATASET`
- `VALIDATION_INPUT`
- `RUN_ROOT`
- `RUN_NAME`
- `TRAIN_GPU`
- `EVAL_DEVICE_ID`
- `MAX_EPOCHS`
- `MICRO_TRAIN_BATCH_SIZE`
- `GRADIENT_ACCUMULATION`
- `LEARNING_RATE`
- `EVAL_BATCH_SIZE`
- `EVAL_MAX_NEW_TOKENS`
- `CLEANUP_NON_BEST_WEIGHTS`
- `CLEANUP_OTHER_RESULTS`
- `SKIP_TRAINING`

## Main Output Files

Under `RUN_ROOT`, the main outputs are:

- `best_final_model/`
- `evaluation_report.md`
- `evaluation_summary.json`
- `best_model_selection.json`
- `training_loss_curve.png`
- `training_loss_curve.pdf`
- `validation_accuracy_curve.png`
- `validation_accuracy_curve.pdf`
- `training_loss_history.md`
- `validation_accuracy_history.md`

## Current Canonical Result Directory

The default runtime result root is:

`/dev/shm/llm/result/`
