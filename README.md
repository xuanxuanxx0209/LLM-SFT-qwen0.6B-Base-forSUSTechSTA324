# Qwen3-0.6B-Base 数学能力 SFT / LoRA 微调

南方科技大学 STA324 期中项目。我们对 [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) 做全量监督微调（SFT），再以数学 SFT 模型 **V3（step 477）** 为起点，用 LoRA 强化条件不足的应用题。训练使用 OpenRLHF 和 DeepSpeed ZeRO-2。

| 评测记录 | 准确率 |
| --- | ---: |
| 原始 V3，自行评测 | 44.30% |
| **LoRA v5，自行评测** | **46.80%** |
| 课程平台榜单，组内最高记录 | 46.40% |

自行评测使用 `valid_1000.jsonl`，v5 比原始 V3 多答对 **25 道**，提升 **2.50 个百分点**。课程平台分数以榜单记录为准。发布权重见 [模型权重获取](#模型权重获取)。

## 训练流程和数据

### 全量 SFT

训练数据包括从 Nemotron-CC-Math-v1 清洗的数学题和后续整理的蒸馏样本。下表列出主要训练产物及其数据。

| 模型或实验 | 训练数据 | 样本数 |
| --- | --- | ---: |
| 初版 SFT `j1/model-j.safetensors` | [`final_en_sft_scoring_aligned.jsonl`](jgy-dataset/final_en_sft_scoring_aligned.jsonl)，答案格式对齐的数学题 | 952 |
| **原始 V3 `global_step477_hf`** | [`merged_dataset_distilled.jsonl`](jgy-dataset/merged_dataset_distilled.jsonl)，以基础题为主的蒸馏数据 | **2,548** |
| 早期 SFT 实验 `2-5` | [`sftdata_plus_general100_plus_critical_thinking100.jsonl`](jgy-dataset/sftdata_plus_general100_plus_critical_thinking100.jsonl)，952 条数学题、100 条通用题和 130 条 CT 题 | 1,182 |
| 扩大数据的 SFT 实验 `VVV2/5-1` | [`combined_7460_merged.jsonl`](jgy-dataset/combined_7460_merged.jsonl)，7,456 条清洗数据加上 V3 的 2,548 条蒸馏数据 | 10,004 |
| 另一组 SFT 实验 `Qwen3-0.6B-Math-SFT` | [`math_sft_think_boxed.jsonl`](jgy-dataset/math_sft_think_boxed.jsonl)，由 `final_cleaned_high_quality.jsonl` 转换而来 | 1,850 |

`model-j` 用于建立初版 SFT 和课程评测流程。后续实验尝试了混入通用数据与拒答样本，以及调整学习率和梯度累积。LoRA 实验采用 **原始 V3 的 step 477 权重**。V3 的完整训练启动参数缺少可核对的日志。

`2-5` 的 130 条 CT 样本与验证集中的 CT 题重合。解读这一轮的验证结果时需考虑这部分重叠。

### V3 诊断和 LoRA 数据构建

用 [`run_eval_with_dump.py`](run_eval_with_dump.py) 对原始 V3 运行逐题评测，并按 `perturbation_type` 汇总。原始 V3 在 **critical thinking（CT）** 上答对 **0/130**。这类题缺少求解所需条件，预期答案为 `None`。

[`build_lora_dataset.py`](build_lora_dataset.py) 以 [`7460_cleaned.jsonl`](jgy-dataset/7460_cleaned.jsonl) 为数据池，生成删除关键信息的 CT 样本，目标答案为 `\boxed{None}`。同时抽取原始数学题作为 **replay 样本**，用于保留已有解题能力。以下三份 LoRA 数据与 `valid_1000.jsonl` 进行了题面文本比对，未发现同题重叠。

| 训练数据 | CT | Replay | 合计 |
| --- | ---: | ---: | ---: |
| [`lora_weakness_train.jsonl`](jgy-dataset/lora_weakness_train.jsonl) | 150 | 400 | 550 |
| [`lora_weakness_train_v2.jsonl`](jgy-dataset/lora_weakness_train_v2.jsonl) | 100 | 700 | 800 |
| [`lora_weakness_train_v4.jsonl`](jgy-dataset/lora_weakness_train_v4.jsonl) | 150 | 700 | 850 |

### 六组 LoRA 实验

每组独立从同一份 V3 权重开始训练。表中的准确率来自自行评测。

| LoRA 版本 | 数据文件 | Rank / alpha | 学习率 | Epoch | 准确率 |
| --- | --- | ---: | ---: | ---: | ---: |
| v1 | `lora_weakness_train.jsonl` | 16 / 32 | `1e-4` | 2 | 40.70% |
| v2 | `lora_weakness_train_v2.jsonl` | 8 / 16 | `5e-5` | 1 | 44.50% |
| v3 | `lora_weakness_train_v2.jsonl` | 8 / 16 | `5e-5` | 2 | 44.90% |
| v4 | `lora_weakness_train_v4.jsonl` | 8 / 16 | `5e-5` | 1 | 44.50% |
| **v5** | **`lora_weakness_train_v2.jsonl`** | **4 / 8** | **`5e-5`** | **2** | **46.80%** |
| v6 | `lora_weakness_train_v2.jsonl` | 8 / 16 | `2.5e-5` | 2 | 43.90% |

- **v3 与 v5** 使用相同数据、学习率和训练轮数。在 `alpha / rank = 2` 下，rank 4 的这一组取得了最高准确率。
- **v2 与 v4** 同时改变了 CT 和 replay 的样本构成。两份数据各有 700 条 replay，其中 449 条的题目和回答完全一致，这组对比无法单独衡量增加 CT 数量的效果。
- **v5 相对原始 V3**，CT 从 **0/130** 提升到 **15/130**。八类题中四类上升、两类持平、两类下降。

## 模型权重获取

- **原始 V3** 的训练记录使用 `qwen3-0.6B-mathsft-V3`，对应 step 477 的 Hugging Face 格式 checkpoint `global_step477_hf`。
- **最终 LoRA v5**（`qwen3-0.6B-mathsft-V3-lora-merged-v5`）由 [`merge_lora.py`](merge_lora.py) 将 v5 adapter 合并进原始 V3 权重后导出。发布地址为 [Unefois/qwen3-0.6B-SFTforMATH-SUSTechSTA324project](https://huggingface.co/Unefois/qwen3-0.6B-SFTforMATH-SUSTechSTA324project)。

评测时，模型路径应指向包含权重、配置和 tokenizer 的完整模型目录。重新训练 LoRA 需要准备原始 V3 的完整 checkpoint。

## 运行评测和训练

脚本使用 Linux、CUDA 和包含 PyTorch、OpenRLHF、DeepSpeed、Transformers、PEFT、vLLM 的 Python 环境。历史配置中有 `/home/ubuntu` 下的绝对路径，运行前把它们替换为当前环境的路径。

### 评测已有权重

1. 原始 V3 使用 [`run_eval_with_dump.py`](run_eval_with_dump.py)，已合并的 LoRA v5 使用 [`run_eval_lora_merged.py`](run_eval_lora_merged.py)。
2. 将对应脚本的 `MODEL_PATH` 指向目标模型目录，设置 `DATA_PATH` 为 [`validation-set/valid_1000.jsonl`](validation-set/valid_1000.jsonl)，并更新评分模块导入目录、`DUMP_PATH` 和 `TAG`。合并版评测脚本的默认模型与输出文件名对应 v6。
3. 运行相应脚本。脚本用 [`score_valid_outputs.py`](validation-set/score_valid_outputs.py) 提取最终答案、判分，并输出总体和分题型准确率及逐题记录。

### 按 v5 配置训练和合并

1. 在 [`run_lora_math_sft.py`](run_lora_math_sft.py) 中，设置 `MODEL_PATH` 为原始 V3，`DATASET_PATH` 为仓库中已有的 `lora_weakness_train_v2.jsonl`，并为 adapter 指定 `SAVE_PATH`。若需要使用数据构建脚本，当前默认输出的是 **v4 的 850 条数据**。
2. 训练脚本当前默认配置对应 v6。改为 v5 时，设置 `LORA_RANK=4`、`LORA_ALPHA=8`、`LR=5e-5`、`MAX_EPOCHS=2`。全局 batch size 为 16，micro batch size 为 4，LoRA dropout 为 0.05。沿用脚本中的注意力和 MLP 投影层作为 LoRA 目标。
3. 在 [`merge_lora.py`](merge_lora.py) 中设置 `BASE_PATH` 为同一份原始 V3，`ADAPTER_PATH` 为刚训练出的 adapter 目录，`MERGED_PATH` 为 v5 完整模型的输出目录。合并后按上一节评测该输出目录。

```bash
python run_lora_math_sft.py
python merge_lora.py
python run_eval_lora_merged.py
```

## 仓库索引

- [`jgy-dataset/`](jgy-dataset/) 收录 SFT 和 LoRA 数据。
- [`validation-set/`](validation-set/) 收录 1,000 题验证集与评分代码，题型由 `perturbation_type` 标注。
- [`run_math_sft.py`](run_math_sft.py) 是全量 SFT 训练入口，当前默认运行表中的 1,850 条数据实验。
- [`ClassResource/`](ClassResource/) 收录课程数据清洗和 vLLM 推理参考资料。

---

## 作者

- **冀国源** —— 模型微调负责人（全量 SFT、LoRA 弱项强化、训练策略探索）
- **缪梓航** —— V3 基模训练、数据清洗方案设计
- **杨皓博** —— 数据清洗与质量把控

---

## License

本项目为课程期中项目，代码与数据集仅供学习交流使用。基模型 `Qwen3-0.6B-Base` 遵循其官方 License。
