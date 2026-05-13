# Qwen3-0.6B-Base 数学能力 SFT / LoRA 微调

> 期中项目：基于 Nemotron-CC-Math-v1 清洗数据，对 Qwen3-0.6B-Base 进行全量 SFT 与 LoRA 弱项强化，提升数学推理能力。

---

## 项目概览

| 项目 | 内容 |
|------|------|
| **基模型** | [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) |
| **训练框架** | OpenRLHF + DeepSpeed ZeRO-2 |
| **数据来源** | Nemotron-CC-Math-v1 清洗结果 |
| **验证集** | 1000 道数学推理题（含 perturbation_type 分桶标签） |
| **最佳模型** | `qwen3-0.6B-mathsft-V3-lora-merged-v5` —— 验证集准确率 **46.80%** |
| **Hugging Face** | [Unefois/qwen3-0.6B-SFTforMATH-SUSTechSTA324project](https://huggingface.co/Unefois/qwen3-0.6B-SFTforMATH-SUSTechSTA324project) |
---

## 目录结构

```
.
├── run_math_sft.py              # 全量 SFT 训练脚本（第一阶段）
├── run_lora_math_sft.py         # LoRA 微调脚本（第三阶段）
├── build_lora_dataset.py        # 构建 LoRA 弱项训练集
├── merge_lora.py                # 将 LoRA adapter 合并为独立模型
├── run_eval.py                  # 基础评测脚本
├── run_eval_with_dump.py        # 逐题 dump 评测（含分题型统计）
├── run_eval_lora_merged.py      # 评测合并后的 LoRA 模型
├── jgy-dataset/                 # 训练数据集
│   ├── 7460.jsonl               # 原始清洗数据（7456 条）
│   ├── 7460_cleaned.jsonl       # 二次清洗数据
│   ├── final_en_sft_scoring_aligned.jsonl   # 第一阶段全量 SFT 训练数据（952 条）
│   ├── merged_dataset_distilled.jsonl       # V3 基模训练数据
│   ├── lora_weakness_train_v2.jsonl   # LoRA v2/v5 主力训练集（100 CT + 700 replay）
│   └── ...
├── validation-set/              # 验证集与评分脚本
│   ├── valid_1000.jsonl         # 1000 题验证集
│   └── score_valid_outputs.py   # 答案提取与匹配评分逻辑
├── ClassResource/               # 课程参考资源
│   ├── Chapter7_Data.ipynb      # 数据清洗教程（课程提供）
│   └── Chapter_4_vllm_lab.md    # vLLM 启动与评测教程（课程提供）
└── eval_dumps/                  # 各版本逐题评测结果（生成）
```

> **注意**：模型权重文件（`qwen3-0.6B*` 等）体积超过 10GB，未包含在本仓库中。如需获取最终模型权重，见下方 [模型权重获取](#模型权重获取) 部分。

---

## 从基模型到最终模型的完整过程

本项目经历了 **三个阶段**：全量 SFT 基线 → 消融探索 → LoRA 弱项强化。

### 第一阶段：全量 SFT 基线（j1 / model-j）

**目标**：建立第一条可运行的训练流水线，产出超越 baseline 的初始模型。

- **基模型**：`Qwen3-0.6B-Base`
- **训练数据**：`jgy-dataset/final_en_sft_scoring_aligned.jsonl`（952 条数学样本）
- **关键超参**：
  - 学习率 `5e-6`
  - `train_batch_size=16`, `micro_batch_size=4`, `gradient_accumulation=4`
  - `max_epochs=3`, `max_len=4096`
- **chat template**：自定义 template，显式保留 `<think>` 思考标签，确保训练与推理两侧形态一致
- **产出**：`Qwen3-0.6B-Base-Math-SFT-v2`（小组第一版超 baseline 模型）

**脚本**：[`run_math_sft.py`](run_math_sft.py)

---

### 第二阶段：训练策略探索与消融

**目标**：在数据组成与训练超参两个维度上系统探索，寻找突破瓶颈的方向。

| 探索方向 | 具体操作 | 结论 |
|----------|----------|------|
| 加入通用领域数据 | 混入编程/生产力/生活/社会/文本领域数据 | 验证集无明显提升，数学样本被稀释 |
| 加入 `\boxed{none}` 拒答样本 | 引入无解题样本，教模型"无法回答时拒答" | 对无解题判别略有响应，但整体准确率下降 |
| 加入批判性思考样本 | 在数据中加入 critical thinking 类型题 | 无稳定增益 |
| 学习率消融 | `5e-6` / `2e-6` / `1e-6` 三档对比 | `5e-6` 收敛最快但易过拟合；`1e-6` 收敛不足 |
| 梯度累积 / 优化器系数 / 正则项 | 多组对照实验 | 差异不显著 |

**关键结论**：在仅依赖单一全量 SFT 的范式下，模型性能已逼近收益上限。必须切换到**更精细的训练策略**（分阶段训练 + 局部精修）才能继续提升。

---

### 第三阶段：基于 V3 的 LoRA 弱项强化（主要产出）

 teammate 缪梓航同学采用"高比例基础题 + 低比例进阶题"的数据组织思路训练出 `qwen3-0.6B-mathsft-V3`（验证集准确率 **44.30%**）。

- **训练数据**：`jgy-dataset/merged_dataset_distilled.jsonl`
- **关键超参**：
  - 学习率 `1.5e-6`
  - `global_batch_size = 16`
  - `epoch = 2`

本阶段在此基础上进一步提升。

#### Step 1 —— V3 分题型能力分析

使用 [`run_eval_with_dump.py`](run_eval_with_dump.py) 对 V3 进行批量推理，逐题保存结果，按 `perturbation_type` 分桶统计。

**关键发现**：V3 在 **critical thinking** 题型上完全失败（`0/130`，占整体 13 个百分点）。这类题是把 GSM8K 题目中的关键数字删掉变成"无解题"，期望答案为 `None`。V3 因从未见过"应当拒答"的样本，会强行编出一个数字。

#### Step 2 —— 构建 LoRA 弱项训练集

编写 [`build_lora_dataset.py`](build_lora_dataset.py)，采用 **"少量 CT 合成题 + 大量 replay 基础题"** 的混合配方：

- **CT 合成题**：从 `7460_cleaned.jsonl` 中选取 GSM8K 风格题目，用正则表达式移除关键数字（价格、成本、速率等），生成"信息缺失"变体，答案模板引导至 `\boxed{None}`。
- **replay 基础题**：从同一份数据池中抽取未修改的原始样本，作为锚点防止灾难性遗忘。
- **防泄漏校验**：与 `validation-set/valid_1000.jsonl` 进行问题文本交叉比对，确保 0 泄漏。

**两轮迭代**：
- `lora_weakness_train.jsonl`：**150 CT + 400 replay = 550 条**（CT 占比 27%，后被验证为过高）
- `lora_weakness_train_v2.jsonl`：**100 CT + 700 replay = 800 条**（CT 占比 12.5%，为后续主力训练集）

#### Step 3 —— LoRA 微调：6 个版本迭代

| 版本 | 关键变更 | 验证集准确率 | 备注 |
|------|----------|-------------|------|
| V3 baseline | — | **44.30%** |  teammate 训练的基模 |
| v1 | rank=8, lr=1e-4, CT 27% | ~40% | 灾难性遗忘，整体下降 3.6% |
| **v2** | rank=8, lr=5e-5, CT 12.5% | ~44.5% | 首次实现稳定正向收益 |
| v3 | rank=8, epoch/CT 微调 | ~44.7% | +0.2% 波动 |
| v4 | rank=8, 继续微调 | ~44.9% | +0.4% 波动，rank=8 已见瓶颈 |
| **v5** | **rank=4, alpha=8, lr=5e-5** | **46.80%** | **全题型同时上升，最佳模型** |
| v6 | rank=8, lr=2.5e-5 | ~43.9% | 反向确认 v5 超参为局部最优 |

**v5 的核心突破**：
- 将 rank 从 8 降到 4（alpha 同比减半到 8），用更小的 LoRA 容量起到更强的正则化作用
- `target_modules` 显式**排除 `lm_head` 与 `embed_tokens`**，降低对输出层先验的扰动
- 结果：**CT +11.54%、digit expansion +5.08%、numerical substitution +4.09%**，且其他题型未回退

**脚本**：[`run_lora_math_sft.py`](run_lora_math_sft.py)

#### Step 4 —— 合并 LoRA 并产出最终模型

使用 [`merge_lora.py`](merge_lora.py) 将 LoRA adapter 合并回 V3 基模，生成独立的 `model.safetensors`：

```python
# 加载基模 + adapter → merge_and_unload → 保存
base = AutoModelForCausalLM.from_pretrained(BASE_PATH, ...)
model = PeftModel.from_pretrained(base, ADAPTER_PATH)
merged = model.merge_and_unload()
merged.save_pretrained(MERGED_PATH, safe_serialization=True)
```

**最终产出**：`qwen3-0.6B-mathsft-V3-lora-merged-v5`
- 相对 V3 baseline 绝对提升：**+2.50%**
- 小组当前综合表现最好的模型

---

## 模型权重获取

最终最佳模型已上传至 Hugging Face Hub：

**[Unefois/qwen3-0.6B-SFTforMATH-SUSTechSTA324project](https://huggingface.co/Unefois/qwen3-0.6B-SFTforMATH-SUSTechSTA324project)**

---

## 快速开始

### 环境要求

- Python 3.10+
- CUDA 12.x
- `llm` conda 虚拟环境（已预装 PyTorch、DeepSpeed、OpenRLHF、vLLM、transformers、peft 等）

```bash
conda activate llm
```

### 1. 全量 SFT（第一阶段）

```bash
python run_math_sft.py
```

产出目录：`Qwen3-0.6B-Math-SFT/`

### 2. 评测基线模型

```bash
python run_eval.py
```

### 3. 构建 LoRA 弱项训练集

```bash
python build_lora_dataset.py
```

产出：`jgy-dataset/lora_weakness_train_v2.jsonl`

### 4. LoRA 微调（第三阶段）

编辑 [`run_lora_math_sft.py`](run_lora_math_sft.py) 中的超参（`LORA_RANK`、`LORA_ALPHA`、`LR` 等），然后：

```bash
python run_lora_math_sft.py
```

产出目录：`qwen3-0.6B-mathsft-V3-lora-r8/`（LoRA adapter）

### 5. 合并 LoRA 产出最终模型

```bash
python merge_lora.py
```

产出目录：`qwen3-0.6B-mathsft-V3-lora-merged-v5/`

### 6. 评测最终模型

```bash
python run_eval_lora_merged.py
```

---

## 关键脚本说明

| 脚本 | 作用 |
|------|------|
| `run_math_sft.py` | 全量 SFT 训练：DeepSpeed + OpenRLHF，自定义 chat template，TensorBoard 监控 |
| `run_lora_math_sft.py` | LoRA 微调：在 V3 基模上加载 LoRA adapter，可配置 rank/alpha/target_modules |
| `build_lora_dataset.py` | 弱项数据集构建：正则合成 CT 题 + replay 缓冲池 + 零泄漏校验 |
| `merge_lora.py` | 合并 LoRA：使用 PEFT 的 `merge_and_unload()` 生成独立 safetensors 模型 |
| `run_eval.py` | 基础评测：vLLM 批量推理，输出整体准确率 |
| `run_eval_with_dump.py` | 增强评测：逐题 dump 到 `eval_dumps/`，并打印分题型准确率 |
| `run_eval_lora_merged.py` | 对合并后的 LoRA 模型进行评测 |
| `validation-set/score_valid_outputs.py` | 答案提取与匹配：支持 `\boxed{}`、`#### `、数字、分数、百分数等多种格式 |

---

## 数据集说明

### 训练数据（`jgy-dataset/`）

- **原始来源**：Nemotron-CC-Math-v1 经过数据组清洗后的结果
- **关键文件**：
  - `7460.jsonl` / `7460_cleaned.jsonl`：原始清洗池（~7456 条）
  - `final_en_sft_scoring_aligned.jsonl`：第一阶段全量 SFT 数据（952 条数学样本）
  - `merged_dataset_distilled.jsonl`：V3 基模训练数据
  - `lora_weakness_train_v2.jsonl`：LoRA 主力训练集（800 条 = 100 CT 合成题 + 700 replay 基础题）

### 验证数据（`validation-set/`）

- `valid_1000.jsonl`：1000 道验证题，含 `perturbation_type` 字段（如 `critical thinking`、`digit expansion`、`numerical substitution` 等）
- `score_valid_outputs.py`：官方评分逻辑，支持多格式答案提取与数值容错匹配

---

## 项目反思与关键经验

1. **全量 SFT 奠定基础，LoRA 做精修**：在已有较强基模时，LoRA 是更可控、更不易破坏既有能力的精修工具。
2. **数据的"针对性"远比"数量"更重要**：模型最大的提升均来源于较小但高度针对性的数据集（800 条 LoRA 数据 > 数千条盲目扩充）。
3. **逐题可追溯的评测是必须的**：没有 `V3_per_item.jsonl` 这样的逐题 dump，就无法发现 CT 0/130 的核心短板。
4. **更小的 LoRA rank 不一定更弱**：在已有强基模时，过大的 rank 反而破坏已有能力。v5（rank=4）优于 v2/v3/v4（rank=8）正是这一直觉的反例验证。

---

## 作者

- **冀国源** —— 模型微调负责人（全量 SFT、LoRA 弱项强化、训练策略探索）
- **缪梓航** —— V3 基模训练、数据清洗方案设计
- **杨皓博** —— 数据清洗与质量把控

---

## License

本项目为课程期中项目，代码与数据集仅供学习交流使用。基模型 `Qwen3-0.6B-Base` 遵循其官方 License。
