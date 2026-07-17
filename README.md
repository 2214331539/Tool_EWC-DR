# Tool EWC-DR: EWC-DR for ToolHCL Continual Tool Retrieval

本仓库包含两部分：

1. EWC-DR 论文配套的原始图像分类代码，保留在 `models/`、`convs/`、`utils/`、`exps/`。
2. 面向 ToolHCL 工具检索持续学习协议的独立迁移实现，位于 `toolhcl_ewcdr_pure/`。

推荐入口是 `toolhcl_ewcdr_pure`。它只迁移 EWC-DR 的核心机制，不引入 ToolHCL 的层级 router、box、soft prompt、回放或其他抗遗忘组件，也不会修改 ToolHCL 源项目。

## 1. 来源与外部资产

### 方法与代码

- EWC-DR 论文：[Elastic Weight Consolidation Done Right for Continual Learning](https://arxiv.org/abs/2603.18596)
- EWC-DR 原始代码：[scarlet0703/EWC-DR](https://github.com/scarlet0703/EWC-DR)
- ToolHCL 项目：[2214331539/ToolHCHL](https://github.com/2214331539/ToolHCHL)

### 数据与模型

- ToolHCL 的 base/task1/task2/task3 派生检索数据由 ToolHCL 项目准备，不在本仓库重复分发。
- ToolHCL base 数据来源及处理背景参考 [OpenBMB/ToolBench](https://github.com/OpenBMB/ToolBench)。
- 查询编码器使用 gated 模型 [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)。使用者需要自行申请访问、接受 Meta Llama 3 许可并下载权重。
- 本仓库不包含 LLaMA 权重、ToolHCL 原始数据、特征缓存、checkpoint 或 importance 文件。

本实现使用完整的 32 层 Meta-Llama-3-8B，而不是 hashing encoder、BERT 或 token-embedding 近似。模型权重只从本地目录加载，代码设置了 `local_files_only=True`。

## 2. 迁移机制

### 2.1 数据流

```text
query text
  -> LLaMA tokenizer，右侧 padding，最大长度 512
  -> 完整冻结的 LLaMA-3-8B
  -> 最后一个有效 token 的 4096 维 hidden state
  -> trainable query projection: 4096 -> 1024 -> 384
  -> 累积扩展的 global linear classifier
  -> visible tool logits
  -> 按 logits 降序映射回全局 tool ID
```

工具 ID 直接读取 ToolHCL 的工具字典。四个阶段的可见工具数固定为：

| stage | visible tools | newly added tools |
| --- | ---: | ---: |
| base | 11,112 | 11,112 |
| task1 | 11,752 | 640 |
| task2 | 12,392 | 640 |
| task3 | 13,035 | 643 |

每次扩展 classifier 时，完整复制旧 query projection、旧 classifier weight rows 和 bias；只初始化新增工具行。训练阶段只读取当前阶段的 train split，不使用旧 query 回放。

### 2.2 EWC-DR

正常训练和评估始终使用原始 logits：

```text
L_task = CE(logits, gold_tool_id)
```

每个阶段结束后，EWC-DR 仅在计算参数重要性时执行 logits reversal：

```text
reversed_logits = -logits
L_importance = CE(reversed_logits, gold_tool_id)
Omega_i = mean_batches((dL_importance / dtheta_i)^2)
```

后续阶段的训练目标为：

```text
L = L_task + lambda / 2 * sum_i Omega_i * (theta_i - theta_i_old)^2
```

默认 `lambda=1000`、`gamma=1`、`omega_max=1e-4`。importance 使用固定种子 42 的 tool-ID coverage-first 子集，每个阶段最多 10,000 条 train 样本。base、task1、task2 计算 importance；task3 后没有 task4，默认跳过最终 importance。

受 EWC 保护的参数只有：

- `query_projection.layers.0.{weight,bias}`
- `query_projection.layers.3.{weight,bias}`
- `query_projection.layers.4.{weight,bias}`
- 历史 classifier 的 `weight` 和 `bias` 前缀

冻结的 LLaMA 参数 `requires_grad=False`，不参与 importance 或 EWC 正则。新增 classifier 行在当前阶段没有历史 EWC 约束。

### 2.3 与论文配套图像实现的差异

这是方法级迁移，不是逐行等价的图像实验复现：

| 维度 | 原始 EWC-DR 图像实现 | ToolHCL 迁移 |
| --- | --- | --- |
| backbone | 可训练 ResNet/IncrementalNet | 冻结完整 LLaMA-3-8B |
| 输出 | 图像类别 | 全局 ToolHCL tool ID |
| 当前任务 CE | 新类别 logits slice | 所有当前可见工具 logits |
| optimizer | SGD | AdamW |
| importance 数据 | 当前任务完整 train loader | 固定 seed，最多 10,000 条 train 样本 |
| importance 累积 | 类别比例融合 | online `Omega_old + Omega_current` |
| 评估 | classification accuracy | Recall@1/3/5、NDCG@1/3/5、MRR |

EWC-DR 最核心的 reversed-logits importance、参数快照和二次正则保持不变。论文或报告中应将该方法描述为 **EWC-DR adapted to the ToolHCL continual tool-retrieval protocol**，不应描述为官方图像代码的 exact reproduction。

## 3. 数据格式与目录

迁移代码读取以下 ToolHCL 文件：

```text
TOOLHCL_DATA_ROOT/
├── train/raw/
│   ├── retrieval_train.json
│   ├── retrieval_eval.json
│   └── train_tools_with_id.json
├── task1/raw/
│   ├── retrieval_train.json
│   ├── retrieval_eval.json
│   └── task1_tools_with_id.json
├── task2/raw/
│   ├── retrieval_train.json
│   ├── retrieval_eval.json
│   └── task2_tools_with_id.json
└── task3/raw/
    ├── retrieval_train.json
    ├── retrieval_eval.json
    └── task3_tools_with_id.json
```

检索样本使用 `conversations` 字段。代码读取 `role=user` 的 `content` 作为 query，并从 `role=assistant` 的 `<<tool_name&&api_name>>` 前缀解析 gold tool。工具字典必须提供连续、稳定的全局 `tool_id`，以及可解析的 `tool_name`/`api_name` 或等价文本描述。

global eval 是四个原始 eval split 的并集；每个 checkpoint 的候选集是该阶段全部 visible tools，不进行 candidate sampling 或 negative sampling。

## 4. 环境安装

推荐环境：

- Linux
- Python 3.10+
- CUDA-capable PyTorch 2.1+
- 单卡至少约 26 GiB 可用显存，用于首次完整 LLaMA 特征预计算

```bash
git clone git@github.com:2214331539/Tool_EWC-DR.git
cd Tool_EWC-DR

python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_toolhcl.txt
```

原始图像实验使用仓库根目录的 `requirements.txt`；ToolHCL 迁移使用更小的 `requirements_toolhcl.txt`。请按本机 CUDA 版本选择 PyTorch wheel，不要盲目复制 CUDA 12.1 命令。

## 5. 链接数据与模型

假设模型目录结构如下：

```text
TOOLHCL_MODELS_ROOT/
└── Meta-Llama-3-8B/
    ├── config.json
    ├── tokenizer.json
    ├── model.safetensors.index.json
    └── model-*.safetensors
```

创建本地软链接：

```bash
TOOLHCL_DATA_ROOT=/path/to/ToolHCL/data \
TOOLHCL_MODELS_ROOT=/path/to/models_hf \
TOOLHCL_ROOT=/path/to/ToolHCHL \
bash scripts/setup_toolhcl_links.sh
```

脚本创建：

```text
toolhcl_links/data     -> TOOLHCL_DATA_ROOT
toolhcl_links/models   -> TOOLHCL_MODELS_ROOT
toolhcl_links/ToolHCHL -> TOOLHCL_ROOT  # optional, training code does not import it
```

`toolhcl_links/` 被 Git 忽略。训练代码独立运行，不修改软链接目标中的任何数据、模型或 ToolHCL 源码。

## 6. 运行实验

### 6.1 Smoke test

Smoke test 使用真实 LLaMA、真实 ToolHCL 小样本、batch forward、reversed-logits importance 和小规模评估：

```bash
PYTHON_BIN="$PWD/.venv/bin/python" \
ARTIFACT_ROOT=/path/to/ewcdr_artifacts \
GPU_ID=0 \
bash scripts/smoke_test_toolhcl_ewcdr_pure.sh
```

不设置 `GPU_ID` 时，脚本会选择空闲显存至少 26,000 MiB 且利用率不高于 10% 的 GPU。

### 6.2 完整训练与评估

一次执行 base -> task1 -> task2 -> task3 和完整评估：

```bash
PYTHON_BIN="$PWD/.venv/bin/python" \
ARTIFACT_ROOT=/path/to/ewcdr_artifacts \
GPU_ID=0 \
bash scripts/run_pure_pipeline.sh
```

也可以拆分执行：

```bash
PYTHON_BIN="$PWD/.venv/bin/python" ARTIFACT_ROOT=/path/to/artifacts GPU_ID=0 \
  bash scripts/run_toolhcl_ewcdr_pure.sh

PYTHON_BIN="$PWD/.venv/bin/python" ARTIFACT_ROOT=/path/to/artifacts GPU_ID=0 \
  bash scripts/eval_toolhcl_ewcdr_pure.sh
```

直接调用 Python 入口：

```bash
python -m toolhcl_ewcdr_pure.train \
  --config configs/toolhcl_ewcdr_pure.yaml \
  --output_dir toolhcl_ewcdr_runs_pure \
  --method ewc_dr

python -m toolhcl_ewcdr_pure.evaluate \
  --config configs/toolhcl_ewcdr_pure.yaml \
  --output_dir toolhcl_ewcdr_runs_pure \
  --checkpoint_dir toolhcl_ewcdr_runs_pure/checkpoints
```

`--method` 支持：

- `seq_ft`: 顺序微调，不使用 EWC。
- `ewc`: 使用原始 logits 估计 importance。
- `ewc_dr`: 使用 reversed logits 估计 importance，正式 EWC-DR 入口。

为了公平比较，三种方法应使用相同模型、数据顺序、训练超参数、候选集和随机种子。

## 7. 特征缓存与运行时间

LLaMA 完全冻结，因此第一次运行会对所有 query 执行完整 LLaMA forward，并把最后 token hidden state 保存为 CPU BF16 分片。后续 epoch 直接读取这些确定性冻结特征，只训练 projection 和 classifier。

该缓存不会绕过任何可训练模块，在当前模型边界下与每个 epoch 重复执行相同冻结 LLaMA forward 等价。首次缓存通常是主要耗时；缓存建立后，每个训练 epoch 只需数秒。修改 tokenizer、encoder 路径、最大长度或输入样本后，缓存 manifest 校验会要求重建缓存。

## 8. 输出文件

每次正式运行生成：

```text
<run>/
├── checkpoints/{base,task1,task2,task3}.pt
├── importance/importance_{base,task1,task2}.pt
├── logs/{train,evaluate}.log
├── training_summary.json
├── metrics.json
├── eval_matrix.csv
├── global_eval.csv
└── summary.md
```

checkpoint、importance 和缓存体积较大，不提交到 Git。仓库只保留一组小型 CSV 参考结果，位于 `results/pure_sampling_fixed_20260716/`。

## 9. 已完成参考运行

配置：seed 42，每阶段 30 epochs，BF16，train batch 256，eval batch 512，importance 最多 10,000 条。该单次运行用于验证代码，不替代多随机种子的正式统计报告。

### Global eval

| checkpoint | candidates | R@1 | R@3 | R@5 | NDCG@1 | NDCG@3 | NDCG@5 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 11,112 | 17.0793 | 51.9509 | 62.1548 | 17.0793 | 37.5174 | 41.7473 | 36.4016 |
| task1 | 11,752 | 8.0503 | 19.1417 | 26.8213 | 8.0503 | 14.3926 | 17.5483 | 17.6411 |
| task2 | 12,392 | 6.6979 | 13.6628 | 19.2140 | 6.6979 | 10.6680 | 12.9466 | 14.0046 |
| task3 | 13,035 | 6.7623 | 13.9848 | 19.5140 | 6.7623 | 10.8786 | 13.1424 | 14.2684 |

task3 当前任务 Recall@1 为 77.3560，但历史任务遗忘仍然明显。该结果表明 EWC loss 确实非零并参与优化，不表示当前 `lambda=1000` 已是最优值。论文级报告还应补充相同架构下的 `seq_ft`、vanilla EWC、多个随机种子和验证集调参。

详细结果：

- [global_eval.csv](results/pure_sampling_fixed_20260716/global_eval.csv)
- [eval_matrix.csv](results/pure_sampling_fixed_20260716/eval_matrix.csv)

## 10. 代码结构

```text
toolhcl_ewcdr_pure/
├── data.py        # ToolHCL 字典、query/gold 解析和采样
├── model.py       # 冻结 LLaMA、query projection、累计 classifier
├── cache.py       # 完整 LLaMA hidden-state 分片缓存
├── ewcdr.py       # reversed-logits importance、snapshot、EWC penalty
├── train.py       # continual train loop
├── evaluate.py    # seen-task matrix 和 global eval
├── metrics.py     # Recall/NDCG/MRR
├── verify.py      # 结构、标签、importance 和 EWC smoke checks
└── precompute.py  # 可选缓存预计算入口
```

## 11. 原始图像 EWC-DR

原论文的 CIFAR-100、ImageNet-Subset 和 Tiny-ImageNet 配置仍保留在 `exps/`。例如：

```bash
python main.py --config=./exps/ewcdr_cifar_bigstart.json
```

原始图像实验环境见 `requirements.txt`，其数据准备方式请参考[上游仓库](https://github.com/scarlet0703/EWC-DR)。

## 12. License 与引用

EWC-DR 原始代码沿用 [Apache License 2.0](LICENSE.txt)。LLaMA 模型权重不随本仓库分发，并受 Meta Llama 3 Community License 约束。ToolHCL/ToolBench 数据的使用与再分发应遵循各自项目许可。

```bibtex
@inproceedings{liu2026elastic,
  title={Elastic Weight Consolidation Done Right for Continual Learning},
  author={Liu, Xuan and Chang, Xiaobin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```
