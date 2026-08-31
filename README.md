# Tool EWC-DR V2

本仓库只保留 **V2 validated full-importance**：EWC-DR 在 ToolHCL 持续工具检索任务上的单一、可复现迁移实现。流程为 `base -> task1 -> task2 -> task3`，使用训练集内验证选择轮次，再从头使用完整训练集重训，最后才读取正式测试集。

本实现不修改或 import ToolHCL 的训练代码，不包含 ToolHCL 的 router、prompt pool、层级 box、geo loss、contrastive loss、replay 或 distillation。它只复用 ToolHCL 数据协议和 Meta-Llama-3-8B 模型资产。

## 来源与外部资产

- EWC-DR 论文：[Elastic Weight Consolidation Done Right for Continual Learning](https://arxiv.org/abs/2603.18596)
- EWC-DR 上游代码：[scarlet0703/EWC-DR](https://github.com/scarlet0703/EWC-DR)
- ToolHCL 项目：[2214331539/ToolHCHL](https://github.com/2214331539/ToolHCHL)
- ToolHCL base 数据来源：[OpenBMB/ToolBench](https://github.com/OpenBMB/ToolBench)
- 冻结查询编码器：[meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)

数据和模型受各自许可证约束，不提交到本仓库。使用者需要从 ToolHCL 项目准备 base/task1/task2/task3 数据，并自行获得 Meta-Llama-3-8B 权限和完整本地权重。

## V2 模型与数据流

```text
query text
  -> LLaMA tokenizer (right padding, max_length=512)
  -> 完整 32 层冻结 Meta-Llama-3-8B
  -> 最后一个有效 token 的 4096 维 hidden state
  -> trainable projection: 4096 -> 1024 -> 384
  -> 累积扩展的 global linear classifier
  -> 所有当前可见 tool logits
  -> 全局 tool ID 排名
```

LLaMA 的最终 hidden state 按样本缓存为 CPU BF16 分片。由于 encoder 完全冻结且处于 eval 模式，复用缓存与每轮重复执行同一个冻结 LLaMA forward 在当前模型边界下等价；projection 和 classifier 始终实时训练。

四阶段候选工具数量固定为：

| stage | visible tools | new tools |
| --- | ---: | ---: |
| base | 11,112 | 11,112 |
| task1 | 11,752 | 640 |
| task2 | 12,392 | 640 |
| task3 | 13,035 | 643 |

增量阶段会逐行精确复制历史 classifier 和完整 projection，只初始化新增 classifier 行。每个阶段只使用该阶段 train split，不回放历史 query。

## V2 EWC-DR 目标

正常训练在所有当前可见工具上使用原始 logits：

```text
L_task = CE(logits_all_visible, global_tool_id)
```

阶段结束后，只在 importance 计算中反转 logits：

```text
reversed_logits = -logits
L_importance = CE(reversed_logits, global_tool_id)
Omega_i = mean((dL_importance / dtheta_i)^2)
```

后续阶段训练目标：

```text
L = L_task + lambda/2 * sum_i Omega_i * (theta_i - theta_old_i)^2
```

V2 固定 `lambda=10000`、`gamma=1`、`omega_max=1e-4`。base、task1、task2 importance 使用各阶段完整训练分区；task3 后没有后续任务，因此不再计算 task3 importance。importance 和参数快照保存为 CPU FP32。

EWC 只保护 `requires_grad=True` 的参数：

- `query_projection.layers.{0,3,4}.{weight,bias}`
- 历史 classifier `weight` 和 `bias` 的公共前缀

冻结 LLaMA 不参与 importance 或 EWC。新增 classifier 行在被首次引入时没有历史 importance，因此当前阶段不受历史 EWC 惩罚。

与原始图像 EWC-DR 的主要差异是：V2 使用冻结 LLaMA、AdamW 和全局可见工具 CE；原图像实现使用可训练 ResNet、SGD 和新增类别 slice CE。核心 reversed-logits importance、参数快照和二次正则保持不变。因此应称为 **EWC-DR adapted to ToolHCL continual retrieval**，不是图像代码的逐行复现。

## 数据协议

期望的数据目录：

```text
TOOLHCL_DATA_ROOT/
├── train/raw/{retrieval_train.json,retrieval_eval.json,train_tools_with_id.json}
├── task1/raw/{retrieval_train.json,retrieval_eval.json,task1_tools_with_id.json}
├── task2/raw/{retrieval_train.json,retrieval_eval.json,task2_tools_with_id.json}
└── task3/raw/{retrieval_train.json,retrieval_eval.json,task3_tools_with_id.json}
```

样本从 `conversations` 中读取 `role=user` 的 `content` 作为 query，并从 `role=assistant` 的 `<<tool_name&&api_name>>` 前缀解析 gold tool。工具文件提供连续稳定的 global `tool_id`。正式 global eval 是四个完整 eval split 的并集；候选集是 checkpoint 阶段全部可见工具，不做负采样或候选裁剪。

V2 的解析样本数：base/task1/task2/task3 train 为 `232982/13157/13932/12989`，eval 为 `54348/3052/3206/3056`，global eval 共 `63662` 条。

## 统一收敛协议

正式多 seed 实验使用 [`configs/toolhcl_ewcdr_v2_converged.yaml`](configs/toolhcl_ewcdr_v2_converged.yaml)：

1. 每个 stage 使用 100% 原始 train split，至少训练 10 epochs，最多训练 60 epochs；较高上限只用于避免尚未满足收敛条件时被截断。
2. 每轮同时计算相邻 epoch 的 CE loss 和 total loss 相对变化率。
3. 两个变化率的最大值连续 3 轮不超过 5% 后才停止，最早只能在第 10 epoch 停止。
4. checkpoint 固定保存触发收敛时的最后一轮，不按 test 或每个 seed 的 validation 最佳点回退。
5. base 不加 EWC；task1/task2/task3 保持 reversed-logits importance 和在线 EWC 正则不变。

每轮日志和 `training_summary.json` 会记录 `relative_ce_loss_change`、
`relative_total_loss_change`、`convergence_relative_change` 和 `stop_reason`，用于审计是否满足标准。

## 历史验证选轮协议

1. 对每个 stage train split 按 tool ID、seed 42 固定划分约 90% train 和 10% validation；单样本 tool 只留在 train。
2. selection pass 最多训练 30 epochs。base 按自身 validation Recall@1 选轮次；增量阶段按“历史任务平均 Recall@1”和“当前任务 Recall@1”的调和均值选轮次。
3. selection 不构建或读取正式 eval/global eval 缓存。
4. 得到轮次后重新随机初始化模型，在 100% 原始 train split 上从头训练。
5. 全量重训结束后才运行完整 seen-task matrix 和 global eval。

seed 42 的已验证选择结果是 `base=7, task1=2, task2=1, task3=2`。
该协议保留用于复核旧结果，不再用于正式多 seed 收敛实验。

## 安装和资产链接

推荐 Linux、Python 3.10+、CUDA PyTorch 2.1+。首次 LLaMA 特征缓存建议单卡至少约 26 GiB 空闲显存。

```bash
git clone git@github.com:2214331539/Tool_EWC-DR.git
cd Tool_EWC-DR
python -m venv .venv
source .venv/bin/activate
# 按本机 CUDA 版本安装 PyTorch，然后：
pip install -r requirements_toolhcl.txt
```

模型目录应包含 `Meta-Llama-3-8B/config.json`、tokenizer 和全部 safetensors。创建只读资产软链接：

```bash
TOOLHCL_DATA_ROOT=/path/to/ToolHCL/data \
TOOLHCL_MODELS_ROOT=/path/to/models_hf \
TOOLHCL_ROOT=/path/to/ToolHCHL \
bash scripts/setup_toolhcl_links.sh
```

脚本创建 `toolhcl_links/data`、`toolhcl_links/models` 和可选的 `toolhcl_links/ToolHCHL`。这些链接及其目标不会被训练代码修改。

## 运行 V2

先运行单元测试：

```bash
python -m unittest discover -s tests -v
```

真实模型 smoke test：

```bash
PYTHON_BIN="$PWD/.venv/bin/python" \
ARTIFACT_ROOT=/path/to/ewcdr_artifacts \
GPU_ID=0 \
bash scripts/smoke_test_toolhcl_ewcdr_v2.sh
```

完整 selection、全量重训和正式评估：

```bash
PYTHON_BIN="$PWD/.venv/bin/python" \
ARTIFACT_ROOT=/path/to/ewcdr_artifacts \
GPU_ID=0 \
bash scripts/run_toolhcl_ewcdr_v2.sh
```

三个独立随机种子的完整运行可以并行启动。`--seed` 会同时控制模型初始化、
DataLoader/importance 采样和 tool-ID 分层验证划分；正式 eval split 与候选集保持不变：

```bash
PYTHON_BIN="$PWD/.venv/bin/python" \
ARTIFACT_ROOT=/path/to/ewcdr_artifacts \
SEEDS="42 43 44" GPU_IDS="0 1 2" \
bash scripts/run_toolhcl_ewcdr_v2_multiseed.sh
```

正式的统一 5% loss 收敛标准三 seed 运行：

```bash
PYTHON_BIN="$PWD/.venv/bin/python" \
ARTIFACT_ROOT=/path/to/ewcdr_artifacts \
SEEDS="42 43 44" GPU_IDS="0 1 2" \
bash scripts/run_toolhcl_ewcdr_v2_converged_multiseed.sh
```

也可以直接调用 `python -m toolhcl_ewcdr_pure.validated_pipeline --seed 43 ...`。

运行完成后可生成逐 seed 合并表和 mean/sample-std/min/max：

```bash
python scripts/summarize_toolhcl_ewcdr_v2_multiseed.py \
  --run 42=/path/to/seed42 --run 43=/path/to/seed43 --run 44=/path/to/seed44 \
  --output_dir /path/to/multiseed_results
```

未设置 `GPU_ID` 时脚本通过 `nvidia-smi` 选择满足显存和利用率阈值的 GPU。只重新评估现有运行：

```bash
PYTHON_BIN="$PWD/.venv/bin/python" GPU_ID=0 \
bash scripts/eval_toolhcl_ewcdr_v2.sh /path/to/run
```

正式收敛配置为 [`configs/toolhcl_ewcdr_v2_converged.yaml`](configs/toolhcl_ewcdr_v2_converged.yaml)；
[`configs/toolhcl_ewcdr_v2.yaml`](configs/toolhcl_ewcdr_v2.yaml) 仅用于复核历史 validation-selected V2。

## 输出

```text
<run>/
├── checkpoints/{base,task1,task2,task3}.pt
├── importance/importance_{base,task1,task2}.pt
├── selection/{config,training_summary}.json
├── selection_manifest.json
├── logs/{full_pipeline,train,evaluate}.log
├── config.json
├── training_summary.json
├── metrics.json
├── eval_matrix.csv
├── global_eval.csv
└── summary.md
```

checkpoint、importance、LLaMA 权重、ToolHCL 数据和特征缓存不提交 Git。seed 42 的轻量结果与审计文件保存在 [`results/v2_validated_seed42`](results/v2_validated_seed42)。

## 已验证结果

Global eval：

| checkpoint | R@1 | R@3 | R@5 | NDCG@1 | NDCG@3 | NDCG@5 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 20.4282 | 47.8166 | 57.1880 | 20.4282 | 36.4567 | 40.3335 | 36.4720 |
| task1 | 6.8047 | 14.4796 | 18.4003 | 6.8047 | 11.2538 | 12.8683 | 12.5827 |
| task2 | 2.8950 | 5.7350 | 7.2806 | 2.8950 | 4.5381 | 5.1744 | 5.3461 |
| task3 | 3.0709 | 6.2659 | 8.0912 | 3.0709 | 4.9191 | 5.6669 | 6.0120 |

完整 seen-task matrix、训练耗时、importance 统计和数据审计见 [`summary.md`](results/v2_validated_seed42/summary.md)。这是单个 seed 的已验证复现结果；论文级结论仍需相同协议下的 SeqFT、vanilla EWC 和多随机种子统计。

## 代码结构

```text
toolhcl_ewcdr_pure/
├── data.py               # ToolHCL 解析、全局 tool mapping、验证划分
├── model.py              # 冻结 LLaMA、projection、累计 classifier
├── cache.py              # 完整 LLaMA final hidden-state 缓存
├── ewcdr.py              # reversed-logits importance、snapshot、EWC loss
├── train.py              # selection/full-data 共用训练循环
├── validated_pipeline.py # selection -> fresh full-data retrain -> eval
├── evaluate.py           # seen matrix 和 global eval
├── metrics.py            # Recall/NDCG/MRR
├── verify.py             # 数据、结构、importance 和正则验证
└── precompute.py         # 可选缓存预计算
```

许可证见 [LICENSE.txt](LICENSE.txt)。ToolHCL 数据、ToolBench 数据和 Meta Llama 权重仍分别受其原始许可约束。
