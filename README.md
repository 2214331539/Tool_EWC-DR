# Tool EWC-DR: ToolRet V6

`toolret` 分支提供 EWC-DR 在 ToolRet Transaction 持续工具检索数据集上的 V6 迁移实现。协议为：

```text
base -> task1 -> task2 -> task3 -> task4
```

本实现不修改或 import ToolHCL 的训练模块。模型只包含完整冻结的 Meta-Llama-3-8B、可训练 query projection、累计扩展的 global classifier，以及 EWC-DR reversed-logits importance 和参数正则。

## 方法边界

原始 EWC-DR 图像代码在增量阶段只对新增类别 slice 计算 CE。直接迁移到 ToolRet 的全局工具检索会出现两个极端：

- `current-stage CE` 不直接压低旧工具，但没有学习跨阶段 logit 校准；
- `all-visible CE` 将每个旧工具持续作为负类，造成严重遗忘。

V6 使用 current-stage CE 为主，并加入由工具数量确定的弱 global calibration CE：

```text
beta_t = new_tools_t / visible_tools_t
L_task = (1 - beta_t) * CE(current_stage_logits, target)
       + beta_t * CE(all_visible_logits, target)
L = L_task + lambda/2 * sum_i Omega_i * (theta_i - theta_i_old)^2
```

base 阶段使用完整 base CE。Transaction 各增量阶段的 `beta_t` 固定为：

| stage | old tools | new tools | visible tools | beta |
| --- | ---: | ---: | ---: | ---: |
| task1 | 14,175 | 2,382 | 16,557 | 0.143867 |
| task2 | 16,557 | 2,373 | 18,930 | 0.125357 |
| task3 | 18,930 | 2,361 | 21,291 | 0.110892 |
| task4 | 21,291 | 2,348 | 23,639 | 0.099327 |

这些权重只由协议中的工具数量确定，不读取 eval/test 指标。V6 是面向 ToolRet 全局检索协议的 calibrated EWC-DR adaptation；严格论文版 current-stage CE 应作为独立 baseline 报告。

## EWC-DR

正常训练和评估始终使用原始 logits。阶段训练结束后，仅在 importance 计算中执行：

```text
reversed_logits = -logits
L_importance = CE(reversed_logits, global_tool_id)
Omega_i = mean((dL_importance / dtheta_i)^2)
```

V6 使用：

- `lambda=10000`；
- 完整阶段训练集估计 importance；
- `omega_max=1e-4`；
- EWC-DR 官方 class-count alpha 累积；
- importance 模型处于 train mode；
- CPU FP32 importance 和参数 snapshot；
- task4 后不计算无后续用途的 final importance。

EWC 只保护 `requires_grad=True` 的 query projection 和历史 classifier 公共前缀。冻结 LLaMA 不参与 EWC；新增 classifier 行在首次引入时没有历史 importance。

## 模型数据流

```text
query text
  -> LLaMA tokenizer (right padding, max_length=512)
  -> 完整 32 层冻结 Meta-Llama-3-8B
  -> 最后一个有效 token 的 4096 维 hidden state
  -> query projection: 4096 -> 1024 -> 384
  -> cumulative global linear classifier
  -> 当前 checkpoint 的全部可见工具 logits
  -> 按原始 logits 降序检索 global tool ID
```

冻结 LLaMA 的最终 hidden state 按样本缓存为 CPU BF16 分片。projection 和 classifier 每轮实时训练。缓存只避免重复执行完全相同的冻结 encoder forward，不改变算法输出。

## 外部资产

- EWC-DR 论文：[Elastic Weight Consolidation Done Right for Continual Learning](https://arxiv.org/abs/2603.18596)
- EWC-DR 上游代码：[scarlet0703/EWC-DR](https://github.com/scarlet0703/EWC-DR)
- ToolHCL 项目：[2214331539/ToolHCHL](https://github.com/2214331539/ToolHCHL)
- ToolHCL base 数据来源：[OpenBMB/ToolBench](https://github.com/OpenBMB/ToolBench)
- 冻结 encoder：[meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)

数据和模型受各自许可证约束，不提交到本仓库。使用者需要自行获得 Meta-Llama-3-8B 权限和 ToolRet Transaction 数据。

## 数据协议

Transaction 根目录需要包含：

```text
Transaction/
├── base/raw/{retrieval_train.json,retrieval_eval.json,train_tools_with_id.json}
├── task1/raw/{retrieval_train.json,retrieval_eval.json,toolret_task1_tools_with_id.json}
├── task2/raw/{retrieval_train.json,retrieval_eval.json,toolret_task2_tools_with_id.json}
├── task3/raw/{retrieval_train.json,retrieval_eval.json,toolret_task3_tools_with_id.json}
└── task4/raw/{retrieval_train.json,retrieval_eval.json,toolret_task4_tools_with_id.json}
```

配置从 `target_tool_id` 读取连续 global label。训练前汇总全部五个原始 eval split 的 `source_id` 和规范化 query；原始 eval 全部保留，从 train 中移除命中任一 eval key 的记录。源 JSON 和候选工具字典不被修改。

V6 实际样本数：

| stage | clean train | complete eval | cumulative candidates |
| --- | ---: | ---: | ---: |
| base | 174,613 | 39,771 | 14,175 |
| task1 | 30,432 | 6,978 | 16,557 |
| task2 | 29,162 | 6,643 | 18,930 |
| task3 | 31,233 | 7,102 | 21,291 |
| task4 | 30,205 | 6,844 | 23,639 |

global eval 是五个完整 eval split 的并集，共 67,338 条，不裁剪候选集。

## 安装

推荐 Linux、Python 3.10+ 和 CUDA PyTorch。首次构建 LLaMA 特征缓存建议单卡至少 30 GiB 空闲显存。

```bash
git clone --branch toolret git@github.com:2214331539/Tool_EWC-DR.git
cd Tool_EWC-DR
python -m venv .venv
source .venv/bin/activate
# 按本机 CUDA 安装 PyTorch，然后：
pip install -r requirements_toolhcl.txt
```

创建只读数据和模型软链接：

```bash
TOOLHCL_DATA_ROOT=/path/to/original/toolhcl_data \
TOOLHCL_MODELS_ROOT=/path/to/models_hf \
TRANSACTION_DATA_ROOT=/path/to/Transaction \
bash scripts/setup_toolhcl_links.sh
```

模型应位于 `toolhcl_links/models/Meta-Llama-3-8B`。Transaction 数据链接位于 `toolhcl_links/transaction`。链接和目标文件不会被训练代码修改。

## 运行

单元测试：

```bash
python -m unittest discover -s tests -v
```

真实模型 smoke test（base -> task1，少量样本）：

```bash
PYTHON_BIN="$PWD/.venv/bin/python" GPU_ID=0 \
bash scripts/smoke_test_toolhcl_ewcdr_transaction_calibrated_currentce.sh
```

完整 5-stage、每阶段 5 epoch 训练和评估：

```bash
PYTHON_BIN="$PWD/.venv/bin/python" GPU_ID=0 \
TRANSACTION_ARTIFACT_ROOT=/path/to/artifacts/transaction \
bash scripts/run_toolhcl_ewcdr_transaction_calibrated_currentce_5epoch.sh
```

未指定 `GPU_ID` 时，脚本通过 `nvidia-smi` 选择满足空闲显存和利用率阈值的 GPU。重新评估现有运行：

```bash
PYTHON_BIN="$PWD/.venv/bin/python" GPU_ID=0 \
bash scripts/eval_toolhcl_ewcdr_transaction_calibrated_currentce.sh /path/to/run
```

正式配置为 [`configs/toolhcl_ewcdr_transaction_calibrated_currentce_5epoch.yaml`](configs/toolhcl_ewcdr_transaction_calibrated_currentce_5epoch.yaml)。配置使用仓库相对路径；大型缓存和运行产物默认写入 `artifacts/transaction`，也可通过脚本环境变量重定向。

## 输出

```text
<run>/
├── checkpoints/{base,task1,task2,task3,task4}.pt
├── importance/importance_{base,task1,task2,task3}.pt
├── logs/{full_pipeline,train,evaluate}.log
├── config.json
├── training_summary.json
├── metrics.json
├── eval_matrix.csv
├── global_eval.csv
└── summary.md
```

checkpoint、importance、模型、数据和特征缓存由 `.gitignore` 排除。

## Seed 42 结果

V6 的 `task4.pt`：

| eval split | R@1 | R@3 | R@5 | NDCG@1 | NDCG@3 | NDCG@5 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 11.4631 | 27.7790 | 35.6265 | 11.4631 | 20.9560 | 24.1888 | 23.2246 |
| task1 | 8.5841 | 25.5517 | 35.1247 | 8.5841 | 18.3194 | 22.2777 | 21.4497 |
| task2 | 14.0900 | 35.7218 | 46.5753 | 14.0900 | 26.5733 | 31.0392 | 28.9645 |
| task3 | 30.0056 | 54.8155 | 63.7848 | 30.0056 | 44.5639 | 48.2645 | 45.1263 |
| task4 | 53.8136 | 71.7563 | 77.6739 | 53.8136 | 64.4780 | 66.9260 | 64.4245 |

最终 global eval：R@1 `17.6839`、R@3 `35.6530`、R@5 `43.8979`、MRR `30.1042`。最终五任务宏平均 R@1 为 `23.5913`；平均 R@1 forgetting 为 `31.3216` 个百分点。

完整轻量结果见 [`results/toolret_v6_seed42`](results/toolret_v6_seed42)。这是单 seed 实验；论文级结论仍需 matched SeqFT、vanilla EWC、严格 current-CE EWC-DR 和多个随机种子。

## 代码结构

```text
toolhcl_ewcdr_pure/
├── data.py       # Transaction 解析、去泄露、global tool ID
├── cache.py      # 完整冻结 LLaMA final hidden-state 缓存
├── model.py      # query projection 和累计 classifier
├── ewcdr.py      # reversed-logits importance、snapshot、EWC loss
├── train.py      # V6 calibrated current-stage CE 和阶段训练
├── evaluate.py   # seen matrix、global eval、结果汇总
├── metrics.py    # Recall/NDCG/MRR 和确定性 tie ranking
└── verify.py     # 数据、模型、importance 和正则审计
```

许可证见 [LICENSE.txt](LICENSE.txt)。
