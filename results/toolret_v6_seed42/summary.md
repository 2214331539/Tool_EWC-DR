# Pure ToolHCL EWC-DR Baseline

## Method And Data Flow

`query -> complete frozen LLaMA forward -> last valid-token hidden state -> 4096->1024->384 query projection -> cumulative global linear classifier -> ranked global tool IDs`.

Frozen LLaMA hidden states are cached once and reused exactly across epochs. Normal training and evaluation use original logits. Post-stage importance alone uses `reversed_logits = -logits` over all classifier rows visible at the stage in `train` mode; squared gradients are accumulated in CPU fp32 and no optimizer update occurs during importance estimation. Historical and current importance use the official class-count alpha blend.

EWC protects every trainable query-projection tensor and the historical prefix of classifier weight/bias. The frozen LLaMA backbone is excluded because `requires_grad=False`; newly added classifier rows have no historical penalty until their stage importance is accumulated.

## Continual Protocol

Stages: base -> task1 -> task2 -> task3 -> task4. Cumulative visible tools: base=14,175, task1=16,557, task2=18,930, task3=21,291, task4=23,639. Incremental additions: base=+14,175, task1=+2,382, task2=+2,373, task3=+2,361, task4=+2,348.

Incremental task loss is a convex combination dominated by current-stage CE plus a weak all-visible calibration CE. The global term weight is the deterministic new-tool/visible-tool class ratio; evaluation still ranks unchanged original logits over every visible tool.

Data decontamination is enabled: every original eval row is preserved, while a train row is removed when its source_id or whitespace-normalized case-folded query occurs in any stage's eval split. The source files and visible candidate dictionary are unchanged.

## Training

| stage | epochs | final CE | final EWC | final total | avg epoch sec | importance sec | stop reason |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| base | 5 | 2.204263 | 0.000000 | 2.204263 | 79.750 | 87.9 | max_epochs_reached |
| task1 | 5 | 1.679393 | 0.192389 | 1.871782 | 19.895 | 28.9 | max_epochs_reached |
| task2 | 5 | 1.701304 | 0.219839 | 1.921143 | 18.303 | 24.8 | max_epochs_reached |
| task3 | 5 | 1.641989 | 0.208163 | 1.850152 | 13.395 | 28.1 | max_epochs_reached |
| task4 | 5 | 1.576193 | 0.203700 | 1.779893 | 15.851 | 0.0 | max_epochs_reached |

## Frozen Encoder Cache

Encoder batch size: 64; length-bucketed encoding: `True`. Length bucketing is used only for the frozen LLaMA forward; hidden states are written back in original sample order before training.

| cache split | samples | duration sec | shards |
| --- | ---: | ---: | ---: |
| base_train | 174613 | 21.4 | 43 |
| task1_train | 30432 | 0.6 | 8 |
| task2_train | 29162 | 0.4 | 8 |
| task3_train | 31233 | 0.7 | 8 |
| task4_train | 30205 | 4.0 | 8 |
| global_eval | 67338 | 1237.6 | 17 |

## Data Audit

| split | raw | parsed | decontamination removed | legacy eval removed | other excluded |
| --- | ---: | ---: | ---: | ---: | ---: |
| base_train | 220804 | 174613 | 46191 | 0 | 0 |
| task1_train | 38664 | 30432 | 8232 | 0 | 0 |
| task2_train | 36798 | 29162 | 7636 | 0 | 0 |
| task3_train | 39468 | 31233 | 8235 | 0 | 0 |
| task4_train | 38153 | 30205 | 7948 | 0 | 0 |
| base_eval | 39771 | 39771 | 0 | 0 | 0 |
| task1_eval | 6978 | 6978 | 0 | 0 | 0 |
| task2_eval | 6643 | 6643 | 0 | 0 | 0 |
| task3_eval | 7102 | 7102 | 0 | 0 | 0 |
| task4_eval | 6844 | 6844 | 0 | 0 | 0 |

Decontamination residuals: source_id=0, normalized_query=0. Eval unique source IDs=60,082; normalized queries=59,237.

## Importance

| stage | parameter | numel | nonzero | mean | max |
| --- | --- | ---: | ---: | ---: | ---: |
| base | query_projection.layers.0.weight | 4194304 | 4194304 | 1.15479e-05 | 0.0001 |
| base | query_projection.layers.0.bias | 1024 | 1024 | 6.85532e-06 | 0.0001 |
| base | query_projection.layers.3.weight | 393216 | 393216 | 2.3683e-05 | 0.0001 |
| base | query_projection.layers.3.bias | 384 | 384 | 7.63083e-05 | 0.0001 |
| base | query_projection.layers.4.weight | 384 | 384 | 9.99976e-05 | 0.0001 |
| base | query_projection.layers.4.bias | 384 | 384 | 6.14283e-05 | 0.0001 |
| base | classifier.weight | 5443200 | 5443200 | 1.45595e-06 | 0.0001 |
| base | classifier.bias | 14175 | 14175 | 5.881e-07 | 2.65804e-05 |
| task1 | query_projection.layers.0.weight | 4194304 | 4194293 | 1.08467e-05 | 0.0001 |
| task1 | query_projection.layers.0.bias | 1024 | 1024 | 7.60592e-06 | 0.0001 |
| task1 | query_projection.layers.3.weight | 393216 | 392448 | 2.44845e-05 | 0.0001 |
| task1 | query_projection.layers.3.bias | 384 | 384 | 8.73829e-05 | 0.0001 |
| task1 | query_projection.layers.4.weight | 384 | 384 | 0.0001 | 0.0001 |
| task1 | query_projection.layers.4.bias | 384 | 384 | 8.44339e-05 | 0.0001 |
| task1 | classifier.weight | 6357888 | 6357888 | 1.1976e-06 | 0.0001 |
| task1 | classifier.bias | 16557 | 16557 | 5.87077e-07 | 0.0001 |
| task2 | query_projection.layers.0.weight | 4194304 | 4194301 | 1.12719e-05 | 0.0001 |
| task2 | query_projection.layers.0.bias | 1024 | 1024 | 7.94957e-06 | 0.0001 |
| task2 | query_projection.layers.3.weight | 393216 | 391680 | 2.54426e-05 | 0.0001 |
| task2 | query_projection.layers.3.bias | 384 | 384 | 8.75302e-05 | 0.0001 |
| task2 | query_projection.layers.4.weight | 384 | 384 | 0.0001 | 0.0001 |
| task2 | query_projection.layers.4.bias | 384 | 384 | 8.18986e-05 | 0.0001 |
| task2 | classifier.weight | 7269120 | 7269120 | 1.06049e-06 | 0.0001 |
| task2 | classifier.bias | 18930 | 18930 | 5.08557e-07 | 0.0001 |
| task3 | query_projection.layers.0.weight | 4194304 | 4194298 | 1.26545e-05 | 0.0001 |
| task3 | query_projection.layers.0.bias | 1024 | 1024 | 8.74471e-06 | 0.0001 |
| task3 | query_projection.layers.3.weight | 393216 | 387072 | 2.64819e-05 | 0.0001 |
| task3 | query_projection.layers.3.bias | 384 | 384 | 9.11861e-05 | 0.0001 |
| task3 | query_projection.layers.4.weight | 384 | 384 | 9.9892e-05 | 0.0001 |
| task3 | query_projection.layers.4.bias | 384 | 384 | 8.28438e-05 | 0.0001 |
| task3 | classifier.weight | 8175744 | 8175744 | 9.17664e-07 | 0.0001 |
| task3 | classifier.bias | 21291 | 21291 | 4.49344e-07 | 0.0001 |

## Global Eval

| checkpoint | eval_split | samples | candidates | Recall@1 | Recall@3 | Recall@5 | NDCG@1 | NDCG@3 | NDCG@5 | MRR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | global | 67338 | 14175 | 22.9529 | 37.4157 | 41.7951 | 22.9529 | 31.4888 | 33.2996 | 31.4217 |
| task1 | global | 67338 | 16557 | 19.6293 | 34.5095 | 40.2952 | 19.6293 | 28.3612 | 30.7448 | 29.0781 |
| task2 | global | 67338 | 18930 | 17.6943 | 33.5353 | 40.3724 | 17.6943 | 26.9331 | 29.7492 | 28.1291 |
| task3 | global | 67338 | 21291 | 17.2399 | 33.9214 | 41.4120 | 17.2399 | 26.9683 | 30.0548 | 28.5913 |
| task4 | global | 67338 | 23639 | 17.6839 | 35.6530 | 43.8979 | 17.6839 | 28.1503 | 31.5495 | 30.1042 |

## Seen Task Matrix

| checkpoint | eval_split | samples | candidates | Recall@1 | Recall@3 | Recall@5 | NDCG@1 | NDCG@3 | NDCG@5 | MRR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | base | 39771 | 14175 | 38.8625 | 63.3502 | 70.7651 | 38.8625 | 53.3150 | 56.3811 | 53.2015 |
| task1 | base | 39771 | 16557 | 24.5330 | 46.3856 | 55.0376 | 24.5330 | 37.3522 | 40.9161 | 38.5313 |
| task1 | task1 | 6978 | 16557 | 49.5987 | 68.6443 | 75.1648 | 49.5987 | 60.7990 | 63.4879 | 60.9965 |
| task2 | base | 39771 | 18930 | 17.2362 | 36.2249 | 44.9976 | 17.2362 | 28.2939 | 31.9053 | 30.1964 |
| task2 | task1 | 6978 | 18930 | 26.3113 | 52.2356 | 61.9948 | 26.3113 | 41.4237 | 45.4421 | 42.0785 |
| task2 | task2 | 6643 | 18930 | 48.5323 | 68.1921 | 74.7253 | 48.5323 | 60.1065 | 62.8107 | 60.1521 |
| task3 | base | 39771 | 21291 | 12.7254 | 29.3807 | 37.7059 | 12.7254 | 22.4034 | 25.8291 | 24.7185 |
| task3 | task1 | 6978 | 21291 | 13.6859 | 36.2855 | 46.4746 | 13.6859 | 26.8226 | 31.0237 | 28.9236 |
| task3 | task2 | 6643 | 21291 | 28.1349 | 53.9214 | 63.2395 | 28.1349 | 43.2987 | 47.1567 | 43.8282 |
| task3 | task3 | 7102 | 21291 | 52.4359 | 71.0082 | 76.6826 | 52.4359 | 63.3886 | 65.7331 | 63.2524 |
| task4 | base | 39771 | 23639 | 11.4631 | 27.7790 | 35.6265 | 11.4631 | 20.9560 | 24.1888 | 23.2246 |
| task4 | task1 | 6978 | 23639 | 8.5841 | 25.5517 | 35.1247 | 8.5841 | 18.3194 | 22.2777 | 21.4497 |
| task4 | task2 | 6643 | 23639 | 14.0900 | 35.7218 | 46.5753 | 14.0900 | 26.5733 | 31.0392 | 28.9645 |
| task4 | task3 | 7102 | 23639 | 30.0056 | 54.8155 | 63.7848 | 30.0056 | 44.5639 | 48.2645 | 45.1263 |
| task4 | task4 | 6844 | 23639 | 53.8136 | 71.7563 | 77.6739 | 53.8136 | 64.4780 | 66.9260 | 64.4245 |

## Top-1 Prediction Stage Distribution

| checkpoint | eval split | base | task1 | task2 | task3 | task4 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| base | base | 100.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| task1 | base | 59.9608 | 40.0392 | 0.0000 | 0.0000 | 0.0000 |
| task1 | task1 | 25.8813 | 74.1187 | 0.0000 | 0.0000 | 0.0000 |
| task2 | base | 41.1908 | 11.7900 | 47.0192 | 0.0000 | 0.0000 |
| task2 | task1 | 20.0057 | 33.5340 | 46.4603 | 0.0000 | 0.0000 |
| task2 | task2 | 18.0190 | 6.8644 | 75.1167 | 0.0000 | 0.0000 |
| task3 | base | 28.7596 | 4.9961 | 15.2372 | 51.0070 | 0.0000 |
| task3 | task1 | 16.2654 | 16.8816 | 15.9071 | 50.9458 | 0.0000 |
| task3 | task2 | 15.1739 | 3.6730 | 37.8895 | 43.2636 | 0.0000 |
| task3 | task3 | 10.3633 | 2.3092 | 8.2794 | 79.0482 | 0.0000 |
| task4 | base | 26.8261 | 2.4918 | 5.2878 | 15.0864 | 50.3080 |
| task4 | task1 | 15.9931 | 10.3611 | 5.9616 | 15.8212 | 51.8630 |
| task4 | task2 | 16.6190 | 2.3634 | 18.0340 | 14.1201 | 48.8635 |
| task4 | task3 | 12.6302 | 1.7037 | 3.4216 | 38.8764 | 43.3681 |
| task4 | task4 | 9.7165 | 1.1397 | 2.2940 | 7.7440 | 79.1058 |

## Interpretation

Complete evaluation duration: 66.1 seconds.

Final `task4.pt` global: R@1=17.6839, R@3=35.6530, R@5=43.8979, NDCG@1=17.6839, NDCG@3=28.1503, NDCG@5=31.5495, MRR=30.1042.

Forgetting at the final checkpoint: base R@1 38.8625->11.4631, task1 R@1 49.5987->8.5841, task2 R@1 48.5323->14.0900, task3 R@1 52.4359->30.0056.

Standard average Recall@1 forgetting (best observed seen-task score minus final score): 31.3216 percentage points.
