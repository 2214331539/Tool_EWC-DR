# Pure ToolHCL EWC-DR Baseline

## Method Boundary

The archived implementation combined EWC-DR with ToolHCL-specific hierarchical routing, geometric boxes, dependency gating, and learned soft prompts. Those components are not part of EWC-DR. This pure implementation contains only a complete frozen LLaMA encoder, a trainable query projection, a cumulatively expanded global linear classifier, reversed-logits importance estimation, and online EWC regularization.

Effective import path: `toolhcl_ewcdr_pure.train -> data/cache/model/ewcdr` and `toolhcl_ewcdr_pure.evaluate -> data/cache/model/metrics`. It does not import the archived ToolHCL+EWC-DR package or ToolHCL model/training modules.

## Data Flow

`query text -> tokenizer -> all 32 frozen LLaMA Transformer layers -> last valid token hidden state (4096) -> query projection (4096->1024->384) -> global linear classifier -> tool logits -> ranked global tool IDs`.

Frozen encoder hidden states are cached after the complete LLaMA forward. Reusing deterministic frozen features across epochs is mathematically equivalent to rerunning the unchanged encoder and changes runtime only.

Normal training uses cross entropy on original logits with classification scope `all_visible`. Incremental CE uses every currently visible classifier row. This is the main ToolHCL task-agnostic retrieval protocol because train-time logits and inference candidates share the same global tool space. Logit negation is used only while estimating importance after a stage; it is never used for optimizer updates or evaluation.

## Continual Protocol

The visible classifier sizes are 11,112, 11,752, 12,392, and 13,035. At expansion, the full query projection and every old classifier row/bias are copied exactly; only added rows are initialized. EWC slices the common prefix, so added rows have no historical penalty. Each stage trains only its current train split and uses no replay, distillation, adapters, or external memory.

## Original-Code Deviations

The official image implementation trains ResNet-18 with SGD, computes importance on the full task train set, clips importance at 1e-4, blends old/current importance by a class-ratio alpha, and trains incremental CE over the new-class slice. This retrieval transfer uses frozen LLaMA plus a projection/classifier, the complete available stage-train partition, configured optional 1e-4 clipping, and gamma=1 online accumulation. Incremental CE covers all visible tools instead of the original new-class slice. This is an explicit ToolHCL adaptation needed for globally calibrated task-agnostic retrieval.

## Epoch Selection Protocol

A fixed-seed, tool-ID-stratified 10% validation partition is derived from each training split. The selection pass trains on the remaining 90% and chooses each stage epoch by the harmonic mean of historical-task mean Recall@1 and current-task Recall@1 over all visible candidates. No test split is used for epoch selection.

After selection, the model is restarted from scratch and every stage is retrained on 100% of its original training split for the selected number of epochs. Final metrics use the unchanged complete test splits and candidate sets.

Selected epochs: base=7, task1=2, task2=1, task3=2.

## Training

| stage | epochs | final CE | final EWC | final total | avg epoch sec | train sec | importance sec | stop reason |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| base | 7 | 2.473039 | 0.000000 | 2.473039 | 26.153 | 183.1 | 29.9 | max_epochs_reached |
| task1 | 2 | 5.295900 | 0.614142 | 5.910042 | 7.870 | 15.7 | 3.6 | max_epochs_reached |
| task2 | 1 | 6.784394 | 0.511494 | 7.295888 | 5.697 | 5.7 | 2.6 | max_epochs_reached |
| task3 | 2 | 5.241607 | 0.363873 | 5.605480 | 3.616 | 7.2 | 0.0 | max_epochs_reached |

### Importance Sampling

| stage | strategy | samples | covered tools | available tools | coverage |
| --- | --- | ---: | ---: | ---: | ---: |
| base | full_train_split | 232982 | 10889 | 10889 | 100.0000% |
| task1 | full_train_split | 13157 | 639 | 639 | 100.0000% |
| task2 | full_train_split | 13932 | 639 | 639 | 100.0000% |
| task3 | not computed (no later stage) | 0 | 0 | 0 | 0.0000% |

## Importance

| stage | parameter | numel | nonzero | mean | max | L1 norm | L2 norm |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | query_projection.layers.0.weight | 4194304 | 4185592 | 9.74173e-06 | 0.0001 | 40.8598 | 0.0539556 |
| base | query_projection.layers.0.bias | 1024 | 1022 | 4.97466e-06 | 0.0001 | 0.00509405 | 0.000473619 |
| base | query_projection.layers.3.weight | 393216 | 382464 | 2.30281e-05 | 0.0001 | 9.05503 | 0.029455 |
| base | query_projection.layers.3.bias | 384 | 384 | 5.46078e-05 | 0.0001 | 0.0209694 | 0.00122179 |
| base | query_projection.layers.4.weight | 384 | 384 | 9.89745e-05 | 0.0001 | 0.0380062 | 0.00194541 |
| base | query_projection.layers.4.bias | 384 | 384 | 3.20455e-05 | 0.0001 | 0.0123055 | 0.000680402 |
| base | classifier.weight | 4267008 | 4267008 | 2.30829e-06 | 7.7053e-05 | 9.8495 | 0.00759706 |
| base | classifier.bias | 11112 | 11112 | 7.18094e-07 | 5.43075e-06 | 0.00797946 | 9.32578e-05 |
| task1 | query_projection.layers.0.weight | 4194304 | 3825254 | 6.39522e-06 | 0.0001 | 26.8235 | 0.0454691 |
| task1 | query_projection.layers.0.bias | 1024 | 934 | 5.23823e-06 | 0.0001 | 0.00536394 | 0.000626371 |
| task1 | query_projection.layers.3.weight | 393216 | 257664 | 1.66312e-05 | 0.0001 | 6.53966 | 0.02427 |
| task1 | query_projection.layers.3.bias | 384 | 384 | 7.5127e-05 | 0.0001 | 0.0288488 | 0.00163351 |
| task1 | query_projection.layers.4.weight | 384 | 384 | 7.87514e-05 | 0.0001 | 0.0302405 | 0.00167751 |
| task1 | query_projection.layers.4.bias | 384 | 384 | 8.97559e-05 | 0.0001 | 0.0344663 | 0.00182956 |
| task1 | classifier.weight | 4512768 | 4512768 | 2.19822e-06 | 0.0001 | 9.92005 | 0.0257337 |
| task1 | classifier.bias | 11752 | 11752 | 1.05083e-06 | 0.0001 | 0.0123493 | 0.000592231 |
| task2 | query_projection.layers.0.weight | 4194304 | 3681423 | 6.1244e-06 | 0.0001 | 25.6876 | 0.0447888 |
| task2 | query_projection.layers.0.bias | 1024 | 898 | 5.109e-06 | 0.0001 | 0.00523161 | 0.000612397 |
| task2 | query_projection.layers.3.weight | 393216 | 248064 | 1.5423e-05 | 0.0001 | 6.06457 | 0.0232838 |
| task2 | query_projection.layers.3.bias | 384 | 384 | 7.18155e-05 | 0.0001 | 0.0275771 | 0.00158612 |
| task2 | query_projection.layers.4.weight | 384 | 384 | 7.20305e-05 | 0.0001 | 0.0276597 | 0.00159321 |
| task2 | query_projection.layers.4.bias | 384 | 384 | 8.80671e-05 | 0.0001 | 0.0338178 | 0.0018072 |
| task2 | classifier.weight | 4758528 | 4758528 | 2.16326e-06 | 0.0001 | 10.294 | 0.0266288 |
| task2 | classifier.bias | 12392 | 12392 | 1.05397e-06 | 0.0001 | 0.0130608 | 0.000633064 |

All trainable query-projection and classifier tensors must have nonzero importance; training aborts otherwise. Incremental-stage logs include EWC/CE ratio and total, old-classifier, and query-projection drift norms.

## Data Audit

Only query text, global tool ID, and source stage survive parsing. L1/L2 hierarchy fields are not present in model inputs, losses, importance, or inference.

| split | raw | parsed | excluded |
| --- | ---: | ---: | ---: |
| base_train | 232983 | 232982 | 1 |
| task1_train | 13159 | 13157 | 2 |
| task2_train | 13952 | 13932 | 20 |
| task3_train | 12989 | 12989 | 0 |
| base_eval | 54350 | 54348 | 2 |
| task1_eval | 3054 | 3052 | 2 |
| task2_eval | 3210 | 3206 | 4 |
| task3_eval | 3056 | 3056 | 0 |

Global evaluation uses 63,662 parsed queries. The normalized tool/API lookup contains 146 collision keys; these are recorded as a data-quality limitation rather than resolved with hierarchy labels.

## Global Eval

| checkpoint | eval_split | samples | candidates | Recall@1 | Recall@3 | Recall@5 | NDCG@1 | NDCG@3 | NDCG@5 | MRR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | global | 63662 | 11112 | 20.4282 | 47.8166 | 57.1880 | 20.4282 | 36.4567 | 40.3335 | 36.4720 |
| task1 | global | 63662 | 11752 | 6.8047 | 14.4796 | 18.4003 | 6.8047 | 11.2538 | 12.8683 | 12.5827 |
| task2 | global | 63662 | 12392 | 2.8950 | 5.7350 | 7.2806 | 2.8950 | 4.5381 | 5.1744 | 5.3461 |
| task3 | global | 63662 | 13035 | 3.0709 | 6.2659 | 8.0912 | 3.0709 | 4.9191 | 5.6669 | 6.0120 |

## Seen Task Matrix

| checkpoint | eval_split | samples | candidates | Recall@1 | Recall@3 | Recall@5 | NDCG@1 | NDCG@3 | NDCG@5 | MRR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | base | 54348 | 11112 | 23.9291 | 56.0113 | 66.9887 | 23.9291 | 42.7045 | 47.2457 | 42.7225 |
| task1 | base | 54348 | 11752 | 6.4823 | 14.4605 | 18.6612 | 6.4823 | 11.1060 | 12.8354 | 12.5886 |
| task1 | task1 | 3052 | 11752 | 26.5072 | 44.5282 | 51.5072 | 26.5072 | 36.9763 | 39.8585 | 38.2950 |
| task2 | base | 54348 | 12392 | 2.2945 | 4.8318 | 6.1217 | 2.2945 | 3.7670 | 4.2984 | 4.4483 |
| task2 | task1 | 3052 | 12392 | 7.8965 | 13.2045 | 16.7431 | 7.8965 | 10.9194 | 12.3673 | 12.7608 |
| task2 | task2 | 3206 | 12392 | 11.0730 | 19.4011 | 24.8596 | 11.0730 | 15.8619 | 18.1089 | 18.6038 |
| task3 | base | 54348 | 13035 | 1.9486 | 4.6110 | 6.2339 | 1.9486 | 3.4865 | 4.1500 | 4.5559 |
| task3 | task1 | 3052 | 13035 | 1.0157 | 2.8506 | 4.1940 | 1.0157 | 2.0404 | 2.5917 | 3.3276 |
| task3 | task2 | 3206 | 13035 | 0.0936 | 0.3431 | 0.8422 | 0.0936 | 0.2306 | 0.4332 | 1.1216 |
| task3 | task3 | 3056 | 13035 | 28.2068 | 45.3207 | 52.6178 | 28.2068 | 38.1904 | 41.2055 | 39.7182 |

## Top-1 Prediction Stage Distribution

| checkpoint | eval split | base | task1 | task2 | task3 |
| --- | --- | ---: | ---: | ---: | ---: |
| base | base | 100.0000 | 0.0000 | 0.0000 | 0.0000 |
| task1 | base | 24.0432 | 75.9568 | 0.0000 | 0.0000 |
| task1 | task1 | 10.8126 | 89.1874 | 0.0000 | 0.0000 |
| task2 | base | 7.7169 | 4.9864 | 87.2967 | 0.0000 |
| task2 | task1 | 3.4731 | 11.8283 | 84.6986 | 0.0000 |
| task2 | task2 | 4.8035 | 4.7411 | 90.4554 | 0.0000 |
| task3 | base | 6.3793 | 0.1693 | 0.0276 | 93.4239 |
| task3 | task1 | 3.4076 | 1.0485 | 0.0000 | 95.5439 |
| task3 | task2 | 5.0530 | 0.1248 | 0.0936 | 94.7286 |
| task3 | task3 | 2.6832 | 0.0654 | 0.0327 | 97.2186 |

## Interpretation

Evaluation duration: 13.0 seconds.

Current-task learning and old-task forgetting must be judged from the diagonal and lower-triangular matrix above. This is a method-clean EWC-DR baseline; publication-level claims still require matched sequential-finetuning and vanilla-EWC runs plus multiple random seeds.

Final task3 global: R@1=3.0709, R@3=6.2659, R@5=8.0912, NDCG@1=3.0709, NDCG@3=4.9191, NDCG@5=5.6669, MRR=6.0120.

## Required Method Audit

1. **ToolHCL-only modules in the archived implementation.** L1/L2 routers, L1/L2 boxes, L2 centers, dependency/gate modules, soft-prompt pool, gold-L2 prompt selection, geometric loss, and L2 contrastive/router loss belong to ToolHCL, not EWC-DR.
2. **Removal status.** The effective pure import graph contains none of those modules. The model has only the frozen complete LLaMA encoder, query projection, and global classifier; preflight source scanning and trainable-parameter assertions enforce this boundary.
3. **Complete data flow.** Query text is tokenized and right-padded, passed through all frozen LLaMA layers, reduced to the last valid token hidden state (4096), projected 4096->1024->384 with GELU/dropout/LayerNorm, then scored by one cumulative linear classifier. Descending logits map directly to global tool IDs.
4. **CE versus reversed logits.** Optimizer training and every evaluation use CE/ranking from original logits. Only post-stage importance estimation negates logits before CE; it performs backward for squared gradients but never optimizer.step().
5. **Protected parameters.** EWC protects query_projection layers 0/3/4 weights and biases plus the historical classifier weight/bias prefix. Frozen LLaMA parameters are excluded because requires_grad=False.
6. **Classifier expansion.** Visible rows are 11,112 -> 11,752 -> 12,392 -> 13,035. The complete projection and old classifier rows/bias are copied bit-exactly; only 640, 640, and 643 newly visible rows are initialized at task1, task2, and task3.
7. **New-row regularization.** New classifier rows are outside the common old/new tensor prefix and therefore receive no historical EWC penalty until their own stage importance is accumulated.
8. **Projection importance.** Every query-projection tensor has at least one nonzero importance element in base/task1/task2; the run would abort if an entire trainable tensor were zero.
9. **EWC is active.** Final-stage EWC losses are task1=0.614142, task2=0.511494, and task3=0.363873; they are not numerical zeros.
10. **Epochs.** base=7 (max_epochs_reached), task1=2 (max_epochs_reached), task2=1 (max_epochs_reached), task3=2 (max_epochs_reached).
11. **Current-task learning.** Diagonal Recall@1 is base=23.9291, task1=26.5072, task2=11.0730, and task3=28.2068. These values must be interpreted together with the prediction-stage distribution under all-visible global CE.
12. **Forgetting.** By task3, base Recall@1 changes 23.9291->1.9486, task1 26.5072->1.0157, and task2 11.0730->0.0936. Forgetting remains severe despite a measurable EWC penalty.
13. **Expected behavior.** Strong current-task accuracy together with substantial old-task loss is plausible for diagonal online EWC without replay in a 13k-class, highly imbalanced class-incremental problem. It is an empirical result, not evidence that lambda=10000 is universally optimal.
14. **Baseline suitability.** The implementation is method-clean enough to serve as an independent EWC-DR baseline. Publication-level comparison still requires the same architecture/hyperparameter protocol for sequential fine-tuning and vanilla EWC, multiple seeds, and disclosure that importance uses the complete available stage-train partition plus the recorded parser exclusions.
