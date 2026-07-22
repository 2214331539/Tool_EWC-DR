from __future__ import annotations

from typing import Any, Iterable

import torch

from .utils import EXPECTED_TOOL_COUNTS, STAGES


METRIC_NAMES = ("Recall@1", "Recall@3", "Recall@5", "NDCG@1", "NDCG@3", "NDCG@5", "MRR")
PREDICTION_STAGE_METRIC_NAMES = tuple(f"top1_pred_{stage}_percent" for stage in STAGES) + (
    "top1_pred_checkpoint_stage_percent",
)


class RetrievalMetrics:
    def __init__(self, topk: Iterable[int] = (1, 3, 5)):
        self.topk = tuple(sorted(int(value) for value in topk))
        self.samples = 0
        self.candidate_sum = 0
        self.recall = {value: 0.0 for value in self.topk}
        self.ndcg = {value: 0.0 for value in self.topk}
        self.mrr = 0.0

    def update(self, logits: torch.Tensor, targets: torch.Tensor, candidate_count: int) -> None:
        logits = logits.detach()
        targets = targets.detach().to(logits.device)
        batch_size = int(targets.numel())
        self.samples += batch_size
        self.candidate_sum += batch_size * int(candidate_count)
        valid = (targets >= 0) & (targets < int(candidate_count))
        if not bool(valid.any()):
            return
        valid_logits = logits[valid, :candidate_count]
        valid_targets = targets[valid].long()
        gold_scores = valid_logits.gather(1, valid_targets.view(-1, 1))
        ranks = (valid_logits > gold_scores).sum(dim=1).long() + 1
        for value in self.topk:
            hits = ranks <= value
            self.recall[value] += float(hits.sum().item())
            if bool(hits.any()):
                self.ndcg[value] += float((1.0 / torch.log2(ranks[hits].float() + 1.0)).sum().item())
        self.mrr += float((1.0 / ranks.float()).sum().item())

    def result(self) -> dict[str, Any]:
        denominator = max(1, self.samples)
        result: dict[str, Any] = {
            "samples": self.samples,
            "candidates": int(round(self.candidate_sum / denominator)) if self.samples else 0,
        }
        for value in self.topk:
            result[f"Recall@{value}"] = self.recall[value] / denominator * 100.0
        for value in self.topk:
            result[f"NDCG@{value}"] = self.ndcg[value] / denominator * 100.0
        result["MRR"] = self.mrr / denominator * 100.0
        return result


class PredictionStageMetrics:
    def __init__(self) -> None:
        self.samples = 0
        self.counts = {stage: 0 for stage in STAGES}

    def update(self, logits: torch.Tensor, candidate_count: int) -> None:
        predictions = logits.detach()[:, :candidate_count].argmax(dim=1)
        self.samples += int(predictions.numel())
        lower = 0
        for stage in STAGES:
            upper = EXPECTED_TOOL_COUNTS[stage]
            self.counts[stage] += int(((predictions >= lower) & (predictions < upper)).sum().item())
            lower = upper

    def result(self, checkpoint_stage: str) -> dict[str, float]:
        denominator = max(1, self.samples)
        result = {
            f"top1_pred_{stage}_percent": self.counts[stage] / denominator * 100.0
            for stage in STAGES
        }
        result["top1_pred_checkpoint_stage_percent"] = result[f"top1_pred_{checkpoint_stage}_percent"]
        return result
