from __future__ import annotations

import unittest

import torch

from toolhcl_ewcdr_pure.data import FeatureDataset, PureSample, stratified_train_validation_indices
from toolhcl_ewcdr_pure.train import _evaluate_seen_validation


class _IdentityLogitModel(torch.nn.Module):
    def __init__(self, num_tools: int):
        super().__init__()
        self.num_tools = num_tools

    def forward_batch(self, batch, device):
        return batch["query_hidden"].to(device)


class ValidatedProtocolTest(unittest.TestCase):
    def test_stratified_split_is_deterministic_and_keeps_all_tools_in_train(self):
        samples = []
        for tool_id, count in ((0, 1), (1, 2), (2, 10), (3, 25)):
            samples.extend(
                PureSample(f"q-{tool_id}-{index}", tool_id, "tool", "api", "base")
                for index in range(count)
            )

        first = stratified_train_validation_indices(samples, validation_fraction=0.1, seed=42)
        second = stratified_train_validation_indices(samples, validation_fraction=0.1, seed=42)
        self.assertEqual(first, second)
        train_indices, validation_indices, report = first
        self.assertFalse(set(train_indices) & set(validation_indices))
        self.assertEqual(len(train_indices) + len(validation_indices), len(samples))
        self.assertEqual({sample.tool_id for sample in samples}, {samples[index].tool_id for index in train_indices})
        self.assertNotIn(0, {samples[index].tool_id for index in validation_indices})
        self.assertEqual(report["singleton_tools_train_only"], 1)

    def test_validation_score_balances_old_and_current_recall(self):
        base = FeatureDataset(
            torch.tensor([[3.0, 1.0, 0.0], [0.0, 2.0, 3.0]]),
            torch.tensor([0, 0]),
            ["base", "base"],
        )
        task1 = FeatureDataset(
            torch.tensor([[0.0, 1.0, 3.0], [0.0, 1.0, 4.0]]),
            torch.tensor([2, 2]),
            ["task1", "task1"],
        )
        config = {
            "evaluation": {
                "batch_size": 2,
                "num_workers": 0,
                "pin_memory": False,
                "persistent_workers": False,
            }
        }
        result = _evaluate_seen_validation(
            _IdentityLogitModel(3),
            {"base": base, "task1": task1},
            ("base", "task1"),
            config,
            torch.device("cpu"),
            False,
            None,
        )
        self.assertEqual(result["historical_mean_recall_at_1"], 50.0)
        self.assertEqual(result["current_recall_at_1"], 100.0)
        self.assertAlmostEqual(result["selection_score"], 200.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
