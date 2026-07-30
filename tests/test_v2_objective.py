from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from toolhcl_ewcdr_pure.train import classification_loss
from toolhcl_ewcdr_pure.metrics import RetrievalMetrics


class V2ObjectiveTest(unittest.TestCase):
    def test_ce_uses_every_visible_tool_and_updates_old_and_new_rows(self):
        old_count = 11_112
        total_count = 11_752
        logits = torch.randn(2, total_count, requires_grad=True)
        targets = torch.tensor([old_count, total_count - 1])

        loss, candidates = classification_loss(logits, targets)
        expected = F.cross_entropy(logits.float(), targets)
        self.assertEqual(candidates, total_count)
        torch.testing.assert_close(loss, expected)

        loss.backward()
        self.assertGreater(int(torch.count_nonzero(logits.grad[:, :old_count]).item()), 0)
        self.assertGreater(int(torch.count_nonzero(logits.grad[:, old_count:]).item()), 0)

    def test_metrics_break_exact_ties_by_tool_id(self):
        metrics = RetrievalMetrics((1, 3))
        logits = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
        metrics.update(logits, torch.tensor([0, 1]), candidate_count=3)

        result = metrics.result()
        self.assertEqual(result["Recall@1"], 50.0)
        self.assertEqual(result["Recall@3"], 100.0)
        self.assertAlmostEqual(result["MRR"], 75.0)

    def test_current_stage_ce_does_not_push_historical_rows(self):
        old_count = 4
        logits = torch.randn(3, 7, requires_grad=True)
        targets = torch.tensor([4, 5, 6])

        loss, candidates = classification_loss(logits, targets, first_class=old_count)
        expected = F.cross_entropy(logits[:, old_count:].float(), targets - old_count)
        self.assertEqual(candidates, 3)
        torch.testing.assert_close(loss, expected)

        loss.backward()
        self.assertEqual(int(torch.count_nonzero(logits.grad[:, :old_count]).item()), 0)
        self.assertGreater(int(torch.count_nonzero(logits.grad[:, old_count:]).item()), 0)

    def test_calibrated_current_ce_is_expected_convex_combination(self):
        old_count = 4
        weight = 0.2
        logits = torch.randn(3, 7, requires_grad=True)
        targets = torch.tensor([4, 5, 6])

        loss, candidates = classification_loss(
            logits,
            targets,
            first_class=old_count,
            global_ce_weight=weight,
        )
        current = F.cross_entropy(logits[:, old_count:].float(), targets - old_count)
        global_loss = F.cross_entropy(logits.float(), targets)
        expected = (1.0 - weight) * current + weight * global_loss
        self.assertEqual(candidates, 7)
        torch.testing.assert_close(loss, expected)

        loss.backward()
        self.assertGreater(int(torch.count_nonzero(logits.grad[:, :old_count]).item()), 0)


if __name__ == "__main__":
    unittest.main()
