from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from toolhcl_ewcdr_pure.train import classification_loss


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


if __name__ == "__main__":
    unittest.main()
