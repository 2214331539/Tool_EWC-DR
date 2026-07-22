from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from toolhcl_ewcdr_pure.ewcdr import compute_importance


class _ToyRetriever(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = torch.nn.Linear(2, 2)

    def forward_batch(self, batch, device):
        return self.classifier(batch["query_hidden"].to(device))


class EwcDrMechanismTest(unittest.TestCase):
    def test_importance_uses_reversed_logits_and_keeps_parameters_unchanged(self):
        torch.manual_seed(42)
        model = _ToyRetriever()
        batch = {
            "query_hidden": torch.tensor([[1.0, 2.0], [-1.0, 0.5]]),
            "tool_id": torch.tensor([0, 1]),
        }
        before = {name: value.detach().clone() for name, value in model.named_parameters()}

        model.zero_grad(set_to_none=True)
        expected_loss = F.cross_entropy(-model.forward_batch(batch, torch.device("cpu")), batch["tool_id"])
        expected_loss.backward()
        expected = {
            name: parameter.grad.detach().float().pow(2).clone()
            for name, parameter in model.named_parameters()
        }
        model.zero_grad(set_to_none=True)

        importance, report = compute_importance(
            model,
            [batch],
            torch.device("cpu"),
            method="ewc_dr",
            verify_unchanged=True,
        )

        self.assertEqual(report["batches"], 1)
        self.assertEqual(report["samples"], 2)
        for name, value in importance.items():
            torch.testing.assert_close(value, expected[name])
            torch.testing.assert_close(model.state_dict()[name], before[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
