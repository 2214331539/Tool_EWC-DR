from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from toolhcl_ewcdr_pure.ewcdr import accumulate_class_ratio, compute_importance


class _ToyRetriever(torch.nn.Module):
    def __init__(self, classes: int = 2) -> None:
        super().__init__()
        self.classifier = torch.nn.Linear(2, classes)
        self.num_tools = classes
        self.forward_training_modes = []

    def forward_batch(self, batch, device):
        self.forward_training_modes.append(self.training)
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

    def test_current_stage_importance_uses_only_active_classifier_rows(self):
        torch.manual_seed(7)
        model = _ToyRetriever(classes=4)
        batch = {
            "query_hidden": torch.tensor([[1.0, -0.5], [0.25, 2.0]]),
            "tool_id": torch.tensor([2, 3]),
        }

        importance, report = compute_importance(
            model,
            [batch],
            torch.device("cpu"),
            method="ewc_dr",
            first_class=2,
            verify_unchanged=True,
        )

        self.assertEqual(report["first_class"], 2)
        self.assertEqual(report["classification_candidates"], 2)
        self.assertEqual(int(torch.count_nonzero(importance["classifier.weight"][:2])), 0)
        self.assertEqual(int(torch.count_nonzero(importance["classifier.bias"][:2])), 0)
        self.assertGreater(int(torch.count_nonzero(importance["classifier.weight"][2:])), 0)
        self.assertGreater(int(torch.count_nonzero(importance["classifier.bias"][2:])), 0)

    def test_official_importance_uses_train_mode_and_restores_mode(self):
        model = _ToyRetriever(classes=3)
        model.eval()
        batch = {
            "query_hidden": torch.tensor([[0.5, 1.5]]),
            "tool_id": torch.tensor([2]),
        }

        _, report = compute_importance(
            model,
            [batch],
            torch.device("cpu"),
            method="ewc_dr",
            first_class=0,
            model_mode="train",
        )

        self.assertEqual(report["model_mode"], "train")
        self.assertEqual(model.forward_training_modes, [True])
        self.assertFalse(model.training)

    def test_official_class_ratio_accumulation_blends_only_common_prefix(self):
        previous = {"classifier.weight": torch.tensor([[2.0], [4.0]])}
        current = {"classifier.weight": torch.tensor([[10.0], [20.0], [30.0]])}

        result = accumulate_class_ratio(previous, current, alpha=0.25)

        expected = torch.tensor([[8.0], [16.0], [30.0]])
        torch.testing.assert_close(result["classifier.weight"], expected)


if __name__ == "__main__":
    unittest.main()
