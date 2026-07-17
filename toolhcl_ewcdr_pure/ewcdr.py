from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from .model import PureEwcDrRetriever, named_trainable_parameters
from .utils import autocast_context, common_prefix_slices


def parameter_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().float().clone() for name, parameter in named_trainable_parameters(model).items()}


def _assert_parameters_equal(before: Mapping[str, torch.Tensor], model: nn.Module) -> None:
    after = named_trainable_parameters(model)
    if set(before) != set(after):
        raise AssertionError("Trainable parameter names changed during importance computation")
    for name, expected in before.items():
        torch.testing.assert_close(after[name].detach().cpu().float(), expected, rtol=0, atol=0)


def compute_importance(
    model: PureEwcDrRetriever,
    dataloader,
    device: torch.device,
    *,
    method: str = "ewc_dr",
    max_batches: int | None = None,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype | None = None,
    accumulation_device: str | torch.device = "cpu",
    omega_max: float | None = None,
    verify_unchanged: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if method not in {"ewc", "ewc_dr"}:
        raise ValueError(f"Unsupported importance method: {method}")
    parameters = named_trainable_parameters(model)
    target_device = torch.device(accumulation_device)
    if target_device.type == "cuda" and device.type != "cuda":
        target_device = torch.device("cpu")
    importance = {
        name: torch.zeros_like(parameter.detach(), dtype=torch.float32, device=target_device)
        for name, parameter in parameters.items()
    }
    before = parameter_snapshot(model) if verify_unchanged else None
    was_training = model.training
    model.eval()
    batches = 0
    samples = 0
    for batch_index, batch in enumerate(dataloader):
        if max_batches is not None and batch_index >= int(max_batches):
            break
        model.zero_grad(set_to_none=True)
        with autocast_context(device, amp_enabled, amp_dtype):
            logits = model.forward_batch(batch, device)
        if method == "ewc_dr":
            logits = -logits
        targets = batch["tool_id"].to(device, non_blocking=True)
        loss = F.cross_entropy(logits.float(), targets).float()
        loss.backward()
        for name, parameter in parameters.items():
            if parameter.grad is not None:
                importance[name].add_(parameter.grad.detach().float().pow(2).to(target_device))
        batches += 1
        samples += int(targets.numel())
    denominator = max(1, batches)
    for name, value in list(importance.items()):
        value = value.div(float(denominator))
        if omega_max is not None and float(omega_max) > 0:
            value.clamp_(max=float(omega_max))
        importance[name] = value.cpu().float()
    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()
    if before is not None:
        _assert_parameters_equal(before, model)
    summary = importance_summary(importance)
    if any(row["nonzero"] == 0 for row in summary):
        zero_names = [row["name"] for row in summary if row["nonzero"] == 0]
        raise RuntimeError(f"Importance is entirely zero for trainable parameters: {zero_names}")
    return importance, {"batches": batches, "samples": samples, "parameters": summary}


def accumulate_online(
    previous: Mapping[str, torch.Tensor] | None,
    current: Mapping[str, torch.Tensor],
    gamma: float,
) -> dict[str, torch.Tensor]:
    total = {name: value.detach().cpu().float().clone() for name, value in current.items()}
    if not previous:
        return total
    for name, old_value in previous.items():
        if name not in total:
            continue
        common = common_prefix_slices(total[name].shape, old_value.shape)
        if common is not None:
            total[name][common].add_(old_value[common].float(), alpha=float(gamma))
    return total


def regularization_loss(
    model: nn.Module,
    snapshot: Mapping[str, torch.Tensor] | None,
    importance: Mapping[str, torch.Tensor] | None,
    lambda_ewc: float,
) -> torch.Tensor:
    device = next(model.parameters()).device
    result = torch.zeros((), dtype=torch.float32, device=device)
    if not snapshot or not importance or float(lambda_ewc) == 0:
        return result
    for name, parameter in named_trainable_parameters(model).items():
        if name not in snapshot or name not in importance:
            continue
        common = common_prefix_slices(parameter.shape, snapshot[name].shape)
        if common is None:
            continue
        old = snapshot[name][common].to(device=device, dtype=torch.float32)
        omega = importance[name][common].to(device=device, dtype=torch.float32)
        result = result + torch.sum(omega * (parameter[common].float() - old).pow(2))
    return 0.5 * float(lambda_ewc) * result


def drift_summary(model: nn.Module, snapshot: Mapping[str, torch.Tensor] | None) -> dict[str, float]:
    if not snapshot:
        return {"parameter_drift_l2": 0.0, "old_classifier_drift_l2": 0.0, "query_projection_drift_l2": 0.0}
    sums = {"all": 0.0, "classifier": 0.0, "projection": 0.0}
    for name, parameter in named_trainable_parameters(model).items():
        if name not in snapshot:
            continue
        common = common_prefix_slices(parameter.shape, snapshot[name].shape)
        if common is None:
            continue
        difference = parameter[common].detach().cpu().float() - snapshot[name][common].float()
        squared = float(difference.pow(2).sum().item())
        sums["all"] += squared
        if name.startswith("classifier."):
            sums["classifier"] += squared
        if name.startswith("query_projection."):
            sums["projection"] += squared
    return {
        "parameter_drift_l2": sums["all"] ** 0.5,
        "old_classifier_drift_l2": sums["classifier"] ** 0.5,
        "query_projection_drift_l2": sums["projection"] ** 0.5,
    }


def importance_summary(importance: Mapping[str, torch.Tensor]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, tensor in importance.items():
        value = tensor.detach().cpu().float()
        rows.append(
            {
                "name": name,
                "numel": int(value.numel()),
                "nonzero": int(torch.count_nonzero(value).item()),
                "mean": float(value.mean().item()),
                "max": float(value.max().item()),
                "l1_norm": float(value.abs().sum().item()),
                "l2_norm": float(value.norm(p=2).item()),
            }
        )
    return rows


def save_importance(
    path: str | Path,
    *,
    stage: str,
    method: str,
    gamma: float,
    current: Mapping[str, torch.Tensor],
    total: Mapping[str, torch.Tensor],
    snapshot: Mapping[str, torch.Tensor],
    metadata: Mapping[str, Any],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "toolhcl_pure_ewcdr_importance_v1",
            "stage": stage,
            "method": method,
            "gamma": float(gamma),
            "importance_current": {k: v.cpu().float() for k, v in current.items()},
            "importance_total": {k: v.cpu().float() for k, v in total.items()},
            "snapshot": {k: v.cpu().float() for k, v in snapshot.items()},
            "metadata": dict(metadata),
        },
        target,
    )


def load_importance(path: str | Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)
