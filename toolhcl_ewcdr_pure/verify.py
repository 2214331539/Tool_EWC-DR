from __future__ import annotations

import argparse
import ast
import random
import tempfile
from pathlib import Path

import torch

from .data import (
    PureSample,
    TextDataset,
    build_stage_samples,
    load_stage_tools,
    make_loader,
    name_mapping,
    sample_by_tool,
    tool_key,
    visible_tools,
)
from .ewcdr import compute_importance, parameter_snapshot, regularization_loss
from .model import (
    build_retriever,
    initialize_from_previous,
    load_frozen_encoder,
    named_trainable_parameters,
    save_checkpoint,
)
from .utils import EXPECTED_TOOL_COUNTS, STAGES, dataloader_options, load_config, resolve_device, save_json, set_seed


EFFECTIVE_FILES = ("data.py", "cache.py", "model.py", "ewcdr.py", "metrics.py", "train.py")
DISALLOWED_IDENTIFIERS = (
    "prompt_pool",
    "soft_prompt",
    "target_l1",
    "target_l2",
    "geo_loss",
    "contrastive_loss",
    "m_global",
    "t_prev",
    "gate_mlp",
)


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            result.append(("." * node.level) + (node.module or ""))
    return result


def verify(config_path: str, output_path: str | None = None) -> dict:
    config = load_config(config_path)
    set_seed(int(config["training"].get("seed", 42)))
    device = resolve_device(config["runtime"].get("device", "cuda"), config["runtime"].get("gpu"))
    package_dir = Path(__file__).resolve().parent
    source_scan: dict[str, list[str]] = {}
    import_graph: dict[str, list[str]] = {}
    for filename in EFFECTIVE_FILES:
        path = package_dir / filename
        text = path.read_text(encoding="utf-8").lower()
        found = [value for value in DISALLOWED_IDENTIFIERS if value.lower() in text]
        if found:
            raise AssertionError(f"Disallowed identifiers in effective pure path {filename}: {found}")
        source_scan[filename] = found
        imports = _imports(path)
        forbidden_imports = [value for value in imports if "toolhcl_ewcdr." in value or value.startswith("models.") or value.startswith("training.")]
        if forbidden_imports:
            raise AssertionError(f"Legacy model/training imports in {filename}: {forbidden_imports}")
        import_graph[filename] = imports

    records = load_stage_tools(config)
    synthetic_samples = [
        PureSample(f"query-{tool_id}-{sample_id}", tool_id, f"tool-{tool_id}", "api", "base")
        for tool_id in range(100)
        for sample_id in range(3)
    ]
    sampled, sampled_indices, sampling_report = sample_by_tool(
        synthetic_samples, max_samples=80, seed=42
    )
    if len(sampled) != 80 or len(sampled_indices) != 80 or sampling_report["covered_tools"] != 80:
        raise AssertionError(f"Tool-coverage sampling verification failed: {sampling_report}")
    label_checks: dict[str, list[dict]] = {}
    rng = random.Random(42)
    for stage in STAGES:
        samples, _ = build_stage_samples(config, stage, "train", records=records, max_samples=2000)
        tools = visible_tools(stage, records)
        mapping, _ = name_mapping(tools)
        chosen = rng.sample(samples, min(20, len(samples)))
        rows = []
        for sample in chosen:
            assert 0 <= sample.tool_id < EXPECTED_TOOL_COUNTS[stage]
            assert mapping[tool_key(sample.tool_name, sample.api_name)] == sample.tool_id
            rows.append({"tool_name": sample.tool_name, "api_name": sample.api_name, "tool_id": sample.tool_id})
        label_checks[stage] = rows

    structure: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="pure_ewcdr_verify_") as temporary:
        previous_path = None
        for stage in STAGES:
            model = build_retriever(config, stage, encoder=None)
            names = list(named_trainable_parameters(model))
            if not names or any(not name.startswith(("query_projection.", "classifier.")) for name in names):
                raise AssertionError(f"Unexpected trainable parameters for {stage}: {names}")
            if previous_path is not None:
                initialize_from_previous(model, previous_path, stage, verify_exact=True)
            fake_hidden = torch.randn(2, int(config["model"]["hidden_size"]))
            logits = model.forward_hidden(fake_hidden)
            expected_shape = (2, EXPECTED_TOOL_COUNTS[stage])
            if tuple(logits.shape) != expected_shape:
                raise AssertionError(f"{stage} logits {tuple(logits.shape)}, expected {expected_shape}")
            structure[stage] = {"trainable": names, "logits_shape": list(logits.shape)}
            previous_path = Path(temporary) / f"{stage}.pt"
            save_checkpoint(previous_path, model, stage=stage, epoch=0, training_history=[], metadata={"verification": True})

        encoder = load_frozen_encoder(config, device)
        model = build_retriever(config, "base", encoder=encoder).to(device)
        samples, _ = build_stage_samples(config, "base", "train", records=records, max_samples=4)
        model.eval()
        with torch.no_grad():
            hidden = model.encode([sample.query_text for sample in samples[:2]])
            full_logits = model.forward_hidden(hidden)
        assert tuple(hidden.shape) == (2, 4096)
        assert tuple(full_logits.shape) == (2, 11112)
        loader = make_loader(
            TextDataset(samples), batch_size=2, shuffle=False, **dataloader_options(config, "importance")
        )
        importance, importance_report = compute_importance(
            model,
            loader,
            device,
            method="ewc_dr",
            max_batches=2,
            amp_enabled=True,
            amp_dtype=torch.bfloat16,
            accumulation_device=device,
            omega_max=float(config["ewcdr"].get("omega_max", 1e-4)),
            verify_unchanged=True,
        )
        snapshot = parameter_snapshot(model)
        zero_loss = regularization_loss(model, snapshot, importance, 1000.0)
        if abs(float(zero_loss.item())) > 1e-12:
            raise AssertionError(f"EWC loss at snapshot is not zero: {zero_loss.item()}")
        with torch.no_grad():
            model.query_projection.layers[0].weight[0, 0].add_(0.1)
            model.classifier.weight[0, 0].add_(0.1)
        perturbed_loss = regularization_loss(model, snapshot, importance, 1000.0)
        if not float(perturbed_loss.item()) > 0:
            raise AssertionError("EWC loss did not become positive after parameter perturbation")
        model.query_projection.load_state_dict({k.replace("query_projection.", ""): v for k, v in snapshot.items() if k.startswith("query_projection.")})
        model.classifier.load_state_dict({k.replace("classifier.", ""): v for k, v in snapshot.items() if k.startswith("classifier.")})
        restored_loss = regularization_loss(model, snapshot, importance, 1000.0)
        if abs(float(restored_loss.item())) > 1e-12:
            raise AssertionError(f"EWC loss after restore is not zero: {restored_loss.item()}")
        del model, encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report = {
        "status": "passed",
        "device": str(device),
        "source_scan": source_scan,
        "import_graph": import_graph,
        "sampling": sampling_report,
        "structure": structure,
        "label_checks": label_checks,
        "full_encoder_hidden_shape": [2, 4096],
        "importance": importance_report,
        "ewc_loss_at_snapshot": float(zero_loss.item()),
        "ewc_loss_after_perturbation": float(perturbed_loss.item()),
        "ewc_loss_after_restore": float(restored_loss.item()),
    }
    if output_path:
        save_json(output_path, report)
    print("PURE EWC-DR VERIFICATION PASSED")
    print("trainable parameters:", structure["base"]["trainable"])
    print("logits shapes:", {stage: values["logits_shape"] for stage, values in structure.items()})
    print("importance:")
    for row in importance_report["parameters"]:
        print(row)
    print("EWC losses:", report["ewc_loss_at_snapshot"], report["ewc_loss_after_perturbation"], report["ewc_loss_after_restore"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    verify(args.config, args.output)


if __name__ == "__main__":
    main()
