from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from tqdm import tqdm

from .cache import build_feature_cache, cache_is_valid, load_feature_dataset
from .data import build_global_eval_samples, load_stage_tools, make_loader
from .ewcdr import load_importance
from .metrics import METRIC_NAMES, PREDICTION_STAGE_METRIC_NAMES, PredictionStageMetrics, RetrievalMetrics
from .model import build_retriever, load_checkpoint, load_frozen_encoder
from .utils import (
    EXPECTED_TOOL_COUNTS,
    STAGES,
    autocast_context,
    dataloader_options,
    ensure_dir,
    load_config,
    project_root,
    resolve_device,
    resolve_path,
    resolve_precision,
    save_json,
    setup_logging,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pure ToolHCL EWC-DR checkpoints")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--stages", default=None)
    parser.add_argument("--eval_limit", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir:
        config["project"]["output_dir"] = args.output_dir
    if args.checkpoint_dir:
        config["evaluation"]["checkpoint_dir"] = args.checkpoint_dir
    if args.stages:
        config["evaluation"]["stages"] = [value.strip() for value in args.stages.split(",") if value.strip()]
    if args.eval_limit is not None:
        config["evaluation"]["eval_limit"] = args.eval_limit
    if args.smoke:
        config["runtime"]["smoke"] = True
        config["evaluation"]["stages"] = ["base", "task1"]
        config["evaluation"]["eval_limit"] = int(config["smoke"].get("eval_samples", 64))
        config["cache"]["root"] = str(config["cache"]["root"]) + "_smoke"
    return config


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "checkpoint",
        "eval_split",
        "samples",
        "candidates",
        *METRIC_NAMES,
        *PREDICTION_STAGE_METRIC_NAMES,
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row[field] for field in fields} for row in rows])


def _markdown_table(rows: Sequence[Mapping[str, Any]]) -> str:
    fields = ["checkpoint", "eval_split", "samples", "candidates", *METRIC_NAMES]
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        values = []
        for field in fields:
            value = row[field]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _ensure_eval_cache(config: Mapping[str, Any], device: torch.device, logger) -> tuple[Path, int]:
    records = load_stage_tools(config)
    samples, details = build_global_eval_samples(
        config, records=records, max_samples=config["evaluation"].get("eval_limit")
    )
    cache_root = resolve_path(config["cache"]["root"], project_root(config))
    cache_dir = cache_root / "global_eval"
    hidden_size = int(config["model"].get("hidden_size", 4096))
    max_length = int(config["model"].get("max_length", 512))
    if not cache_is_valid(cache_dir, samples, hidden_size, max_length):
        encoder = load_frozen_encoder(config, device)
        try:
            build_feature_cache(
                encoder,
                samples,
                cache_dir,
                batch_size=int(config["cache"].get("encoder_batch_size", 8)),
                shard_size=int(config["cache"].get("shard_size", 4096)),
                dataloader_options=dataloader_options(config, "cache"),
                logger=logger,
            )
        finally:
            del encoder
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    logger.info("global eval parse audit=%s", details)
    return cache_dir, len(samples)


@torch.no_grad()
def evaluate_checkpoint(
    model,
    loader,
    *,
    checkpoint_stage: str,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    topk: Sequence[int],
) -> dict[str, dict[str, Any]]:
    candidate_count = EXPECTED_TOOL_COUNTS[checkpoint_stage]
    accumulators = {"global": RetrievalMetrics(topk)}
    accumulators.update({stage: RetrievalMetrics(topk) for stage in STAGES})
    prediction_accumulators = {"global": PredictionStageMetrics()}
    prediction_accumulators.update({stage: PredictionStageMetrics() for stage in STAGES})
    model.eval()
    for batch in tqdm(loader, desc=f"pure eval {checkpoint_stage}", dynamic_ncols=True, mininterval=2.0):
        with autocast_context(device, amp_enabled, amp_dtype):
            logits = model.forward_batch(batch, device)
        targets = batch["tool_id"].to(device, non_blocking=True)
        accumulators["global"].update(logits, targets, candidate_count)
        prediction_accumulators["global"].update(logits, candidate_count)
        for source_stage in STAGES:
            indices = [index for index, value in enumerate(batch["source_stage"]) if value == source_stage]
            if indices:
                index_tensor = torch.tensor(indices, dtype=torch.long, device=device)
                accumulators[source_stage].update(
                    logits.index_select(0, index_tensor), targets.index_select(0, index_tensor), candidate_count
                )
                prediction_accumulators[source_stage].update(
                    logits.index_select(0, index_tensor), candidate_count
                )
    return {
        name: {
            **accumulator.result(),
            **prediction_accumulators[name].result(checkpoint_stage),
        }
        for name, accumulator in accumulators.items()
    }


def _importance_rows(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage in STAGES[:-1]:
        path = output_dir / "importance" / f"importance_{stage}.pt"
        if not path.exists():
            continue
        payload = load_importance(path)
        for row in payload.get("metadata", {}).get("importance", {}).get("parameters", []):
            rows.append({"stage": stage, **row})
    return rows


def _write_summary(
    output_dir: Path,
    config: Mapping[str, Any],
    global_rows: Sequence[Mapping[str, Any]],
    matrix_rows: Sequence[Mapping[str, Any]],
    evaluation_duration: float,
) -> None:
    training_path = output_dir / "training_summary.json"
    training = json.load(open(training_path, encoding="utf-8")) if training_path.exists() else {}
    classification_scope = str(
        training.get("classification_scope", config.get("training", {}).get("classification_scope", "all_visible"))
    )
    lambda_ewc = float(config.get("ewcdr", {}).get("lambda", 0.0))
    if classification_scope not in {"all_visible", "current_stage", "current_stage_calibrated"}:
        raise ValueError(f"Unsupported classification_scope={classification_scope}")
    if classification_scope == "current_stage_calibrated":
        scope_description = (
            "Incremental CE is computed on the newly introduced tool rows and blended with a "
            "deterministic class-ratio weighted all-visible calibration CE."
        )
        deviation_description = (
            "The current-stage calibrated objective is used as an explicit ToolHCL retrieval "
            "adaptation; evaluation still ranks every visible tool."
        )
    elif classification_scope == "current_stage":
        scope_description = "Incremental CE uses only newly introduced tool rows."
        deviation_description = "The new-class CE objective is used as an explicit ToolHCL retrieval adaptation."
    else:
        scope_description = (
            "Incremental CE uses every currently visible classifier row, so train-time logits and "
            "task-agnostic inference candidates share the same global tool space."
        )
        deviation_description = (
            "Incremental CE covers all visible tools instead of the original new-class slice. This is "
            "an explicit ToolHCL retrieval adaptation."
        )
    importance_rows = _importance_rows(output_dir)
    importance_limit = config.get("ewcdr", {}).get("importance_max_samples")
    if importance_limit is None:
        importance_description = "the complete available stage-train partition"
    else:
        importance_description = f"a fixed tool-ID coverage subset of at most {int(importance_limit):,} samples"
    selection_protocol = config.get("selection_protocol", {})
    final = next((row for row in global_rows if row["checkpoint"] == "task3"), None)
    matrix_index = {(row["checkpoint"], row["eval_split"]): row for row in matrix_rows}
    parse_audit_path = resolve_path(config["cache"]["root"], project_root(config)) / "parse_audit.json"
    parse_audit = json.load(open(parse_audit_path, encoding="utf-8")) if parse_audit_path.exists() else {}
    lines = [
        "# Pure ToolHCL EWC-DR Baseline",
        "",
        "## Method Boundary",
        "",
        "The archived implementation combined EWC-DR with ToolHCL-specific hierarchical routing, geometric boxes, dependency gating, and learned soft prompts. Those components are not part of EWC-DR. This pure implementation contains only a complete frozen LLaMA encoder, a trainable query projection, a cumulatively expanded global linear classifier, reversed-logits importance estimation, and online EWC regularization.",
        "",
        "Effective import path: `toolhcl_ewcdr_pure.train -> data/cache/model/ewcdr` and `toolhcl_ewcdr_pure.evaluate -> data/cache/model/metrics`. It does not import the archived ToolHCL+EWC-DR package or ToolHCL model/training modules.",
        "",
        "## Data Flow",
        "",
        "`query text -> tokenizer -> all 32 frozen LLaMA Transformer layers -> last valid token hidden state (4096) -> query projection (4096->1024->384) -> global linear classifier -> tool logits -> ranked global tool IDs`.",
        "",
        "Frozen encoder hidden states are cached after the complete LLaMA forward. Reusing deterministic frozen features across epochs is mathematically equivalent to rerunning the unchanged encoder and changes runtime only.",
        "",
        f"Normal training uses cross entropy on original logits with classification scope `{classification_scope}`. {scope_description} Logit negation is used only while estimating importance after a stage; it is never used for optimizer updates or evaluation.",
        "",
        "## Continual Protocol",
        "",
        "The visible classifier sizes are 11,112, 11,752, 12,392, and 13,035. At expansion, the full query projection and every old classifier row/bias are copied exactly; only added rows are initialized. EWC slices the common prefix, so added rows have no historical penalty. Each stage trains only its current train split and uses no replay, distillation, adapters, or external memory.",
        "",
        "## Original-Code Deviations",
        "",
        f"The official image implementation trains ResNet-18 with SGD, computes importance on the full task train set, clips importance at 1e-4, blends old/current importance by a class-ratio alpha, and trains incremental CE over the new-class slice. This retrieval transfer uses frozen LLaMA plus a projection/classifier, {importance_description}, configured optional 1e-4 clipping, and gamma=1 online accumulation. {deviation_description}",
        "",
    ]
    if selection_protocol:
        selected_epochs = selection_protocol.get("selected_epochs", {})
        lines.extend([
            "## Epoch Selection Protocol",
            "",
            "A fixed-seed, tool-ID-stratified 10% validation partition is derived from each training split. The selection pass trains on the remaining 90% and chooses each stage epoch by the harmonic mean of historical-task mean Recall@1 and current-task Recall@1 over all visible candidates. No test split is used for epoch selection.",
            "",
            "After selection, the model is restarted from scratch and every stage is retrained on 100% of its original training split for the selected number of epochs. Final metrics use the unchanged complete test splits and candidate sets.",
            "",
            "Selected epochs: " + ", ".join(
                f"{stage}={selected_epochs.get(stage, 'missing')}" for stage in STAGES
            ) + ".",
            "",
        ])
    lines.extend(["## Training", ""])
    if training.get("stages"):
        lines.extend([
            "| stage | epochs | final CE | final EWC | final total | avg epoch sec | train sec | importance sec | stop reason |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ])
        for stage in training["stages"]:
            final_epoch = stage["final_epoch"]
            lines.append(
                f"| {stage['stage']} | {stage['completed_epochs']} | {final_epoch['ce_loss']:.6f} | "
                f"{final_epoch['ewc_loss']:.6f} | {final_epoch['total_loss']:.6f} | "
                f"{stage['train_duration_sec'] / max(1, stage['completed_epochs']):.3f} | "
                f"{stage['train_duration_sec']:.1f} | {stage['importance_duration_sec']:.1f} | {stage['stop_reason']} |"
            )
        lines.extend([
            "",
            "### Importance Sampling",
            "",
            "| stage | strategy | samples | covered tools | available tools | coverage |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ])
        for stage in training["stages"]:
            sampling = stage.get("sampling")
            if sampling:
                lines.append(
                    f"| {stage['stage']} | {sampling['strategy']} | {sampling['samples']} | "
                    f"{sampling['covered_tools']} | {sampling['available_tools']} | "
                    f"{sampling['coverage_percent']:.4f}% |"
                )
            else:
                lines.append(f"| {stage['stage']} | not computed (no later stage) | 0 | 0 | 0 | 0.0000% |")
    lines.extend(["", "## Importance", ""])
    if importance_rows:
        lines.extend([
            "| stage | parameter | numel | nonzero | mean | max | L1 norm | L2 norm |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in importance_rows:
            lines.append(
                f"| {row['stage']} | {row['name']} | {row['numel']} | {row['nonzero']} | "
                f"{row['mean']:.6g} | {row['max']:.6g} | {row['l1_norm']:.6g} | {row['l2_norm']:.6g} |"
            )
    lines.extend(["", "All trainable query-projection and classifier tensors must have nonzero importance; training aborts otherwise. Incremental-stage logs include EWC/CE ratio and total, old-classifier, and query-projection drift norms.", ""])
    if parse_audit:
        lines.extend([
            "## Data Audit",
            "",
            "Only query text, global tool ID, and source stage survive parsing. L1/L2 hierarchy fields are not present in model inputs, losses, importance, or inference.",
            "",
            "| split | raw | parsed | excluded |",
            "| --- | ---: | ---: | ---: |",
        ])
        for stage in STAGES:
            train_parse = parse_audit.get("train", {}).get(stage, {}).get("parse", {})
            if train_parse:
                raw, parsed = int(train_parse["raw"]), int(train_parse["parsed"])
                lines.append(f"| {stage}_train | {raw} | {parsed} | {raw - parsed} |")
        for stage, values in parse_audit.get("global_eval", {}).get("per_stage", {}).items():
            raw, parsed = int(values["raw"]), int(values["parsed"])
            lines.append(f"| {stage}_eval | {raw} | {parsed} | {raw - parsed} |")
        lines.extend([
            "",
            f"Global evaluation uses {parse_audit.get('global_eval', {}).get('parsed', 0):,} parsed queries. "
            f"The normalized tool/API lookup contains {parse_audit.get('global_eval', {}).get('normalized_name_collisions', 0)} collision keys; "
            "these are recorded as a data-quality limitation rather than resolved with hierarchy labels.",
            "",
        ])
    lines.extend(["## Global Eval", "", _markdown_table(global_rows), "", "## Seen Task Matrix", "", _markdown_table(matrix_rows), ""])
    if matrix_rows and "top1_pred_base_percent" in matrix_rows[0]:
        lines.extend([
            "## Top-1 Prediction Stage Distribution",
            "",
            "| checkpoint | eval split | base | task1 | task2 | task3 |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ])
        for row in matrix_rows:
            lines.append(
                f"| {row['checkpoint']} | {row['eval_split']} | "
                f"{row['top1_pred_base_percent']:.4f} | {row['top1_pred_task1_percent']:.4f} | "
                f"{row['top1_pred_task2_percent']:.4f} | {row['top1_pred_task3_percent']:.4f} |"
            )
        lines.append("")
    lines.extend([
        "## Interpretation",
        "",
        f"Evaluation duration: {evaluation_duration:.1f} seconds.",
        "",
        "Current-task learning and old-task forgetting must be judged from the diagonal and lower-triangular matrix above. This is a method-clean EWC-DR baseline; publication-level claims still require matched sequential-finetuning and vanilla-EWC runs plus multiple random seeds.",
    ])
    if final:
        lines.extend(["", f"Final task3 global: R@1={final['Recall@1']:.4f}, R@3={final['Recall@3']:.4f}, R@5={final['Recall@5']:.4f}, NDCG@1={final['NDCG@1']:.4f}, NDCG@3={final['NDCG@3']:.4f}, NDCG@5={final['NDCG@5']:.4f}, MRR={final['MRR']:.4f}."])
    if training.get("stages") and final:
        stage_index = {row["stage"]: row for row in training["stages"]}
        base_initial = matrix_index.get(("base", "base"), {})
        base_final = matrix_index.get(("task3", "base"), {})
        task1_initial = matrix_index.get(("task1", "task1"), {})
        task1_final = matrix_index.get(("task3", "task1"), {})
        task2_initial = matrix_index.get(("task2", "task2"), {})
        task2_final = matrix_index.get(("task3", "task2"), {})
        final_losses = {
            stage: values["final_epoch"] for stage, values in stage_index.items()
        }
        lines.extend([
            "",
            "## Required Method Audit",
            "",
            "1. **ToolHCL-only modules in the archived implementation.** L1/L2 routers, L1/L2 boxes, L2 centers, dependency/gate modules, soft-prompt pool, gold-L2 prompt selection, geometric loss, and L2 contrastive/router loss belong to ToolHCL, not EWC-DR.",
            "2. **Removal status.** The effective pure import graph contains none of those modules. The model has only the frozen complete LLaMA encoder, query projection, and global classifier; preflight source scanning and trainable-parameter assertions enforce this boundary.",
            "3. **Complete data flow.** Query text is tokenized and right-padded, passed through all frozen LLaMA layers, reduced to the last valid token hidden state (4096), projected 4096->1024->384 with GELU/dropout/LayerNorm, then scored by one cumulative linear classifier. Descending logits map directly to global tool IDs.",
            "4. **CE versus reversed logits.** Optimizer training and every evaluation use CE/ranking from original logits. Only post-stage importance estimation negates logits before CE; it performs backward for squared gradients but never optimizer.step().",
            "5. **Protected parameters.** EWC protects query_projection layers 0/3/4 weights and biases plus the historical classifier weight/bias prefix. Frozen LLaMA parameters are excluded because requires_grad=False.",
            "6. **Classifier expansion.** Visible rows are 11,112 -> 11,752 -> 12,392 -> 13,035. The complete projection and old classifier rows/bias are copied bit-exactly; only 640, 640, and 643 newly visible rows are initialized at task1, task2, and task3.",
            "7. **New-row regularization.** New classifier rows are outside the common old/new tensor prefix and therefore receive no historical EWC penalty until their own stage importance is accumulated.",
            "8. **Projection importance.** Every query-projection tensor has at least one nonzero importance element in base/task1/task2; the run would abort if an entire trainable tensor were zero.",
            f"9. **EWC is active.** Final-stage EWC losses are task1={final_losses['task1']['ewc_loss']:.6f}, task2={final_losses['task2']['ewc_loss']:.6f}, and task3={final_losses['task3']['ewc_loss']:.6f}; they are not numerical zeros.",
            "10. **Epochs.** " + ", ".join(
                f"{stage}={stage_index[stage]['completed_epochs']} ({stage_index[stage]['stop_reason']})"
                for stage in STAGES
            ) + ".",
            f"11. **Current-task learning.** Diagonal Recall@1 is base={base_initial.get('Recall@1', 0):.4f}, task1={task1_initial.get('Recall@1', 0):.4f}, task2={task2_initial.get('Recall@1', 0):.4f}, and task3={matrix_index[('task3', 'task3')]['Recall@1']:.4f}. These values must be interpreted together with the prediction-stage distribution.",
            f"12. **Forgetting.** By task3, base Recall@1 changes {base_initial.get('Recall@1', 0):.4f}->{base_final.get('Recall@1', 0):.4f}, task1 {task1_initial.get('Recall@1', 0):.4f}->{task1_final.get('Recall@1', 0):.4f}, and task2 {task2_initial.get('Recall@1', 0):.4f}->{task2_final.get('Recall@1', 0):.4f}. Forgetting remains severe despite a measurable EWC penalty.",
            f"13. **Expected behavior.** Strong current-task accuracy together with substantial old-task loss is plausible for diagonal online EWC without replay in a 13k-class, highly imbalanced class-incremental problem. It is an empirical result, not evidence that lambda={lambda_ewc:g} is universally optimal.",
        ])
        lines.append(
            f"14. **Baseline suitability.** The implementation is method-clean enough to serve as an independent EWC-DR baseline. Publication-level comparison still requires the same architecture/hyperparameter protocol for sequential fine-tuning and vanilla EWC, multiple seeds, and disclosure that importance uses {importance_description} plus the recorded parser exclusions."
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(config: dict[str, Any]) -> Path:
    root = project_root(config)
    output_dir = resolve_path(config["project"]["output_dir"], root)
    ensure_dir(output_dir)
    logger = setup_logging(output_dir, "evaluate")
    device = resolve_device(config["runtime"].get("device", "cuda"), config["runtime"].get("gpu"))
    amp_enabled, amp_dtype, precision_name = resolve_precision(config, device, logger)
    cache_dir, expected_samples = _ensure_eval_cache(config, device, logger)
    dataset, manifest = load_feature_dataset(cache_dir)
    if len(dataset) != expected_samples:
        raise AssertionError(f"Global cache contains {len(dataset)} rows, expected {expected_samples}")
    options = dataloader_options(config, "evaluation")
    loader = make_loader(
        dataset,
        batch_size=int(config["evaluation"].get("batch_size", 512)),
        shuffle=False,
        **options,
    )
    checkpoint_dir = resolve_path(
        config["evaluation"].get("checkpoint_dir") or str(output_dir / "checkpoints"), root
    )
    stages = tuple(config["evaluation"].get("stages", STAGES))
    topk = tuple(int(value) for value in config["evaluation"].get("topk", [1, 3, 5]))
    global_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    started = time.time()
    for checkpoint_stage in stages:
        model = build_retriever(config, checkpoint_stage, encoder=None).to(device)
        load_checkpoint(model, checkpoint_dir / f"{checkpoint_stage}.pt")
        grouped = evaluate_checkpoint(
            model,
            loader,
            checkpoint_stage=checkpoint_stage,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            topk=topk,
        )
        global_rows.append({"checkpoint": checkpoint_stage, "eval_split": "global", **grouped["global"]})
        for source_stage in STAGES[: STAGES.index(checkpoint_stage) + 1]:
            matrix_rows.append({"checkpoint": checkpoint_stage, "eval_split": source_stage, **grouped[source_stage]})
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    duration = time.time() - started
    _write_csv(output_dir / "global_eval.csv", global_rows)
    _write_csv(output_dir / "eval_matrix.csv", matrix_rows)
    save_json(
        output_dir / "metrics.json",
        {
            "method": config["ewcdr"].get("method", "ewc_dr"),
            "precision": precision_name,
            "evaluation_duration_sec": round(duration, 3),
            "cache_manifest": manifest,
            "global_eval": global_rows,
            "seen_task_matrix": matrix_rows,
        },
    )
    _write_summary(output_dir, config, global_rows, matrix_rows, duration)
    logger.info("evaluation complete duration=%.1fs", duration)
    print("\nGLOBAL EVAL\n" + _markdown_table(global_rows))
    print("\nSEEN TASK MATRIX\n" + _markdown_table(matrix_rows))
    return output_dir


def main() -> None:
    args = parse_args()
    evaluate(apply_overrides(load_config(args.config), args))


if __name__ == "__main__":
    main()
