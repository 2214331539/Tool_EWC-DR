from __future__ import annotations

import argparse
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Subset
from tqdm import tqdm

from .cache import build_feature_cache, cache_is_valid, load_feature_dataset
from .data import (
    FeatureDataset,
    build_global_eval_samples,
    build_stage_samples,
    load_stage_tools,
    make_loader,
    sample_by_tool,
    stratified_train_validation_indices,
)
from .ewcdr import (
    accumulate_online,
    compute_importance,
    drift_summary,
    load_importance,
    parameter_snapshot,
    regularization_loss,
    save_importance,
)
from .model import (
    build_retriever,
    initialize_from_previous,
    load_checkpoint,
    load_frozen_encoder,
    named_trainable_parameters,
    save_checkpoint,
    trainable_summary,
)
from .metrics import RetrievalMetrics
from .utils import (
    EXPECTED_OLD_TOOL_COUNTS,
    STAGES,
    autocast_context,
    dataloader_options,
    ensure_dir,
    gpu_memory_summary,
    load_config,
    load_json,
    project_root,
    resolve_device,
    resolve_path,
    resolve_precision,
    save_json,
    set_seed,
    setup_logging,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the pure ToolHCL EWC-DR baseline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--method", choices=("seq_ft", "ewc", "ewc_dr"), default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stages", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--importance_max_samples", type=int, default=None)
    parser.add_argument("--ewc_lambda", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir:
        config["project"]["output_dir"] = args.output_dir
    if args.method:
        config["ewcdr"]["method"] = args.method
    if args.resume:
        config["runtime"]["resume"] = True
    if args.stages:
        config["training"]["stages"] = [value.strip() for value in args.stages.split(",") if value.strip()]
    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError("--epochs must be positive")
        config["training"]["stage_epochs"] = {
            stage: {"min": args.epochs, "max": args.epochs} for stage in STAGES
        }
    if args.max_train_samples is not None:
        config["training"]["max_train_samples"] = args.max_train_samples
    if args.importance_max_samples is not None:
        config["ewcdr"]["importance_max_samples"] = args.importance_max_samples
    if args.ewc_lambda is not None:
        if args.ewc_lambda < 0:
            raise ValueError("--ewc_lambda must be non-negative")
        config["ewcdr"]["lambda"] = args.ewc_lambda
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    if args.smoke:
        config["runtime"]["smoke"] = True
        config["training"]["stages"] = ["base", "task1"]
        config["training"]["max_train_samples"] = int(config["smoke"].get("train_samples_per_stage", 32))
        config["training"]["batch_size"] = int(config["smoke"].get("batch_size", 4))
        config["training"]["stage_epochs"] = {
            "base": {"min": 1, "max": 1},
            "task1": {"min": 1, "max": 1},
        }
        config["ewcdr"]["importance_max_samples"] = int(config["smoke"].get("importance_samples", 16))
        config["evaluation"]["eval_limit"] = int(config["smoke"].get("eval_samples", 64))
        config["cache"]["root"] = str(config["cache"]["root"]) + "_smoke"
    return config


def _hardware() -> dict[str, Any]:
    result: dict[str, Any] = {"torch": torch.__version__, "cuda": torch.version.cuda}
    if torch.cuda.is_available():
        result["gpu_name"] = torch.cuda.get_device_name()
        result["gpu_count"] = torch.cuda.device_count()
        try:
            result["nvidia_smi"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader"],
                text=True,
            ).strip().splitlines()
        except Exception:
            pass
    return result


def _loader(dataset, config: Mapping[str, Any], *, shuffle: bool, importance: bool = False):
    section = "importance" if importance else "training"
    options = dataloader_options(config, section)
    batch_size = int(config[section].get("batch_size", config["training"]["batch_size"]))
    return make_loader(dataset, batch_size=batch_size, shuffle=shuffle, **options)


def _prepare_caches(
    config: Mapping[str, Any],
    selected_stages: tuple[str, ...],
    device: torch.device,
    logger,
) -> tuple[dict[str, Any], dict[str, Any]]:
    records = load_stage_tools(config)
    max_train_samples = config["training"].get("max_train_samples")
    stage_samples: dict[str, Any] = {}
    parse_audit: dict[str, Any] = {"train": {}, "global_eval": {}}
    for stage in selected_stages:
        samples, details = build_stage_samples(
            config, stage, "train", records=records, max_samples=max_train_samples
        )
        stage_samples[stage] = samples
        parse_audit["train"][stage] = details
    prepare_global_eval = bool(config.get("cache", {}).get("prepare_global_eval", True))
    global_samples: list[Any] = []
    if prepare_global_eval:
        eval_limit = config.get("evaluation", {}).get("eval_limit")
        global_samples, global_details = build_global_eval_samples(config, records=records, max_samples=eval_limit)
        parse_audit["global_eval"] = global_details
    else:
        parse_audit["global_eval"] = {"skipped_during_training": True}

    cache_root = resolve_path(config["cache"]["root"], project_root(config))
    ensure_dir(cache_root)
    cache_specs = [(f"{stage}_train", stage_samples[stage]) for stage in selected_stages]
    if prepare_global_eval:
        cache_specs.append(("global_eval", global_samples))
    encoder = None
    cache_metadata: dict[str, Any] = {}
    options = dataloader_options(config, "cache")
    hidden_size = int(config["model"].get("hidden_size", 4096))
    max_length = int(config["model"].get("max_length", 512))
    missing_specs: list[tuple[str, Any]] = []
    for name, samples in cache_specs:
        cache_dir = cache_root / name
        if cache_is_valid(cache_dir, samples, hidden_size, max_length):
            manifest = load_json(cache_dir / "manifest.json")
            cache_metadata[name] = {"path": str(cache_dir), **manifest}
            logger.info("reusing frozen-encoder feature cache without loading LLaMA: %s", cache_dir)
        else:
            missing_specs.append((name, samples))
    try:
        if missing_specs:
            encoder = load_frozen_encoder(config, device)
            logger.info("loaded complete frozen LLaMA encoder for %s missing feature cache(s)", len(missing_specs))
        for name, samples in missing_specs:
            if encoder is None:
                raise AssertionError("Missing feature caches require a loaded encoder")
            cache_dir = cache_root / name
            manifest = build_feature_cache(
                encoder,
                samples,
                cache_dir,
                batch_size=int(config["cache"].get("encoder_batch_size", 8)),
                shard_size=int(config["cache"].get("shard_size", 4096)),
                dataloader_options=options,
                logger=logger,
            )
            cache_metadata[name] = {"path": str(cache_dir), **manifest}
    finally:
        if encoder is not None:
            del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    save_json(cache_root / "parse_audit.json", parse_audit)
    return stage_samples, cache_metadata


def _stage_epoch_limits(config: Mapping[str, Any], stage: str) -> tuple[int, int]:
    values = config["training"]["stage_epochs"][stage]
    minimum, maximum = int(values["min"]), int(values["max"])
    if minimum <= 0 or maximum < minimum:
        raise ValueError(f"Invalid epoch limits for {stage}: min={minimum}, max={maximum}")
    return minimum, maximum


def _moving_average(history: list[dict[str, Any]], key: str, window: int) -> float:
    rows = history[-max(1, int(window)) :]
    return sum(float(row[key]) for row in rows) / len(rows)


def _relative_loss_change(previous: float, current: float) -> float:
    return abs(current - previous) / max(abs(previous), 1e-12)


def _cache_regularizer(values: Mapping[str, torch.Tensor] | None, device: torch.device):
    if not values:
        return values
    return {name: tensor.to(device=device, dtype=torch.float32) for name, tensor in values.items()}


def classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    first_class: int = 0,
    global_ce_weight: float = 0.0,
) -> tuple[torch.Tensor, int]:
    """Compute current-stage CE with optional all-visible calibration."""
    start = int(first_class)
    if start < 0 or start >= int(logits.shape[1]):
        raise ValueError(f"Invalid first_class={start} for {logits.shape[1]} logits")
    if bool(torch.any(targets < start)) or bool(torch.any(targets >= logits.shape[1])):
        raise ValueError(f"Targets must lie in [{start}, {logits.shape[1]})")
    weight = float(global_ce_weight)
    if weight < 0.0 or weight > 1.0:
        raise ValueError(f"global_ce_weight must lie in [0, 1], got {weight}")
    active_logits = logits[:, start:]
    current_loss = F.cross_entropy(active_logits.float(), targets - start).float()
    if start == 0 or weight == 0.0:
        return current_loss, int(active_logits.shape[1])
    global_loss = F.cross_entropy(logits.float(), targets).float()
    return ((1.0 - weight) * current_loss + weight * global_loss).float(), int(logits.shape[1])


def _materialize_feature_subset(dataset: FeatureDataset, indices: Sequence[int]) -> FeatureDataset:
    index_tensor = torch.tensor(list(indices), dtype=torch.long)
    return FeatureDataset(
        dataset.hidden.index_select(0, index_tensor),
        dataset.tool_ids.index_select(0, index_tensor),
        [dataset.source_stages[index] for index in indices],
    )


@torch.no_grad()
def _evaluate_seen_validation(
    model,
    validation_datasets: Mapping[str, FeatureDataset],
    seen_stages: Sequence[str],
    config: Mapping[str, Any],
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
) -> dict[str, Any]:
    model.eval()
    per_stage: dict[str, Any] = {}
    validation_config = config.get("training", {}).get("validation", {})
    options = dataloader_options({"validation": validation_config}, "validation")
    batch_size = int(
        validation_config.get("batch_size", config.get("evaluation", {}).get("batch_size", 512))
    )
    for eval_stage in seen_stages:
        accumulator = RetrievalMetrics((1,))
        loader = make_loader(
            validation_datasets[eval_stage],
            batch_size=batch_size,
            shuffle=False,
            **options,
        )
        for batch in loader:
            with autocast_context(device, amp_enabled, amp_dtype):
                logits = model.forward_batch(batch, device)
            targets = batch["tool_id"].to(device, non_blocking=True)
            accumulator.update(logits, targets, model.num_tools)
        per_stage[eval_stage] = accumulator.result()

    current_stage = seen_stages[-1]
    current_recall = float(per_stage[current_stage]["Recall@1"])
    historical_values = [float(per_stage[stage]["Recall@1"]) for stage in seen_stages[:-1]]
    historical_mean = sum(historical_values) / len(historical_values) if historical_values else current_recall
    if historical_values:
        denominator = historical_mean + current_recall
        selection_score = 2.0 * historical_mean * current_recall / denominator if denominator > 0 else 0.0
    else:
        selection_score = current_recall
    return {
        "selection_metric": "harmonic_mean_historical_current_recall_at_1",
        "selection_score": selection_score,
        "historical_mean_recall_at_1": historical_mean,
        "current_recall_at_1": current_recall,
        "per_stage": per_stage,
    }


def train(config: dict[str, Any]) -> Path:
    root = project_root(config)
    output_dir = resolve_path(config["project"]["output_dir"], root)
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    importance_dir = ensure_dir(output_dir / "importance")
    logger = setup_logging(output_dir, "train")
    save_json(output_dir / "config.json", config)
    set_seed(int(config["training"].get("seed", 42)))
    device = resolve_device(config["runtime"].get("device", "cuda"), config["runtime"].get("gpu"))
    amp_enabled, amp_dtype, precision_name = resolve_precision(config, device, logger)
    selected_stages = tuple(config["training"].get("stages", STAGES))
    if any(stage not in STAGES for stage in selected_stages):
        raise ValueError(f"Invalid stages: {selected_stages}")
    logger.info("pure EWC-DR training device=%s stages=%s output=%s", device, selected_stages, output_dir)
    logger.info("hardware=%s", _hardware())

    stage_samples, cache_metadata = _prepare_caches(config, selected_stages, device, logger)
    cache_root = resolve_path(config["cache"]["root"], root)
    method = str(config["ewcdr"].get("method", "ewc_dr"))
    classification_scope = str(config["training"].get("classification_scope", "all_visible"))
    if classification_scope not in {"all_visible", "current_stage", "current_stage_calibrated"}:
        raise ValueError(f"Unsupported training.classification_scope={classification_scope}")
    lambda_ewc = float(config["ewcdr"].get("lambda", 1000.0))
    gamma = float(config["ewcdr"].get("gamma", 1.0))
    resume = bool(config["runtime"].get("resume", False))
    save_final_importance = bool(config["ewcdr"].get("save_final_importance", False))
    validation_config = config["training"].get("validation", {})
    validation_enabled = bool(validation_config.get("enabled", False))
    checkpoint_selection = str(
        config["training"].get(
            "checkpoint_selection", "validation" if validation_enabled else "train_loss"
        )
    )
    if checkpoint_selection not in {"train_loss", "validation", "last_epoch"}:
        raise ValueError(f"Unsupported training.checkpoint_selection: {checkpoint_selection}")
    if checkpoint_selection == "validation" and not validation_enabled:
        raise ValueError("Validation checkpoint selection requires training.validation.enabled=true")
    validation_datasets: dict[str, FeatureDataset] = {}
    validation_split_reports: dict[str, Any] = {}
    logger.info("classification_scope=%s", classification_scope)
    logger.info(
        "checkpoint_selection=%s validation_enabled=%s", checkpoint_selection, validation_enabled
    )
    previous_checkpoint: Path | None = None
    snapshot: dict[str, torch.Tensor] | None = None
    accumulated: dict[str, torch.Tensor] | None = None
    stage_summaries: list[dict[str, Any]] = []
    all_epochs: list[dict[str, Any]] = []
    existing_summary = output_dir / "training_summary.json"
    if resume and existing_summary.exists():
        existing = __import__("json").load(open(existing_summary, encoding="utf-8"))
        stage_summaries = list(existing.get("stages", []))
        all_epochs = list(existing.get("epochs", []))

    for stage_position, stage in enumerate(selected_stages):
        stage_started = time.time()
        checkpoint_path = checkpoint_dir / f"{stage}.pt"
        importance_path = importance_dir / f"importance_{stage}.pt"
        needs_importance = method in {"ewc", "ewc_dr"} and (save_final_importance or stage_position < len(selected_stages) - 1)
        if resume and checkpoint_path.exists() and (not needs_importance or importance_path.exists()):
            logger.info("resume: reusing completed stage=%s", stage)
            previous_checkpoint = checkpoint_path
            if importance_path.exists():
                payload = load_importance(importance_path)
                snapshot = payload["snapshot"]
                accumulated = payload["importance_total"]
            continue

        full_dataset, manifest = load_feature_dataset(cache_root / f"{stage}_train")
        expected_samples = len(stage_samples[stage])
        if len(full_dataset) != expected_samples:
            raise AssertionError(f"{stage} cache has {len(full_dataset)} rows, expected {expected_samples}")
        if validation_enabled:
            train_indices, validation_indices, split_report = stratified_train_validation_indices(
                stage_samples[stage],
                validation_fraction=float(validation_config.get("fraction", 0.1)),
                seed=int(validation_config.get("seed", config["training"].get("seed", 42))),
            )
            dataset = Subset(full_dataset, train_indices)
            validation_datasets[stage] = _materialize_feature_subset(full_dataset, validation_indices)
            validation_split_reports[stage] = split_report
            training_samples = [stage_samples[stage][index] for index in train_indices]
            logger.info("stage=%s validation_split=%s", stage, split_report)
        else:
            train_indices = list(range(expected_samples))
            dataset = full_dataset
            training_samples = stage_samples[stage]
        model = build_retriever(config, stage, encoder=None).to(device)
        inheritance = None
        if previous_checkpoint is not None:
            inheritance = initialize_from_previous(model, previous_checkpoint, stage, verify_exact=True)
            logger.info("verified exact inherited classifier prefix for stage=%s old_tools=%s", stage, inheritance["num_tools"])
        names = list(named_trainable_parameters(model))
        expected_prefixes = ("query_projection.", "classifier.")
        unexpected = [name for name in names if not name.startswith(expected_prefixes)]
        if unexpected:
            raise AssertionError(f"Unexpected trainable parameters: {unexpected}")
        logger.info("stage=%s trainable=%s", stage, trainable_summary(model))

        classification_first = (
            int(EXPECTED_OLD_TOOL_COUNTS.get(stage, 0))
            if classification_scope in {"current_stage", "current_stage_calibrated"}
            else 0
        )
        global_ce_weight = 0.0
        if classification_scope == "current_stage_calibrated" and classification_first > 0:
            configured_weight = config["training"].get("global_ce_weight", "class_ratio")
            if str(configured_weight).lower() == "class_ratio":
                global_ce_weight = (model.num_tools - classification_first) / model.num_tools
            else:
                global_ce_weight = float(configured_weight)
            if not 0.0 < global_ce_weight < 1.0:
                raise ValueError(f"global_ce_weight must lie in (0, 1), got {global_ce_weight}")
        logger.info(
            "stage=%s CE classifier rows=[%s,%s) global_ce_weight=%.6f",
            stage, classification_first, model.num_tools, global_ce_weight,
        )

        loader = _loader(dataset, config, shuffle=True)
        optimizer = AdamW(
            named_trainable_parameters(model).values(),
            lr=float(config["training"]["stage_lrs"].get(stage, config["training"]["lr"])),
            weight_decay=float(config["training"].get("weight_decay", 0.01)),
        )
        scaler = torch.cuda.amp.GradScaler(enabled=bool(amp_enabled and amp_dtype == torch.float16))
        regularizer_snapshot = _cache_regularizer(snapshot, device)
        regularizer_importance = _cache_regularizer(accumulated, device)
        minimum_epochs, maximum_epochs = _stage_epoch_limits(config, stage)
        early = config["training"]["early_stopping"]
        patience = int(early.get("patience", 3))
        window = int(early.get("window", 3))
        absolute_delta = float(early.get("absolute_min_delta", 0.001))
        relative_delta = float(early.get("relative_min_delta", 0.001))
        convergence_criterion = str(early.get("criterion", "best_smoothed_loss"))
        relative_change_threshold = float(early.get("relative_change_threshold", 0.05))
        if convergence_criterion not in {"best_smoothed_loss", "consecutive_relative_change"}:
            raise ValueError(f"Unsupported early-stopping criterion: {convergence_criterion}")
        if not 0.0 < relative_change_threshold < 1.0:
            raise ValueError("training.early_stopping.relative_change_threshold must be in (0, 1)")
        stale = 0
        best_smoothed_total = math.inf
        best_smoothed_ce = math.inf
        best_raw_total = math.inf
        best_selection_score = -math.inf
        best_patience_score = -math.inf
        best_epoch = -1
        stop_reason = "max_epochs_reached"
        best_path = checkpoint_dir / f".{stage}.best.pt"
        history: list[dict[str, Any]] = []
        train_started = time.time()

        for epoch in range(maximum_epochs):
            epoch_started = time.time()
            model.train()
            totals = {"ce": 0.0, "ewc": 0.0, "total": 0.0}
            batches = 0
            samples_seen = 0
            progress = tqdm(loader, desc=f"pure {stage} epoch {epoch + 1}", dynamic_ncols=True, mininterval=2.0)
            for batch in progress:
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(device, amp_enabled, amp_dtype):
                    logits = model.forward_batch(batch, device)
                targets = batch["tool_id"].to(device, non_blocking=True)
                ce_loss, classification_candidates = classification_loss(
                    logits,
                    targets,
                    first_class=classification_first,
                    global_ce_weight=global_ce_weight,
                )
                if stage_position == 0 or method == "seq_ft":
                    ewc_loss = torch.zeros((), device=device, dtype=torch.float32)
                else:
                    ewc_loss = regularization_loss(
                        model, regularizer_snapshot, regularizer_importance, lambda_ewc
                    )
                total_loss = ce_loss + ewc_loss
                if not torch.isfinite(total_loss):
                    raise RuntimeError(f"Non-finite loss at {stage} epoch {epoch + 1}")
                if scaler.is_enabled():
                    scaler.scale(total_loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"].get("grad_clip", 1.0)))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"].get("grad_clip", 1.0)))
                    optimizer.step()
                batches += 1
                batch_samples = int(targets.numel())
                samples_seen += batch_samples
                totals["ce"] += float(ce_loss.detach().cpu())
                totals["ewc"] += float(ewc_loss.detach().cpu())
                totals["total"] += float(total_loss.detach().cpu())
                if batches == 1 or batches % int(config["training"].get("log_every", 50)) == 0:
                    progress.set_postfix(ce=f"{totals['ce']/batches:.4f}", ewc=f"{totals['ewc']/batches:.4f}")
            denominator = max(1, batches)
            row: dict[str, Any] = {
                "stage": stage,
                "epoch": epoch + 1,
                "ce_loss": totals["ce"] / denominator,
                "ewc_loss": totals["ewc"] / denominator,
                "total_loss": totals["total"] / denominator,
                "ewc_to_ce_ratio": (totals["ewc"] / max(totals["ce"], 1e-12)),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "classification_scope": classification_scope,
                "classification_candidates": classification_candidates,
                "classification_first_class": classification_first,
                "global_ce_weight": global_ce_weight,
                "batches": batches,
                "samples": samples_seen,
                "duration_sec": round(time.time() - epoch_started, 3),
                "gpu_memory": gpu_memory_summary(),
                **drift_summary(model, snapshot),
            }
            if history:
                row["relative_ce_loss_change"] = _relative_loss_change(
                    float(history[-1]["ce_loss"]), float(row["ce_loss"])
                )
                row["relative_total_loss_change"] = _relative_loss_change(
                    float(history[-1]["total_loss"]), float(row["total_loss"])
                )
                row["convergence_relative_change"] = max(
                    float(row["relative_ce_loss_change"]),
                    float(row["relative_total_loss_change"]),
                )
            else:
                row["relative_ce_loss_change"] = None
                row["relative_total_loss_change"] = None
                row["convergence_relative_change"] = None
            if validation_enabled and (epoch + 1) % int(validation_config.get("eval_every", 1)) == 0:
                validation_started = time.time()
                row["validation"] = _evaluate_seen_validation(
                    model,
                    validation_datasets,
                    selected_stages[: stage_position + 1],
                    config,
                    device,
                    amp_enabled,
                    amp_dtype,
                )
                row["validation_duration_sec"] = round(time.time() - validation_started, 3)
            history.append(row)
            all_epochs.append(row)
            logger.info("epoch=%s", row)
            best_raw_total = min(best_raw_total, float(row["total_loss"]))
            if checkpoint_selection == "validation":
                if "validation" not in row:
                    is_best = False
                else:
                    selection_score = float(row["validation"]["selection_score"])
                    is_best = selection_score > best_selection_score
                    if is_best:
                        best_selection_score = selection_score
            elif checkpoint_selection == "last_epoch":
                is_best = True
            else:
                is_best = float(row["total_loss"]) <= best_raw_total
            if is_best:
                best_epoch = epoch + 1
                save_checkpoint(
                    best_path,
                    model,
                    stage=stage,
                    epoch=best_epoch,
                    training_history=history,
                    metadata={
                        "precision": precision_name,
                        "feature_cache": str(cache_root / f"{stage}_train"),
                        "classification_scope": classification_scope,
                        "classification_candidates": classification_candidates,
                        "checkpoint_selection": checkpoint_selection,
                        "selection_score": (
                            row.get("validation", {}).get("selection_score")
                            if checkpoint_selection == "validation"
                            else None
                        ),
                    },
                )
            if validation_enabled and "validation" in row:
                validation_delta = float(validation_config.get("min_delta", 0.05))
                validation_score = float(row["validation"]["selection_score"])
                if validation_score > best_patience_score + validation_delta:
                    best_patience_score = validation_score
                    stale = 0
                elif epoch + 1 >= minimum_epochs:
                    stale += 1
                validation_patience = int(validation_config.get("patience", patience))
                if epoch + 1 >= minimum_epochs and stale >= validation_patience:
                    stop_reason = (
                        "validation_converged(metric=harmonic_mean_historical_current_recall_at_1,"
                        f"patience={validation_patience},min_delta={validation_delta})"
                    )
                    break
            elif not validation_enabled and convergence_criterion == "consecutive_relative_change":
                change = row["convergence_relative_change"]
                if change is not None and float(change) <= relative_change_threshold:
                    stale += 1
                else:
                    stale = 0
                if epoch + 1 >= minimum_epochs and stale >= patience:
                    stop_reason = (
                        "train_loss_relative_change_converged("
                        f"patience={patience},threshold={relative_change_threshold})"
                    )
                    break
            elif not validation_enabled and len(history) >= window:
                smooth_total = _moving_average(history, "total_loss", window)
                smooth_ce = _moving_average(history, "ce_loss", window)
                total_threshold = max(absolute_delta, abs(best_smoothed_total) * relative_delta) if math.isfinite(best_smoothed_total) else 0.0
                ce_threshold = max(absolute_delta, abs(best_smoothed_ce) * relative_delta) if math.isfinite(best_smoothed_ce) else 0.0
                improved = (
                    not math.isfinite(best_smoothed_total)
                    or smooth_total < best_smoothed_total - total_threshold
                    or smooth_ce < best_smoothed_ce - ce_threshold
                )
                if improved:
                    best_smoothed_total = min(best_smoothed_total, smooth_total)
                    best_smoothed_ce = min(best_smoothed_ce, smooth_ce)
                    stale = 0
                elif epoch + 1 >= minimum_epochs:
                    stale += 1
                if epoch + 1 >= minimum_epochs and stale >= patience:
                    stop_reason = f"train_loss_converged(window={window},patience={patience},abs_delta={absolute_delta},rel_delta={relative_delta})"
                    break

        train_duration = time.time() - train_started
        if not best_path.exists():
            raise RuntimeError(f"No best checkpoint was produced for {stage}")
        load_checkpoint(model, best_path)
        save_checkpoint(
            checkpoint_path,
            model,
            stage=stage,
            epoch=best_epoch,
            training_history=history,
            metadata={
                "precision": precision_name,
                "stop_reason": stop_reason,
                "completed_epochs": len(history),
                "best_total_loss": best_raw_total,
                "feature_cache": str(cache_root / f"{stage}_train"),
                "classification_scope": classification_scope,
                "classification_candidates": history[-1]["classification_candidates"],
                "checkpoint_selection": checkpoint_selection,
                "selection_score": (
                    history[best_epoch - 1].get("validation", {}).get("selection_score")
                    if checkpoint_selection == "validation"
                    else None
                ),
            },
        )
        best_path.unlink(missing_ok=True)
        logger.info("saved stage checkpoint: %s", checkpoint_path)
        del optimizer, scaler, regularizer_snapshot, regularizer_importance
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        importance_duration = 0.0
        importance_report: dict[str, Any] | None = None
        sampling_report: dict[str, Any] | None = None
        if needs_importance:
            _, selected_indices, sampling_report = sample_by_tool(
                training_samples,
                max_samples=config["ewcdr"].get("importance_max_samples"),
                seed=int(config["training"].get("seed", 42)),
            )
            full_selected_indices = [train_indices[index] for index in selected_indices]
            importance_dataset = Subset(full_dataset, full_selected_indices)
            importance_loader = _loader(importance_dataset, config, shuffle=False, importance=True)
            importance_started = time.time()
            current_importance, importance_report = compute_importance(
                model,
                importance_loader,
                device,
                method=method,
                max_batches=config["ewcdr"].get("importance_max_batches"),
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                accumulation_device=config["ewcdr"].get("accumulation_device", "cuda"),
                omega_max=config["ewcdr"].get("omega_max"),
                verify_unchanged=bool(config["ewcdr"].get("verify_unchanged", False)),
            )
            importance_duration = time.time() - importance_started
            accumulated = accumulate_online(accumulated, current_importance, gamma)
            snapshot = parameter_snapshot(model)
            save_importance(
                importance_path,
                stage=stage,
                method=method,
                gamma=gamma,
                current=current_importance,
                total=accumulated,
                snapshot=snapshot,
                metadata={
                    "sampling": sampling_report,
                    "importance": importance_report,
                    "duration_sec": round(importance_duration, 3),
                    "omega_max": config["ewcdr"].get("omega_max"),
                },
            )
            logger.info("saved importance: %s", importance_path)
        else:
            logger.info("skipping final importance for stage=%s because no later stage uses it", stage)

        summary = {
            "stage": stage,
            "train_samples": len(dataset),
            "full_train_samples": len(full_dataset),
            "validation_samples": len(validation_datasets[stage]) if validation_enabled else 0,
            "validation_split": validation_split_reports.get(stage),
            "completed_epochs": len(history),
            "best_epoch": best_epoch,
            "checkpoint_selection": checkpoint_selection,
            "best_selection_score": best_selection_score if checkpoint_selection == "validation" else None,
            "stop_reason": stop_reason,
            "train_duration_sec": round(train_duration, 3),
            "importance_duration_sec": round(importance_duration, 3),
            "total_stage_duration_sec": round(time.time() - stage_started, 3),
            "checkpoint": str(checkpoint_path),
            "sampling": sampling_report,
            "importance_summary": importance_report,
            "final_epoch": history[-1],
            "trainable": trainable_summary(model),
            "classification_scope": classification_scope,
            "classification_candidates": history[-1]["classification_candidates"],
        }
        stage_summaries = [row for row in stage_summaries if row.get("stage") != stage]
        stage_summaries.append(summary)
        save_json(
            output_dir / "training_summary.json",
            {
                "method": method,
                "classification_scope": classification_scope,
                "hardware": _hardware(),
                "precision": precision_name,
                "feature_cache": cache_metadata,
                "checkpoint_selection": checkpoint_selection,
                "validation": {
                    "enabled": validation_enabled,
                    "config": dict(validation_config),
                    "splits": validation_split_reports,
                },
                "stages": stage_summaries,
                "epochs": all_epochs,
            },
        )
        previous_checkpoint = checkpoint_path
        del dataset, full_dataset, loader, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("pure EWC-DR training complete: %s", output_dir)
    return output_dir


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    train(config)


if __name__ == "__main__":
    main()
