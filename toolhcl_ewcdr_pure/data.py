from __future__ import annotations

import random
import re
from functools import lru_cache
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from .utils import load_json, protocol_stages, protocol_tool_counts, save_json, stage_index


TARGET_RE = re.compile(r"^<<(.+?)&&(.+?)>>")
API_RE = re.compile(r"API:\s*(.*?)\.(?:\s*API Description:|$)")
TOOL_RE = re.compile(r"Tool:\s*(.*?)\.(?:\s*Description:|\s*API:|$)")
API_NAME_RE = re.compile(r"Api Name:\s*(.*?)\s+Api Description:", re.IGNORECASE | re.DOTALL)
TOOL_ONLY_RE = re.compile(r"Tool:\s*(.*?)\.\s*API:", re.IGNORECASE | re.DOTALL)


def normalize_name(value: Any) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "", str(value)).lower()


def tool_key(tool_name: str, api_name: str) -> str:
    return normalize_name(f"{api_name}for{tool_name}")


def query_text(item: Mapping[str, Any]) -> str:
    """Return the first user query, matching the Transaction split builder."""
    for message in item.get("conversations", []):
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("content", "")).strip()
    return ""


def normalized_query(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


@lru_cache(maxsize=8)
def _collect_eval_decontamination_keys(
    eval_paths: tuple[str, ...],
) -> tuple[frozenset[str], frozenset[str]]:
    source_ids: set[str] = set()
    queries: set[str] = set()
    for path in eval_paths:
        raw = load_json(path)
        if not isinstance(raw, list):
            raise ValueError(f"Retrieval split must be a list: {path}")
        for item in raw:
            if not isinstance(item, dict):
                continue
            source = item.get("source_id")
            if source is not None:
                source_ids.add(str(source))
            query = normalized_query(query_text(item))
            if query:
                queries.add(query)
    return frozenset(source_ids), frozenset(queries)


def eval_decontamination_keys(config: Mapping[str, Any]) -> tuple[frozenset[str], frozenset[str]]:
    stages = protocol_stages(config)
    paths = tuple(str(Path(config["data"][f"{stage}_eval"]).resolve()) for stage in stages)
    return _collect_eval_decontamination_keys(paths)


@dataclass(frozen=True)
class ToolRecord:
    tool_id: int
    tool_name: str
    api_name: str
    source_stage: str


@dataclass(frozen=True)
class PureSample:
    query_text: str
    tool_id: int
    tool_name: str
    api_name: str
    source_stage: str
    source_id: str | None = None


def _tool_names(record: Any, tool_id: int) -> tuple[str, str]:
    if isinstance(record, dict) and record.get("tool_name") and record.get("api_name"):
        return str(record["tool_name"]).strip(), str(record["api_name"]).strip()
    text = str(record.get("text", "")) if isinstance(record, dict) else str(record)
    tool_match = TOOL_RE.search(text) or TOOL_ONLY_RE.search(text)
    api_match = API_NAME_RE.search(text) or API_RE.search(text)
    tool_name = tool_match.group(1).strip() if tool_match else f"__unmapped_tool_{tool_id}"
    api_name = api_match.group(1).strip() if api_match else f"__unmapped_api_{tool_id}"
    return tool_name, api_name


def load_stage_tools(config: Mapping[str, Any]) -> dict[str, dict[int, ToolRecord]]:
    result: dict[str, dict[int, ToolRecord]] = {}
    accumulated: set[int] = set()
    stages = protocol_stages(config)
    tool_counts = protocol_tool_counts(config)
    for stage in stages:
        raw = load_json(config["data"][f"{stage}_tools"])
        records: dict[int, ToolRecord] = {}
        items = ((int(item["tool_id"]), item) for item in raw) if isinstance(raw, list) else ((int(k), v) for k, v in raw.items())
        for tool_id, item in items:
            names = _tool_names(item, tool_id)
            records[tool_id] = ToolRecord(tool_id, names[0], names[1], stage)
        result[stage] = dict(sorted(records.items()))
        accumulated.update(records)
        expected = tool_counts[stage]
        if len(accumulated) != expected or accumulated != set(range(expected)):
            raise AssertionError(
                f"{stage} tool IDs must be contiguous [0, {expected}); got {len(accumulated)} unique IDs"
            )
    return result


def visible_tools(
    stage: str,
    records: Mapping[str, Mapping[int, ToolRecord]],
    config: Mapping[str, Any] | None = None,
) -> dict[int, ToolRecord]:
    stages = protocol_stages(config) if config is not None else tuple(records)
    if stage not in stages:
        raise ValueError(f"Unknown stage: {stage}")
    visible: dict[int, ToolRecord] = {}
    for current in stages[: stage_index(stage, stages) + 1]:
        visible.update(records[current])
    expected = protocol_tool_counts(config)[stage] if config is not None else len(visible)
    if set(visible) != set(range(expected)):
        raise AssertionError(f"{stage} visible IDs do not match classifier rows 0..{expected - 1}")
    return dict(sorted(visible.items()))


def name_mapping(tools: Mapping[int, ToolRecord]) -> tuple[dict[str, int], dict[str, list[int]]]:
    mapping: dict[str, int] = {}
    collisions: dict[str, list[int]] = {}
    for tool_id, record in tools.items():
        key = tool_key(record.tool_name, record.api_name)
        if key in mapping and mapping[key] != tool_id:
            collisions.setdefault(key, [mapping[key]]).append(tool_id)
        mapping[key] = tool_id
    return mapping, collisions


def parse_split(
    path: str | Path,
    *,
    mapping: Mapping[str, int],
    tools: Mapping[int, ToolRecord],
    source_stage: str,
    visible_count: int,
    max_samples: int | None = None,
    target_id_field: str | None = None,
    excluded_queries: set[str] | None = None,
    excluded_source_ids: frozenset[str] | set[str] | None = None,
    excluded_normalized_queries: frozenset[str] | set[str] | None = None,
) -> tuple[list[PureSample], dict[str, int]]:
    raw = load_json(path)
    if not isinstance(raw, list):
        raise ValueError(f"Retrieval split must be a list: {path}")
    samples: list[PureSample] = []
    stats = {
        "raw": len(raw),
        "missing_conversation": 0,
        "malformed_target": 0,
        "unmapped_target": 0,
        "invalid_direct_target": 0,
        "direct_target": 0,
        "target_name_unmapped": 0,
        "target_name_id_mismatch": 0,
        "excluded_train_query_overlap": 0,
        "excluded_eval_source_overlap": 0,
        "excluded_eval_query_overlap": 0,
        "excluded_eval_source_or_query_overlap": 0,
        "out_of_range": 0,
    }
    for item in raw:
        query = query_text(item) if isinstance(item, dict) else ""
        target = ""
        conversations = item.get("conversations", []) if isinstance(item, dict) else []
        for message in conversations:
            if not isinstance(message, dict):
                continue
            if message.get("role") == "assistant":
                target = str(message.get("content", "")).strip()
        if not query or (not target and not target_id_field):
            stats["missing_conversation"] += 1
            continue
        if excluded_queries is not None and query in excluded_queries:
            stats["excluded_train_query_overlap"] += 1
            continue
        source = item.get("source_id") if isinstance(item, dict) else None
        source_overlap = (
            excluded_source_ids is not None
            and source is not None
            and str(source) in excluded_source_ids
        )
        query_overlap = (
            excluded_normalized_queries is not None
            and normalized_query(query) in excluded_normalized_queries
        )
        stats["excluded_eval_source_overlap"] += int(source_overlap)
        stats["excluded_eval_query_overlap"] += int(query_overlap)
        if source_overlap or query_overlap:
            stats["excluded_eval_source_or_query_overlap"] += 1
            continue
        mapped_tool_id: int | None = None
        match = TARGET_RE.match(target)
        if match:
            tool_name, api_name = match.group(1).strip(), match.group(2).strip()
            key = tool_key(tool_name, api_name)
            if key in mapping:
                mapped_tool_id = int(mapping[key])
            elif target_id_field:
                stats["target_name_unmapped"] += 1
            else:
                stats["unmapped_target"] += 1
                continue
        elif target_id_field:
            stats["malformed_target"] += 1
        else:
            stats["malformed_target"] += 1
            continue
        if target_id_field:
            try:
                tool_id = int(item[target_id_field])
            except (KeyError, TypeError, ValueError):
                stats["invalid_direct_target"] += 1
                continue
            stats["direct_target"] += 1
            if mapped_tool_id is not None and mapped_tool_id != tool_id:
                stats["target_name_id_mismatch"] += 1
        else:
            if mapped_tool_id is None:
                stats["unmapped_target"] += 1
                continue
            tool_id = mapped_tool_id
        if tool_id < 0 or tool_id >= visible_count:
            stats["out_of_range"] += 1
            continue
        if tool_id not in tools:
            stats["out_of_range"] += 1
            continue
        canonical = tools[tool_id]
        samples.append(
            PureSample(
                query,
                tool_id,
                canonical.tool_name,
                canonical.api_name,
                source_stage,
                None if source is None else str(source),
            )
        )
        if max_samples is not None and len(samples) >= int(max_samples):
            break
    stats["parsed"] = len(samples)
    return samples, stats


def build_stage_samples(
    config: Mapping[str, Any],
    stage: str,
    split: str,
    *,
    records: Mapping[str, Mapping[int, ToolRecord]] | None = None,
    max_samples: int | None = None,
    apply_decontamination: bool | None = None,
) -> tuple[list[PureSample], dict[str, Any]]:
    records = records or load_stage_tools(config)
    tools = visible_tools(stage, records, config)
    tool_counts = protocol_tool_counts(config)
    mapping, collisions = name_mapping(tools)
    decontamination = config.get("data", {}).get("decontamination", {})
    enabled = bool(decontamination.get("enabled", False))
    if apply_decontamination is not None:
        enabled = bool(apply_decontamination)
    excluded_sources = excluded_queries = None
    if split == "train" and enabled:
        excluded_sources, excluded_queries = eval_decontamination_keys(config)
    samples, stats = parse_split(
        config["data"][f"{stage}_{split}"],
        mapping=mapping,
        tools=tools,
        source_stage=stage,
        visible_count=tool_counts[stage],
        max_samples=max_samples,
        target_id_field=config.get("data", {}).get("target_id_field"),
        excluded_source_ids=excluded_sources,
        excluded_normalized_queries=excluded_queries,
    )
    return samples, {
        "parse": stats,
        "normalized_name_collisions": len(collisions),
        "decontamination": {
            "enabled": bool(split == "train" and enabled),
            "policy": "remove_train_if_source_id_or_whitespace_casefold_query_occurs_in_any_eval_split",
            "eval_unique_source_ids": len(excluded_sources or ()),
            "eval_unique_normalized_queries": len(excluded_queries or ()),
            "preserve_original_eval": True,
        },
    }


def build_global_eval_samples(
    config: Mapping[str, Any],
    *,
    records: Mapping[str, Mapping[int, ToolRecord]] | None = None,
    max_samples: int | None = None,
) -> tuple[list[PureSample], dict[str, Any]]:
    records = records or load_stage_tools(config)
    stages = protocol_stages(config)
    tool_counts = protocol_tool_counts(config)
    final_stage = stages[-1]
    tools = visible_tools(final_stage, records, config)
    mapping, collisions = name_mapping(tools)
    all_samples: list[PureSample] = []
    per_stage: dict[str, Any] = {}
    stage_limits: dict[str, int | None] = {stage: None for stage in stages}
    if max_samples is not None:
        base_limit, remainder = divmod(int(max_samples), len(stages))
        stage_limits = {
            stage: base_limit + (1 if index < remainder else 0)
            for index, stage in enumerate(stages)
        }
    filter_overlap = bool(config.get("evaluation", {}).get("filter_train_query_overlap", False))
    for stage in stages:
        stage_limit = stage_limits[stage]
        if stage_limit == 0:
            continue
        excluded_queries = None
        if filter_overlap:
            train_samples, _ = build_stage_samples(config, stage, "train", records=records)
            excluded_queries = {sample.query_text for sample in train_samples}
        samples, stats = parse_split(
            config["data"][f"{stage}_eval"],
            mapping=mapping,
            tools=tools,
            source_stage=stage,
            visible_count=tool_counts[final_stage],
            max_samples=stage_limit,
            target_id_field=config.get("data", {}).get("target_id_field"),
            excluded_queries=excluded_queries,
        )
        all_samples.extend(samples)
        per_stage[stage] = stats
    decontamination = config.get("data", {}).get("decontamination", {})
    return all_samples, {
        "per_stage": per_stage,
        "normalized_name_collisions": len(collisions),
        "parsed": len(all_samples),
        "decontamination": {
            "enabled": bool(decontamination.get("enabled", False)),
            "preserve_original_eval": True,
            "legacy_eval_filter_enabled": filter_overlap,
        },
    }


class TextDataset(Dataset):
    def __init__(self, samples: Sequence[PureSample]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        return {
            "query_text": sample.query_text,
            "tool_id": sample.tool_id,
            "tool_name": sample.tool_name,
            "api_name": sample.api_name,
            "source_stage": sample.source_stage,
        }


class FeatureDataset(Dataset):
    def __init__(self, hidden: torch.Tensor, tool_ids: torch.Tensor, source_stages: Sequence[str]):
        if hidden.shape[0] != tool_ids.shape[0] or hidden.shape[0] != len(source_stages):
            raise ValueError("Feature cache arrays have inconsistent lengths")
        self.hidden = hidden
        self.tool_ids = tool_ids.long()
        self.source_stages = list(source_stages)

    def __len__(self) -> int:
        return int(self.tool_ids.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "query_hidden": self.hidden[index],
            "tool_id": self.tool_ids[index],
            "source_stage": self.source_stages[index],
        }


def collate_batch(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "tool_id": torch.tensor([int(x["tool_id"]) for x in batch], dtype=torch.long),
        "source_stage": [str(x["source_stage"]) for x in batch],
    }
    if "query_hidden" in batch[0]:
        result["query_hidden"] = torch.stack([x["query_hidden"] for x in batch])
    else:
        result["query_text"] = [str(x["query_text"]) for x in batch]
        result["tool_name"] = [str(x["tool_name"]) for x in batch]
        result["api_name"] = [str(x["api_name"]) for x in batch]
    return result


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int | None,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "collate_fn": collate_batch,
    }
    if int(num_workers) > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(dataset, **kwargs)


def sample_by_tool(
    samples: Sequence[PureSample], *, max_samples: int | None, seed: int
) -> tuple[list[PureSample], list[int], dict[str, Any]]:
    full = list(samples)
    unique_tools = len({sample.tool_id for sample in full})
    if max_samples is None or int(max_samples) <= 0 or len(full) <= int(max_samples):
        indices = list(range(len(full)))
        return full, indices, {
            "strategy": "full_train_split",
            "samples": len(full),
            "covered_tools": unique_tools,
            "available_tools": unique_tools,
            "coverage_percent": 100.0 if unique_tools else 0.0,
        }
    rng = random.Random(int(seed))
    groups: dict[int, list[int]] = {}
    for index, sample in enumerate(full):
        groups.setdefault(sample.tool_id, []).append(index)
    for values in groups.values():
        rng.shuffle(values)
    keys = list(groups)
    rng.shuffle(keys)
    limit = int(max_samples)

    # Spend the sampling budget on distinct tools first. Filling in a single
    # pass avoids cursor-reset bias when singleton tool groups are exhausted.
    coverage_count = min(limit, len(keys))
    selected = [groups[key].pop() for key in keys[:coverage_count]]
    if len(selected) < limit:
        remaining = [index for values in groups.values() for index in values]
        rng.shuffle(remaining)
        selected.extend(remaining[: limit - len(selected)])

    selected.sort()
    subset = [full[index] for index in selected]
    covered = len({sample.tool_id for sample in subset})
    expected_coverage = min(unique_tools, limit)
    if covered != expected_coverage:
        raise AssertionError(
            f"Tool-stratified sampling covered {covered} tools; expected {expected_coverage}"
        )
    return subset, selected, {
        "strategy": f"tool_id_coverage_first_then_uniform(seed={seed})",
        "samples": len(subset),
        "covered_tools": covered,
        "available_tools": unique_tools,
        "coverage_percent": 100.0 * covered / max(1, unique_tools),
    }


def stratified_train_validation_indices(
    samples: Sequence[PureSample], *, validation_fraction: float, seed: int
) -> tuple[list[int], list[int], dict[str, Any]]:
    """Create a deterministic tool-ID-stratified holdout without losing train-only tools."""
    fraction = float(validation_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"validation_fraction must be between 0 and 1, got {fraction}")

    groups: dict[int, list[int]] = {}
    for index, sample in enumerate(samples):
        groups.setdefault(int(sample.tool_id), []).append(index)

    rng = random.Random(int(seed))
    train_indices: list[int] = []
    validation_indices: list[int] = []
    singleton_tools = 0
    for tool_id in sorted(groups):
        indices = list(groups[tool_id])
        rng.shuffle(indices)
        if len(indices) == 1:
            singleton_tools += 1
            train_indices.extend(indices)
            continue
        validation_count = max(1, int(round(len(indices) * fraction)))
        validation_count = min(validation_count, len(indices) - 1)
        validation_indices.extend(indices[:validation_count])
        train_indices.extend(indices[validation_count:])

    train_indices.sort()
    validation_indices.sort()
    if set(train_indices) & set(validation_indices):
        raise AssertionError("Training and validation indices overlap")
    if sorted(train_indices + validation_indices) != list(range(len(samples))):
        raise AssertionError("Training/validation split does not cover the full input")

    train_tools = {int(samples[index].tool_id) for index in train_indices}
    validation_tools = {int(samples[index].tool_id) for index in validation_indices}
    available_tools = len(groups)
    if len(train_tools) != available_tools:
        raise AssertionError("At least one tool was removed entirely from the training partition")
    report = {
        "strategy": f"tool_id_stratified_holdout(seed={seed})",
        "validation_fraction_requested": fraction,
        "full_samples": len(samples),
        "train_samples": len(train_indices),
        "validation_samples": len(validation_indices),
        "validation_fraction_actual": len(validation_indices) / max(1, len(samples)),
        "available_tools": available_tools,
        "train_tools": len(train_tools),
        "validation_tools": len(validation_tools),
        "singleton_tools_train_only": singleton_tools,
        "validation_tool_coverage_percent": 100.0 * len(validation_tools) / max(1, available_tools),
    }
    return train_indices, validation_indices, report


def write_data_audit(config: Mapping[str, Any], output_path: Path) -> dict[str, Any]:
    records = load_stage_tools(config)
    audit: dict[str, Any] = {"tool_counts": {}, "splits": {}, "global_eval": {}}
    for stage in protocol_stages(config):
        audit["tool_counts"][stage] = len(visible_tools(stage, records, config))
        for split in ("train", "eval"):
            _, details = build_stage_samples(config, stage, split, records=records)
            audit["splits"][f"{stage}_{split}"] = details
    _, global_details = build_global_eval_samples(config, records=records)
    audit["global_eval"] = global_details
    save_json(output_path, audit)
    return audit


def serialize_samples(path: Path, samples: Sequence[PureSample]) -> None:
    save_json(path, [asdict(sample) for sample in samples])
