from __future__ import annotations

import json
import logging
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml


STAGES = ("base", "task1", "task2", "task3")
EXPECTED_TOOL_COUNTS = {"base": 11112, "task1": 11752, "task2": 12392, "task3": 13035}
EXPECTED_OLD_TOOL_COUNTS = {"task1": 11112, "task2": 11752, "task3": 12392}


def protocol_stages(config: Mapping[str, Any]) -> tuple[str, ...]:
    values = config.get("protocol", {}).get(
        "stages", config.get("training", {}).get("stages", STAGES)
    )
    stages = tuple(str(value) for value in values)
    if not stages or len(set(stages)) != len(stages):
        raise ValueError(f"Protocol stages must be a non-empty ordered unique list: {stages}")
    return stages


def protocol_tool_counts(config: Mapping[str, Any]) -> dict[str, int]:
    stages = protocol_stages(config)
    configured = config.get("protocol", {}).get("tool_counts")
    source = configured if configured is not None else EXPECTED_TOOL_COUNTS
    counts = {stage: int(source[stage]) for stage in stages}
    previous = 0
    for stage in stages:
        if counts[stage] <= previous:
            raise ValueError(
                f"Cumulative tool counts must increase at every stage: {stage}={counts[stage]} after {previous}"
            )
        previous = counts[stage]
    return counts


def protocol_old_tool_counts(config: Mapping[str, Any]) -> dict[str, int]:
    stages = protocol_stages(config)
    counts = protocol_tool_counts(config)
    return {stages[index]: counts[stages[index - 1]] for index in range(1, len(stages))}


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def project_root(config: Mapping[str, Any]) -> Path:
    return Path(config["project"]["root"]).resolve()


def resolve_path(path: str | os.PathLike[str], root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    result = Path(path)
    result.mkdir(parents=True, exist_ok=True)
    return result


def save_json(path: str | os.PathLike[str], value: Any) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, target)


def load_json(path: str | os.PathLike[str]) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def setup_logging(output_dir: Path, name: str) -> logging.Logger:
    log_dir = ensure_dir(output_dir / "logs")
    logger = logging.getLogger(f"pure_ewcdr.{name}.{output_dir}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str, gpu: str | int | None = None) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        if gpu is None:
            return torch.device("cuda")
        return torch.device(f"cuda:{int(gpu)}")
    return torch.device("cpu")


def resolve_precision(config: Mapping[str, Any], device: torch.device, logger: logging.Logger):
    requested = str(config.get("runtime", {}).get("precision", "bf16")).lower()
    use_amp = bool(config.get("runtime", {}).get("use_amp", True)) and device.type == "cuda"
    if use_amp and requested == "bf16" and torch.cuda.is_bf16_supported():
        return True, torch.bfloat16, "bf16"
    if use_amp and requested in {"bf16", "fp16"}:
        logger.warning("Requested %s is unavailable; falling back to fp16", requested)
        return True, torch.float16, "fp16"
    logger.info("AMP disabled; using fp32")
    return False, None, "fp32"


def autocast_context(device: torch.device, enabled: bool, dtype: torch.dtype | None):
    if enabled and device.type == "cuda" and dtype is not None:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def common_prefix_slices(current_shape: torch.Size, previous_shape: torch.Size):
    if len(current_shape) != len(previous_shape):
        return None
    return tuple(slice(0, min(int(a), int(b))) for a, b in zip(current_shape, previous_shape))


def gpu_memory_summary() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {"allocated_gb": 0.0, "reserved_gb": 0.0, "max_allocated_gb": 0.0}
    scale = 1024**3
    return {
        "allocated_gb": round(torch.cuda.memory_allocated() / scale, 3),
        "reserved_gb": round(torch.cuda.memory_reserved() / scale, 3),
        "max_allocated_gb": round(torch.cuda.max_memory_allocated() / scale, 3),
    }


def dataloader_options(config: Mapping[str, Any], section: str) -> dict[str, Any]:
    values = config.get(section, {})
    workers = int(values.get("num_workers", 0))
    return {
        "num_workers": workers,
        "pin_memory": bool(values.get("pin_memory", False)),
        "persistent_workers": bool(values.get("persistent_workers", False)) if workers > 0 else False,
        "prefetch_factor": int(values.get("prefetch_factor", 2)) if workers > 0 else None,
    }


def stage_index(stage: str, stages: tuple[str, ...] = STAGES) -> int:
    if stage not in stages:
        raise ValueError(f"Unknown stage: {stage}")
    return stages.index(stage)
