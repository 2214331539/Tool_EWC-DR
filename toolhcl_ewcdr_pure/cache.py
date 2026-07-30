from __future__ import annotations

import hashlib
import time
from collections import deque
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from tqdm import tqdm

from .data import FeatureDataset, PureSample, TextDataset, make_loader
from .model import FrozenLlamaEncoder
from .utils import ensure_dir, load_json, save_json


def samples_digest(samples: Sequence[PureSample]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(sample.query_text.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(sample.tool_id).encode("ascii"))
        digest.update(b"\0")
        digest.update(sample.source_stage.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def cache_is_valid(cache_dir: Path, samples: Sequence[PureSample], hidden_size: int, max_length: int) -> bool:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    manifest = load_json(manifest_path)
    if (
        int(manifest.get("samples", -1)) != len(samples)
        or int(manifest.get("hidden_size", -1)) != int(hidden_size)
        or int(manifest.get("max_length", -1)) != int(max_length)
        or manifest.get("sample_digest") != samples_digest(samples)
    ):
        return False
    return all((cache_dir / shard["file"]).exists() for shard in manifest.get("shards", []))


def _sample_key(sample: PureSample) -> tuple[str, int, str]:
    return sample.query_text, int(sample.tool_id), sample.source_stage


def build_subset_feature_cache(
    source_cache_dir: Path,
    source_samples: Sequence[PureSample],
    target_samples: Sequence[PureSample],
    cache_dir: Path,
    *,
    hidden_size: int,
    max_length: int,
    shard_size: int,
    logger,
) -> dict[str, Any] | None:
    """Materialize an order-preserving subset without another encoder forward."""
    if cache_is_valid(cache_dir, target_samples, hidden_size, max_length):
        return load_json(cache_dir / "manifest.json")
    if not cache_is_valid(source_cache_dir, source_samples, hidden_size, max_length):
        return None

    positions: dict[tuple[str, int, str], deque[int]] = {}
    for index, sample in enumerate(source_samples):
        positions.setdefault(_sample_key(sample), deque()).append(index)
    selected: list[int] = []
    for sample in target_samples:
        candidates = positions.get(_sample_key(sample))
        if not candidates:
            return None
        selected.append(candidates.popleft())
    if selected != sorted(selected):
        return None

    ensure_dir(cache_dir)
    for old_shard in cache_dir.glob("shard_*.pt"):
        old_shard.unlink()
    started = time.time()
    target_cursor = 0
    source_offset = 0
    buffered_hidden: list[torch.Tensor] = []
    buffered_labels: list[torch.Tensor] = []
    buffered_sources: list[str] = []
    buffered_count = 0
    shards: list[dict[str, Any]] = []

    def flush(force: bool = False) -> None:
        nonlocal buffered_hidden, buffered_labels, buffered_sources, buffered_count
        if buffered_count == 0 or (not force and buffered_count < int(shard_size)):
            return
        hidden = torch.cat(buffered_hidden, dim=0)
        labels = torch.cat(buffered_labels, dim=0)
        write_count = int(hidden.shape[0]) if force else (int(hidden.shape[0]) // int(shard_size)) * int(shard_size)
        offset = 0
        while offset < write_count:
            take = min(int(shard_size), write_count - offset)
            filename = f"shard_{len(shards):05d}.pt"
            torch.save(
                {
                    "hidden": hidden[offset : offset + take].clone(),
                    "tool_ids": labels[offset : offset + take].clone(),
                    "source_stages": buffered_sources[offset : offset + take],
                },
                cache_dir / filename,
            )
            shards.append({"file": filename, "samples": take})
            offset += take
        buffered_hidden = [hidden[write_count:].clone()] if write_count < int(hidden.shape[0]) else []
        buffered_labels = [labels[write_count:].clone()] if write_count < int(labels.shape[0]) else []
        buffered_sources = buffered_sources[write_count:]
        buffered_count = int(hidden.shape[0]) - write_count

    source_manifest = load_json(source_cache_dir / "manifest.json")
    for shard in source_manifest["shards"]:
        payload = torch.load(source_cache_dir / shard["file"], map_location="cpu", weights_only=False)
        shard_count = int(payload["hidden"].shape[0])
        local_indices: list[int] = []
        while target_cursor < len(selected) and selected[target_cursor] < source_offset + shard_count:
            local_indices.append(selected[target_cursor] - source_offset)
            target_cursor += 1
        if local_indices:
            index_tensor = torch.tensor(local_indices, dtype=torch.long)
            buffered_hidden.append(payload["hidden"].index_select(0, index_tensor))
            buffered_labels.append(payload["tool_ids"].index_select(0, index_tensor))
            buffered_sources.extend(payload["source_stages"][index] for index in local_indices)
            buffered_count += len(local_indices)
            flush()
        source_offset += shard_count
    flush(force=True)
    if target_cursor != len(target_samples):
        raise AssertionError(f"Only materialized {target_cursor}/{len(target_samples)} cached samples")
    manifest = {
        "format": "pure_ewcdr_frozen_llama_features_v1",
        "samples": len(target_samples),
        "hidden_size": int(hidden_size),
        "dtype": "bfloat16",
        "max_length": int(max_length),
        "sample_digest": samples_digest(target_samples),
        "length_bucketed_encoding": bool(source_manifest.get("length_bucketed_encoding", False)),
        "derived_without_encoder_forward": True,
        "source_cache": str(source_cache_dir.resolve()),
        "source_sample_digest": source_manifest.get("sample_digest"),
        "shards": shards,
        "duration_sec": round(time.time() - started, 3),
    }
    save_json(cache_dir / "manifest.json", manifest)
    logger.info("derived feature cache from %s: %s samples=%s", source_cache_dir, cache_dir, len(target_samples))
    return manifest


@torch.no_grad()
def build_feature_cache(
    encoder: FrozenLlamaEncoder,
    samples: Sequence[PureSample],
    cache_dir: Path,
    *,
    batch_size: int,
    shard_size: int,
    dataloader_options: Mapping[str, Any],
    logger,
    length_bucketed: bool = False,
) -> dict[str, Any]:
    if cache_is_valid(cache_dir, samples, encoder.hidden_size, encoder.max_length):
        manifest = load_json(cache_dir / "manifest.json")
        logger.info("reusing frozen-encoder feature cache: %s", cache_dir)
        return manifest
    ensure_dir(cache_dir)
    for old_shard in cache_dir.glob("shard_*.pt"):
        old_shard.unlink()
    encode_order = list(range(len(samples)))
    if length_bucketed:
        encode_order.sort(key=lambda index: (len(samples[index].query_text), index))
    encoding_samples = [samples[index] for index in encode_order]
    loader = make_loader(
        TextDataset(encoding_samples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(dataloader_options.get("num_workers", 0)),
        pin_memory=bool(dataloader_options.get("pin_memory", False)),
        persistent_workers=bool(dataloader_options.get("persistent_workers", False)),
        prefetch_factor=dataloader_options.get("prefetch_factor"),
    )
    started = time.time()
    buffered_hidden: list[torch.Tensor] = []
    buffered_labels: list[torch.Tensor] = []
    buffered_sources: list[str] = []
    buffered_count = 0
    shards: list[dict[str, Any]] = []
    reordered_hidden = (
        torch.empty((len(samples), encoder.hidden_size), dtype=torch.bfloat16)
        if length_bucketed
        else None
    )
    encoded_offset = 0

    def flush() -> None:
        nonlocal buffered_hidden, buffered_labels, buffered_sources, buffered_count
        if buffered_count == 0:
            return
        hidden = torch.cat(buffered_hidden, dim=0)
        labels = torch.cat(buffered_labels, dim=0)
        while hidden.shape[0] > 0:
            take = min(int(shard_size), int(hidden.shape[0]))
            filename = f"shard_{len(shards):05d}.pt"
            payload = {
                "hidden": hidden[:take].contiguous(),
                "tool_ids": labels[:take].contiguous(),
                "source_stages": buffered_sources[:take],
            }
            torch.save(payload, cache_dir / filename)
            shards.append({"file": filename, "samples": take})
            hidden = hidden[take:]
            labels = labels[take:]
            buffered_sources = buffered_sources[take:]
        buffered_hidden = []
        buffered_labels = []
        buffered_sources = []
        buffered_count = 0

    encoder.eval()
    for batch in tqdm(loader, desc=f"encode {cache_dir.name}", dynamic_ncols=True, mininterval=2.0):
        hidden = encoder(batch["query_text"]).to(device="cpu", dtype=torch.bfloat16)
        if reordered_hidden is not None:
            batch_indices = encode_order[encoded_offset : encoded_offset + int(hidden.shape[0])]
            reordered_hidden.index_copy_(0, torch.tensor(batch_indices, dtype=torch.long), hidden)
            encoded_offset += int(hidden.shape[0])
        else:
            buffered_hidden.append(hidden)
            buffered_labels.append(batch["tool_id"].cpu().long())
            buffered_sources.extend(batch["source_stage"])
            buffered_count += int(hidden.shape[0])
            if buffered_count >= int(shard_size):
                flush()
    if reordered_hidden is not None:
        labels = torch.tensor([sample.tool_id for sample in samples], dtype=torch.long)
        sources = [sample.source_stage for sample in samples]
        for start in range(0, len(samples), int(shard_size)):
            stop = min(start + int(shard_size), len(samples))
            filename = f"shard_{len(shards):05d}.pt"
            torch.save(
                {
                    "hidden": reordered_hidden[start:stop].clone(),
                    "tool_ids": labels[start:stop].clone(),
                    "source_stages": sources[start:stop],
                },
                cache_dir / filename,
            )
            shards.append({"file": filename, "samples": stop - start})
    else:
        flush()
    manifest = {
        "format": "pure_ewcdr_frozen_llama_features_v1",
        "samples": len(samples),
        "hidden_size": encoder.hidden_size,
        "dtype": "bfloat16",
        "max_length": encoder.max_length,
        "sample_digest": samples_digest(samples),
        "length_bucketed_encoding": bool(length_bucketed),
        "shards": shards,
        "duration_sec": round(time.time() - started, 3),
    }
    save_json(cache_dir / "manifest.json", manifest)
    logger.info("saved feature cache: %s samples=%s duration=%.1fs", cache_dir, len(samples), manifest["duration_sec"])
    return manifest


def load_feature_dataset(cache_dir: Path) -> tuple[FeatureDataset, dict[str, Any]]:
    manifest = load_json(cache_dir / "manifest.json")
    count = int(manifest["samples"])
    hidden_size = int(manifest["hidden_size"])
    hidden = torch.empty((count, hidden_size), dtype=torch.bfloat16)
    labels = torch.empty((count,), dtype=torch.long)
    sources: list[str] = []
    offset = 0
    for shard in manifest["shards"]:
        payload = torch.load(cache_dir / shard["file"], map_location="cpu", weights_only=False)
        take = int(payload["hidden"].shape[0])
        hidden[offset : offset + take].copy_(payload["hidden"])
        labels[offset : offset + take].copy_(payload["tool_ids"])
        sources.extend(payload["source_stages"])
        offset += take
    if offset != count:
        raise AssertionError(f"Cache manifest says {count} samples but loaded {offset}")
    return FeatureDataset(hidden, labels, sources), manifest
