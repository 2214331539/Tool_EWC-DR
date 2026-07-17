from __future__ import annotations

import hashlib
import time
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
) -> dict[str, Any]:
    if cache_is_valid(cache_dir, samples, encoder.hidden_size, encoder.max_length):
        manifest = load_json(cache_dir / "manifest.json")
        logger.info("reusing frozen-encoder feature cache: %s", cache_dir)
        return manifest
    ensure_dir(cache_dir)
    for old_shard in cache_dir.glob("shard_*.pt"):
        old_shard.unlink()
    loader = make_loader(
        TextDataset(samples),
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
        buffered_hidden.append(hidden)
        buffered_labels.append(batch["tool_id"].cpu().long())
        buffered_sources.extend(batch["source_stage"])
        buffered_count += int(hidden.shape[0])
        if buffered_count >= int(shard_size):
            flush()
    flush()
    manifest = {
        "format": "pure_ewcdr_frozen_llama_features_v1",
        "samples": len(samples),
        "hidden_size": encoder.hidden_size,
        "dtype": "bfloat16",
        "max_length": encoder.max_length,
        "sample_digest": samples_digest(samples),
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
