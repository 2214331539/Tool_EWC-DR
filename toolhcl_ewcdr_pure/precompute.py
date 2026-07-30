from __future__ import annotations

import argparse

import torch

from .cache import build_feature_cache
from .data import build_global_eval_samples, build_stage_samples, load_stage_tools
from .model import load_frozen_encoder
from .utils import dataloader_options, load_config, project_root, protocol_stages, resolve_device, resolve_path, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute disjoint pure EWC-DR frozen-encoder cache splits")
    parser.add_argument("--config", required=True)
    parser.add_argument("--splits", required=True, help="Comma-separated names such as task1_train,global_eval")
    parser.add_argument("--log_dir", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    device = resolve_device(config["runtime"].get("device", "cuda"), config["runtime"].get("gpu"))
    logger = setup_logging(resolve_path(args.log_dir, project_root(config)), "precompute")
    records = load_stage_tools(config)
    stages = protocol_stages(config)
    requested = [value.strip() for value in args.splits.split(",") if value.strip()]
    specs = []
    for name in requested:
        if name == "global_eval":
            samples, _ = build_global_eval_samples(config, records=records)
        elif name.endswith("_train") and name[:-6] in stages:
            samples, _ = build_stage_samples(config, name[:-6], "train", records=records)
        else:
            raise ValueError(f"Unsupported cache split: {name}")
        specs.append((name, samples))
    encoder = load_frozen_encoder(config, device)
    cache_root = resolve_path(config["cache"]["root"], project_root(config))
    try:
        for name, samples in specs:
            build_feature_cache(
                encoder,
                samples,
                cache_root / name,
                batch_size=int(config["cache"].get("encoder_batch_size", 64)),
                shard_size=int(config["cache"].get("shard_size", 4096)),
                dataloader_options=dataloader_options(config, "cache"),
                logger=logger,
                length_bucketed=bool(config["cache"].get("length_bucketed", False)),
            )
    finally:
        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    logger.info("parallel cache precompute complete splits=%s", requested)


if __name__ == "__main__":
    main()
