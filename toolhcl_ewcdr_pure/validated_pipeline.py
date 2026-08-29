from __future__ import annotations

import argparse
import copy
import shutil
from pathlib import Path
from typing import Any

from .evaluate import evaluate
from .train import train
from .utils import STAGES, load_config, load_json, project_root, resolve_path, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run validation-selected, full-data EWC-DR training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--selection_dir", default=None)
    parser.add_argument("--method", choices=("seq_ft", "ewc", "ewc_dr"), default="ewc_dr")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override both the training seed and validation-split seed.",
    )
    parser.add_argument("--keep_selection_checkpoints", action="store_true")
    return parser.parse_args()


def _require_new_directory(path: Path) -> None:
    if not path.exists():
        return
    entries = list(path.iterdir())
    if not entries:
        return
    # Shell launchers may open the tee log before Python starts.
    if len(entries) == 1 and entries[0].name == "logs":
        unexpected_logs = [item for item in entries[0].iterdir() if item.name != "full_pipeline.log"]
        if not unexpected_logs:
            return
    raise FileExistsError(f"Refusing to overwrite existing run directory: {path}")


def run_validated_pipeline(
    config: dict[str, Any],
    *,
    output_dir_value: str,
    selection_dir_value: str | None = None,
    method: str = "ewc_dr",
    seed: int | None = None,
    keep_selection_checkpoints: bool = False,
) -> Path:
    config = copy.deepcopy(config)
    if seed is not None:
        config["training"]["seed"] = seed
        config["training"].setdefault("validation", {})["seed"] = seed

    root = project_root(config)
    output_dir = resolve_path(output_dir_value, root)
    selection_dir = resolve_path(selection_dir_value or f"{output_dir_value}_selection", root)
    _require_new_directory(output_dir)
    _require_new_directory(selection_dir)

    selection_config = copy.deepcopy(config)
    selection_config["project"]["output_dir"] = str(selection_dir)
    selection_config["ewcdr"]["method"] = method
    selection_config["training"]["checkpoint_selection"] = "validation"
    selection_config["training"].setdefault("validation", {})["enabled"] = True
    # Selection must never inspect or prepare the test set.
    selection_config["cache"]["prepare_global_eval"] = False
    train(selection_config)

    selection_summary = load_json(selection_dir / "training_summary.json")
    selected_epochs = {
        row["stage"]: int(row["best_epoch"])
        for row in selection_summary.get("stages", [])
    }
    expected_stages = tuple(config["training"].get("stages", STAGES))
    if set(selected_epochs) != set(expected_stages) or any(value <= 0 for value in selected_epochs.values()):
        raise RuntimeError(f"Selection pass did not produce valid epochs for every stage: {selected_epochs}")

    final_config = copy.deepcopy(config)
    final_config["project"]["output_dir"] = str(output_dir)
    final_config["ewcdr"]["method"] = method
    final_config["training"]["checkpoint_selection"] = "last_epoch"
    final_config["training"].setdefault("validation", {})["enabled"] = False
    final_config["training"]["stage_epochs"] = {
        stage: {"min": selected_epochs[stage], "max": selected_epochs[stage]}
        for stage in expected_stages
    }
    final_config["cache"]["prepare_global_eval"] = True
    final_config["selection_protocol"] = {
        "selection_pass": "90% tool-ID-stratified train partition",
        "final_pass": "100% original train split from a fresh initialization",
        "test_usage": "The test/global splits are used only after final training",
        "metric": "harmonic mean of historical-task mean Recall@1 and current-task Recall@1",
        "selected_epochs": selected_epochs,
        "selection_output": str(selection_dir),
        "validation": selection_summary.get("validation", {}),
    }
    train(final_config)

    selection_artifacts = output_dir / "selection"
    selection_artifacts.mkdir(parents=True, exist_ok=False)
    save_json(selection_artifacts / "training_summary.json", selection_summary)
    save_json(selection_artifacts / "config.json", selection_config)
    save_json(output_dir / "selection_manifest.json", final_config["selection_protocol"])
    evaluate(final_config)

    if not keep_selection_checkpoints:
        shutil.rmtree(selection_dir)
    return output_dir


def main() -> None:
    args = parse_args()
    run_validated_pipeline(
        load_config(args.config),
        output_dir_value=args.output_dir,
        selection_dir_value=args.selection_dir,
        method=args.method,
        seed=args.seed,
        keep_selection_checkpoints=args.keep_selection_checkpoints,
    )


if __name__ == "__main__":
    main()
