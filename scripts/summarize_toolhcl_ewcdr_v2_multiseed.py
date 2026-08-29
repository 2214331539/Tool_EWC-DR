from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


METRICS = ("Recall@1", "Recall@3", "Recall@5", "NDCG@1", "NDCG@3", "NDCG@5", "MRR")
KEY_FIELDS = ("checkpoint", "eval_split", "samples", "candidates")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate ToolBench EWC-DR V2 seed runs")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="SEED=PATH",
        help="Seed and completed run directory; repeat once per seed.",
    )
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def parse_runs(values: list[str]) -> list[tuple[int, Path]]:
    runs: list[tuple[int, Path]] = []
    for value in values:
        seed_text, separator, path_text = value.partition("=")
        if not separator:
            raise ValueError(f"Expected SEED=PATH, got: {value}")
        run_dir = Path(path_text).expanduser().resolve()
        for filename in ("eval_matrix.csv", "global_eval.csv", "selection_manifest.json"):
            if not (run_dir / filename).is_file():
                raise FileNotFoundError(run_dir / filename)
        runs.append((int(seed_text), run_dir))
    seeds = [seed for seed, _ in runs]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"Duplicate seeds: {seeds}")
    return sorted(runs)


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {path}")
        rows = list(reader)
        return list(reader.fieldnames), rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(
    seeded_rows: list[tuple[int, list[dict[str, str]]]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, ...], dict[str, list[float]]] = {}
    seed_sets: dict[tuple[str, ...], set[int]] = {}
    for seed, rows in seeded_rows:
        for row in rows:
            key = tuple(row[field] for field in KEY_FIELDS)
            grouped.setdefault(key, {metric: [] for metric in METRICS})
            seed_sets.setdefault(key, set()).add(seed)
            for metric in METRICS:
                grouped[key][metric].append(float(row[metric]))

    expected_seeds = {seed for seed, _ in seeded_rows}
    results: list[dict[str, object]] = []
    for key, values_by_metric in grouped.items():
        if seed_sets[key] != expected_seeds:
            raise ValueError(f"Missing seed rows for {key}: {seed_sets[key]} != {expected_seeds}")
        row: dict[str, object] = dict(zip(KEY_FIELDS, key))
        row["seeds"] = len(expected_seeds)
        for metric, values in values_by_metric.items():
            row[f"{metric}_mean"] = statistics.fmean(values)
            row[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            row[f"{metric}_min"] = min(values)
            row[f"{metric}_max"] = max(values)
        results.append(row)
    return results


def summary_markdown(
    runs: list[tuple[int, Path]],
    global_stats: list[dict[str, object]],
) -> str:
    lines = [
        "# ToolBench EWC-DR V2 Multi-Seed Results",
        "",
        "Metrics are percentages. Standard deviation is the sample standard deviation across seeds.",
        "",
        "## Runs",
        "",
        "| seed | selected epochs (base/task1/task2/task3) | run |",
        "| ---: | --- | --- |",
    ]
    for seed, run_dir in runs:
        with (run_dir / "selection_manifest.json").open(encoding="utf-8") as handle:
            selected = json.load(handle)["selected_epochs"]
        epochs = "/".join(str(selected[stage]) for stage in ("base", "task1", "task2", "task3"))
        lines.append(f"| {seed} | {epochs} | `{run_dir.name}` |")

    lines.extend(
        [
            "",
            "## Global Eval Mean and Range",
            "",
            "| checkpoint | R@1 mean+/-std [min,max] | R@3 | R@5 | MRR |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in global_stats:
        cells = []
        for metric in ("Recall@1", "Recall@3", "Recall@5", "MRR"):
            cells.append(
                f"{float(row[f'{metric}_mean']):.4f}+/-{float(row[f'{metric}_std']):.4f} "
                f"[{float(row[f'{metric}_min']):.4f},{float(row[f'{metric}_max']):.4f}]"
            )
        lines.append(f"| {row['checkpoint']} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    runs = parse_runs(args.run)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, combined_name, stats_name in (
        ("eval_matrix.csv", "all_seed_eval_matrix.csv", "seen_stats.csv"),
        ("global_eval.csv", "all_seed_global_eval.csv", "global_stats.csv"),
    ):
        seeded_rows: list[tuple[int, list[dict[str, str]]]] = []
        combined_rows: list[dict[str, object]] = []
        fieldnames: list[str] | None = None
        for seed, run_dir in runs:
            current_fields, rows = load_csv(run_dir / filename)
            if fieldnames is None:
                fieldnames = current_fields
            elif current_fields != fieldnames:
                raise ValueError(f"CSV schema mismatch in {run_dir / filename}")
            seeded_rows.append((seed, rows))
            combined_rows.extend({"seed": seed, **row} for row in rows)
        assert fieldnames is not None
        write_csv(output_dir / combined_name, ["seed", *fieldnames], combined_rows)
        stats = aggregate_rows(seeded_rows)
        stats_fields = [*KEY_FIELDS, "seeds"] + [
            f"{metric}_{stat}"
            for metric in METRICS
            for stat in ("mean", "std", "min", "max")
        ]
        write_csv(output_dir / stats_name, stats_fields, stats)
        if filename == "global_eval.csv":
            global_stats = stats

    (output_dir / "summary.md").write_text(summary_markdown(runs, global_stats), encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
