#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def aggregate_summary_rows(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    if not rows:
        return []

    metric_keys = [
        "auc",
        "flip_rate",
        "malware_to_benign_rate",
        "mean_abs_probability_shift",
        "mean_signed_probability_shift",
        "mean_margin_shift",
        "baseline_auc",
        "delta_auc",
    ]
    boundary_metric_keys = sorted(
        key
        for key in rows[0].keys()
        if key.startswith("boundary_low_")
    )
    metric_keys.extend(boundary_metric_keys)

    group_keys = [
        "feature_group",
        "operator",
        "selection_type",
        "selection_mode",
        "selection_value",
        "density_scope",
        "density_bin",
        "n_selected_features",
    ]

    grouped: Dict[tuple, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = tuple(row[name] for name in group_keys)
        grouped[key].append(row)

    aggregate_rows: List[Dict[str, object]] = []
    for key, bucket in sorted(grouped.items()):
        out: Dict[str, object] = {name: value for name, value in zip(group_keys, key)}
        out["n_rows"] = len(bucket)
        out["n_seeds"] = len({row["seed"] for row in bucket})
        out["trial_ids_json"] = json.dumps(sorted({int(row["trial_id"]) for row in bucket}))

        n_samples = [int(row["n_samples"]) for row in bucket]
        n_malware = [int(row["n_malware"]) for row in bucket]
        n_benign = [int(row["n_benign"]) for row in bucket]
        out["n_samples_mean"] = float(np.mean(n_samples))
        out["n_malware_mean"] = float(np.mean(n_malware))
        out["n_benign_mean"] = float(np.mean(n_benign))

        for metric in metric_keys:
            stats = summarize([float(row[metric]) for row in bucket])
            out[f"{metric}_mean"] = stats["mean"]
            out[f"{metric}_std"] = stats["std"]
            out[f"{metric}_min"] = stats["min"]
            out[f"{metric}_max"] = stats["max"]

        aggregate_rows.append(out)

    return aggregate_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge seed-split density-conditioned intervention probe results.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_dirs = sorted(path for path in args.input_root.glob("seed_*") if path.is_dir())

    merged_summary_rows: List[Dict[str, str]] = []
    merged_density_rows: List[Dict[str, str]] = []
    configs: Dict[str, object] = {}

    for seed_dir in seed_dirs:
        merged_summary_rows.extend(read_csv_rows(seed_dir / "summary_rows.csv"))
        merged_density_rows.extend(read_csv_rows(seed_dir / "density_rows.csv"))
        config_path = seed_dir / "config.json"
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as handle:
                configs[seed_dir.name] = json.load(handle)

    aggregate_rows = aggregate_summary_rows(merged_summary_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(args.output_dir / "summary_rows.csv", merged_summary_rows)  # type: ignore[arg-type]
    if merged_density_rows:
        write_csv_rows(args.output_dir / "density_rows.csv", merged_density_rows)  # type: ignore[arg-type]
    if aggregate_rows:
        write_csv_rows(args.output_dir / "aggregate_summary_rows.csv", aggregate_rows)
    with (args.output_dir / "merge_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "input_root": str(args.input_root),
                "seed_dirs": [seed_dir.name for seed_dir in seed_dirs],
                "n_summary_rows": len(merged_summary_rows),
                "n_density_rows": len(merged_density_rows),
                "n_aggregate_rows": len(aggregate_rows),
                "configs": configs,
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
