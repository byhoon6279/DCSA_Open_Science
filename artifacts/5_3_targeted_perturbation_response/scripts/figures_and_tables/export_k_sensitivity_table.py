#!/usr/bin/env python3
"""
Export a compact Wild (B) k-sensitivity table for RQ2-2.

The table reports Delta Mix@k at 10% masking for:
- LR
- LightGBM-gain
- LightGBM-permutation

Rows are split by subset (Header / Imports) and masking type
(Important / Random), and columns correspond to k = 5, 10, 15, 20.

This script's output regenerates byte-for-byte from the raw per-k trial data
bundled under `results/{LR,LightGBM}/feature_perturbation_*/k_*/`, and matches
Appendix Table D.1 in the manuscript exactly.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


K_PATHS = {
    "5": "aggregate_results.csv",
    "10": None,  # handled per model
    "15": "aggregate_results.csv",
    "20": "aggregate_results.csv",
}

MODEL_DIRS = {
    "LR": "LR/feature_perturbation_balanced_main",
    "LightGBM-gain": "LightGBM/feature_perturbation_lightgbm_gain_balanced_main",
    "LightGBM-permutation": "LightGBM/feature_perturbation_lightgbm_permutation_balanced_main",
}

SUBSET_LABELS = {
    "header": "Header",
    "imports": "Imports",
}

MASKING_LABELS = {
    "important": "Important",
    "random": "Random",
}


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def resolve_k10_path(model: str, model_dir: Path) -> Path:
    if model == "LR":
        return model_dir / "k_10" / "balanced_main_aggregate_results.csv"
    return model_dir / "k_10" / "aggregate_results.csv"


def resolve_path(model: str, model_dir: Path, k: str) -> Path:
    if k == "10":
        return resolve_k10_path(model, model_dir)
    return model_dir / f"k_{k}" / "aggregate_results.csv"


def extract_delta_mix(path: Path, feature_group: str, perturbation_type: str) -> float:
    rows = read_rows(path)
    matches = [
        float(row["delta_mix_at_k_mean"])
        for row in rows
        if row["feature_group"] == feature_group
        and row["perturbation_type"] == perturbation_type
        and float(row["strength"]) == 0.10
    ]
    if not matches:
        raise ValueError(f"No matches found in {path} for {feature_group} / {perturbation_type} / 10%")
    return sum(matches) / len(matches)


def build_rows(results_dir: Path) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for model, rel_dir in MODEL_DIRS.items():
        model_dir = results_dir / rel_dir
        for feature_group, subset_label in SUBSET_LABELS.items():
            for perturbation_type, masking_label in MASKING_LABELS.items():
                row = {
                    "Model": model,
                    "Subset": subset_label,
                    "Masking": masking_label,
                }
                for k in ["5", "10", "15", "20"]:
                    value = extract_delta_mix(resolve_path(model, model_dir, k), feature_group, perturbation_type)
                    row[f"k={k}"] = f"{value:.4f}"
                out.append(row)
    return out


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a compact Wild (B) k-sensitivity table for RQ2-2.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows(args.results_dir)
    write_csv(args.output_csv, rows)
    print(f"Saved k-sensitivity table to {args.output_csv}")


if __name__ == "__main__":
    main()
