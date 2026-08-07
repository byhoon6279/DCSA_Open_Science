#!/usr/bin/env python3
"""
Build the manuscript-facing measurement summary table used for Table 3.

This table combines:
- AUC / Mix@10 / JS from the LR class-balanced RQ1 summary
- Silhouette from the matched LR class-balanced rerun with silhouette enabled
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict


BASE_DIR = Path(__file__).resolve().parents[2]
TABLE_DIR = BASE_DIR / "results" / "manuscript_tables"
REPO_ROOT = BASE_DIR.parents[1]

DEFAULT_BALANCED_SUMMARY = (
    BASE_DIR
    / "results"
    / "experiment_results"
    / "win32_all_train_all_test_balanced_test"
    / "balanced_test_aggregate_summary.csv"
)
DEFAULT_SILHOUETTE_SUMMARY = (
    REPO_ROOT
    / "Tmp"
    / "RQ1"
    / "RQ1-1"
    / "results"
    / "experiment_runs"
    / "win32_all_train_all_test_balanced_test_with_silhouette"
    / "aggregate_summary.csv"
)
DEFAULT_OUTPUT = TABLE_DIR / "table_rq1_measurement_summary.csv"

FEATURE_ORDER = ["all", "header", "section", "strings", "imports"]


def load_metric_summary(path: Path) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["scope"] != "metric":
                continue
            summary.setdefault(row["feature_group"], {})[row["metric"]] = float(row["mean"])
    return summary


def write_table(
    balanced_summary: Dict[str, Dict[str, float]],
    silhouette_summary: Dict[str, Dict[str, float]],
    out_path: Path,
) -> None:
    rows = []
    for feature in FEATURE_ORDER:
        balanced_metrics = balanced_summary[feature]
        silhouette_metrics = silhouette_summary[feature]
        rows.append(
            {
                "feature_subset": feature,
                "auc": f"{balanced_metrics['auc']:.4f}",
                "mix_at_10": f"{balanced_metrics['mix_at_k']:.4f}",
                "silhouette": f"{silhouette_metrics['silhouette']:.4f}",
                "js_divergence": f"{balanced_metrics['js_divergence']:.4f}",
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["feature_subset", "auc", "mix_at_10", "silhouette", "js_divergence"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--balanced-summary", type=Path, default=DEFAULT_BALANCED_SUMMARY)
    parser.add_argument("--silhouette-summary", type=Path, default=DEFAULT_SILHOUETTE_SUMMARY)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    balanced_summary = load_metric_summary(args.balanced_summary)
    silhouette_summary = load_metric_summary(args.silhouette_summary)
    write_table(balanced_summary, silhouette_summary, args.output_csv)
    print(f"Wrote measurement summary table to {args.output_csv}")


if __name__ == "__main__":
    main()
