#!/usr/bin/env python3
"""
Summarize RQ2-4 week-level aggregate results into paper-ready tables.

Input:
- aggregate_results.csv from run_pe_feature_intervention_validation.py

Outputs:
- summary_over_weeks.csv
- manuscript_tables/summary_table.csv
- manuscript_tables/summary_table.md
- manuscript_tables/imports_focus_table.csv
- manuscript_tables/imports_focus_table.md
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


FEATURE_ORDER = ["header", "imports"]
SELECTION_TYPE_ORDER = ["important", "random"]


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def selection_sort_key(row: Dict[str, str]) -> tuple:
    feature_group = row["feature_group"]
    selection_mode = row["selection_mode"]
    selection_value = float(row["selection_value"])
    feature_order = FEATURE_ORDER.index(feature_group) if feature_group in FEATURE_ORDER else 99
    selection_type_order = (
        SELECTION_TYPE_ORDER.index(row["selection_type"])
        if row["selection_type"] in SELECTION_TYPE_ORDER
        else 99
    )
    mode_order = 0 if selection_mode == "budget" else 1 if selection_mode == "ratio" else 2
    return (
        feature_order,
        mode_order,
        selection_value,
        selection_type_order,
        row["operator"],
    )


def format_selection_label(selection_mode: str, selection_value: float) -> str:
    if selection_mode == "budget":
        return f"Top-{int(round(selection_value))}"
    if selection_mode == "ratio":
        return f"{int(round(selection_value * 100))}%"
    return "Baseline"


def build_summary_over_weeks(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple, List[Dict[str, str]]] = {}
    for row in rows:
        key = (
            row["feature_group"],
            row["operator"],
            row["selection_type"],
            row["selection_mode"],
            row["selection_value"],
        )
        grouped.setdefault(key, []).append(row)

    metric_prefixes = [
        "auc",
        "delta_auc",
        "flip_rate",
        "malware_to_benign_rate",
        "mean_abs_probability_shift",
        "mean_signed_probability_shift",
        "mean_margin_shift",
        "boundary_low_10_flip_rate",
        "boundary_low_10_malware_to_benign_rate",
    ]

    out_rows: List[Dict[str, object]] = []
    for key, group_rows in grouped.items():
        feature_group, operator, selection_type, selection_mode, selection_value = key
        out: Dict[str, object] = {
            "feature_group": feature_group,
            "operator": operator,
            "selection_type": selection_type,
            "selection_mode": selection_mode,
            "selection_value": float(selection_value),
            "selection_label": format_selection_label(selection_mode, float(selection_value)),
            "n_weeks": len(group_rows),
            "n_selected_features_mean": float(
                np.mean([float(row["n_selected_features"]) for row in group_rows])
            ),
            "n_candidate_features": int(float(group_rows[0]["n_candidate_features"])),
        }
        for metric in metric_prefixes:
            column = f"{metric}_mean"
            stats = summarize(float(row[column]) for row in group_rows)
            for stat_name, stat_value in stats.items():
                out[f"{metric}_across_weeks_{stat_name}"] = stat_value
        out_rows.append(out)

    out_rows.sort(key=lambda row: selection_sort_key({k: str(v) for k, v in row.items()}))
    return out_rows


def fmt_num(value: float, decimals: int = 3) -> str:
    return f"{value:.{decimals}f}"


def build_summary_table(rows: List[Dict[str, object]]) -> List[Dict[str, str]]:
    table_rows: List[Dict[str, str]] = []
    filtered = [row for row in rows if row["selection_type"] in {"important", "random"}]
    filtered.sort(key=lambda row: selection_sort_key({k: str(v) for k, v in row.items()}))
    for row in filtered:
        table_rows.append(
            {
                "Subset": str(row["feature_group"]).capitalize(),
                "Selection": str(row["selection_label"]),
                "Type": str(row["selection_type"]).capitalize(),
                "Delta AUC": fmt_num(float(row["delta_auc_across_weeks_mean"])),
                "Flip Rate": fmt_num(float(row["flip_rate_across_weeks_mean"])),
                "M->B Rate": fmt_num(float(row["malware_to_benign_rate_across_weeks_mean"])),
                "Signed Shift": fmt_num(float(row["mean_signed_probability_shift_across_weeks_mean"])),
            }
        )
    return table_rows


def build_imports_focus_table(rows: List[Dict[str, object]]) -> List[Dict[str, str]]:
    lookup = {
        (
            row["feature_group"],
            row["selection_type"],
            row["selection_mode"],
            float(row["selection_value"]),
        ): row
        for row in rows
        if row["feature_group"] == "imports" and row["selection_type"] in {"important", "random"}
    }

    selection_keys = sorted(
        {
            (row["selection_mode"], float(row["selection_value"]))
            for row in rows
            if row["feature_group"] == "imports" and row["selection_type"] == "important"
        },
        key=lambda item: (0 if item[0] == "budget" else 1, item[1]),
    )

    table_rows: List[Dict[str, str]] = []
    for selection_mode, selection_value in selection_keys:
        imp = lookup[("imports", "important", selection_mode, selection_value)]
        rnd = lookup[("imports", "random", selection_mode, selection_value)]
        table_rows.append(
            {
                "Selection": format_selection_label(selection_mode, selection_value),
                "Delta AUC (Imp)": fmt_num(float(imp["delta_auc_across_weeks_mean"])),
                "Delta AUC (Rand)": fmt_num(float(rnd["delta_auc_across_weeks_mean"])),
                "M->B Rate (Imp)": fmt_num(float(imp["malware_to_benign_rate_across_weeks_mean"])),
                "M->B Rate (Rand)": fmt_num(float(rnd["malware_to_benign_rate_across_weeks_mean"])),
                "Signed Shift (Imp)": fmt_num(float(imp["mean_signed_probability_shift_across_weeks_mean"])),
                "Signed Shift (Rand)": fmt_num(float(rnd["mean_signed_probability_shift_across_weeks_mean"])),
                "Flip Rate (Imp)": fmt_num(float(imp["flip_rate_across_weeks_mean"])),
                "Flip Rate (Rand)": fmt_num(float(rnd["flip_rate_across_weeks_mean"])),
            }
        )
    return table_rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize RQ2-4 PE-inspired intervention results.")
    parser.add_argument("--input-csv", type=Path, required=True, help="Path to aggregate_results.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for summary outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = build_summary_over_weeks(rows)
    write_csv(args.output_dir / "summary_over_weeks.csv", summary_rows)

    manuscript_dir = args.output_dir / "manuscript_tables"
    summary_table_rows = build_summary_table(summary_rows)
    imports_focus_rows = build_imports_focus_table(summary_rows)
    write_csv(manuscript_dir / "summary_table.csv", summary_table_rows)
    write_markdown(manuscript_dir / "summary_table.md", summary_table_rows)
    write_csv(manuscript_dir / "imports_focus_table.csv", imports_focus_rows)
    write_markdown(manuscript_dir / "imports_focus_table.md", imports_focus_rows)
    print(f"Saved summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
