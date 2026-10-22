#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


FEATURE_ORDER = ["all", "header", "section", "strings", "imports"]
LABELS = {
    "all": "All",
    "header": "Header",
    "section": "Section",
    "strings": "Strings",
    "imports": "Imports",
}
COLORS = {
    "all": "#264653",
    "header": "#2A9D8F",
    "section": "#E9C46A",
    "strings": "#F4A261",
    "imports": "#E76F51",
}
OLD_ROOT = Path(__file__).resolve().parents[1] / "5_1_representation_level_separability_disagreement/results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True, type=Path)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_new(result_dir: Path, view: str, k: int = 10) -> dict[tuple[str, str], dict[str, float]]:
    rows = read_csv(result_dir / view / "aggregate_metrics.csv")
    output: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        if int(row["k"]) != k:
            continue
        output[(row["model"], row["feature_group"])] = {
            "geometry": float(row["same_family_rate_mean"]),
            "f1": float(row["family_macro_f1_mean"]),
        }
    return output


def old_paths(model: str, view: str) -> tuple[Path, Path]:
    if model == "lr":
        suffix = "family_aligned_rq1_matched" if view == "wild_b" else "family_aligned_rq1_matched_unpacked"
        directory = OLD_ROOT / "LR" / suffix
        summary_name = "matched_aggregate_summary.csv" if view == "wild_b" else "matched_unpacked_aggregate_summary.csv"
    elif model == "lightgbm":
        suffix = (
            "family_aligned_rq1_matched_lightgbm"
            if view == "wild_b"
            else "family_aligned_rq1_matched_unpacked_lightgbm"
        )
        directory = OLD_ROOT / "LightGBM" / suffix
        summary_name = "aggregate_summary.csv"
    else:
        # The historical RF Figure 2 result only preserved its Wild(B) family-
        # aligned output. Missing Unpacked(B) cells are reported explicitly.
        directory = OLD_ROOT / "RF" / "classification_vs_structure_wild_b"
        summary_name = "aggregate_summary.csv"
    return directory / summary_name, directory / "aggregate_by_k.csv"


def load_old(model: str, view: str, k: int = 10) -> dict[str, dict[str, float]]:
    summary_path, by_k_path = old_paths(model, view)
    if view == "unpacked_b" and model == "rf":
        return {}
    if not summary_path.exists() or not by_k_path.exists():
        return {}
    summary_rows = read_csv(summary_path)
    by_k_rows = read_csv(by_k_path)
    f1 = {
        row["feature_group"]: float(row["mean"])
        for row in summary_rows
        if row.get("scope") == "metric" and row.get("metric") == "family_macro_f1"
    }
    geometry = {
        row["feature_group"]: float(row["mean"])
        for row in by_k_rows
        if int(row.get("k", -1)) == k
        and row.get("scope") == "metric"
        and row.get("metric") == "same_family_rate_at_k"
    }
    return {
        group: {"f1": f1[group], "geometry": geometry[group]}
        for group in FEATURE_ORDER
        if group in f1 and group in geometry
    }


def rank_map(values: dict[str, float]) -> dict[str, int]:
    ordered = sorted(values, key=lambda group: (-values[group], group))
    return {group: index + 1 for index, group in enumerate(ordered)}


def create_comparison(result_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for view in ["wild_b", "unpacked_b"]:
        new = load_new(result_dir, view)
        for model in ["lr", "lightgbm", "rf"]:
            old = load_old(model, view)
            new_geometry = {group: new[(model, group)]["geometry"] for group in FEATURE_ORDER}
            new_f1 = {group: new[(model, group)]["f1"] for group in FEATURE_ORDER}
            new_geometry_ranks = rank_map(new_geometry)
            new_f1_ranks = rank_map(new_f1)
            old_geometry_ranks = rank_map({group: values["geometry"] for group, values in old.items()}) if old else {}
            old_f1_ranks = rank_map({group: values["f1"] for group, values in old.items()}) if old else {}
            for group in FEATURE_ORDER:
                old_values = old.get(group, {})
                rows.append(
                    {
                        "view": view,
                        "model": model,
                        "feature_group": group,
                        "old_same_family_rate_k10": old_values.get("geometry", ""),
                        "new_same_family_rate_k10": new_geometry[group],
                        "geometry_delta": (
                            new_geometry[group] - old_values["geometry"] if "geometry" in old_values else ""
                        ),
                        "old_geometry_rank": old_geometry_ranks.get(group, ""),
                        "new_geometry_rank": new_geometry_ranks[group],
                        "old_macro_f1": old_values.get("f1", ""),
                        "new_macro_f1": new_f1[group],
                        "macro_f1_delta": new_f1[group] - old_values["f1"] if "f1" in old_values else "",
                        "old_f1_rank": old_f1_ranks.get(group, ""),
                        "new_f1_rank": new_f1_ranks[group],
                    }
                )
    return rows


def plot_view(result_dir: Path, view: str, common_limits: tuple[float, float, float, float]) -> None:
    data = load_new(result_dir, view)
    x_low, x_high, y_low, y_high = common_limits
    figures = result_dir / "figures"
    figures.mkdir(exist_ok=True)
    for model, title in [("lr", "LR"), ("lightgbm", "LightGBM"), ("rf", "RF")]:
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        for group in FEATURE_ORDER:
            point = data[(model, group)]
            ax.scatter(point["f1"], point["geometry"], s=90, color=COLORS[group], zorder=3)
            ax.annotate(
                LABELS[group],
                (point["f1"], point["geometry"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=11,
            )
        ax.set_xlim(x_low, x_high)
        ax.set_ylim(y_low, y_high)
        ax.set_xlabel("Classification Performance (Macro-F1)")
        ax.set_ylabel("Same-Family Rate@10")
        ax.set_title(f"{title} {view.replace('_b', ' (B)').replace('_', ' ').title()}")
        ax.grid(True, linewidth=0.8, alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(figures / f"classification_vs_structure_scatter_{view}_{model}.pdf", bbox_inches="tight")
        fig.savefig(figures / f"classification_vs_structure_scatter_{view}_{model}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    comparison = create_comparison(result_dir)
    write_csv(result_dir / "before_after_comparison.csv", comparison)

    all_points: list[dict[str, float]] = []
    for view in ["wild_b", "unpacked_b"]:
        all_points.extend(load_new(result_dir, view).values())
    xs = np.asarray([point["f1"] for point in all_points])
    ys = np.asarray([point["geometry"] for point in all_points])
    x_pad = max(0.01, float(np.ptp(xs)) * 0.08)
    y_pad = max(0.01, float(np.ptp(ys)) * 0.08)
    limits = (float(xs.min() - x_pad), float(xs.max() + x_pad), float(ys.min() - y_pad), float(ys.max() + y_pad))
    for view in ["wild_b", "unpacked_b"]:
        plot_view(result_dir, view, limits)

    changed_geometry_ranks = sum(
        1
        for row in comparison
        if row["old_geometry_rank"] != "" and row["old_geometry_rank"] != row["new_geometry_rank"]
    )
    changed_f1_ranks = sum(
        1
        for row in comparison
        if row["old_f1_rank"] != "" and row["old_f1_rank"] != row["new_f1_rank"]
    )
    report = f"""# Canonical Figure 2 reanalysis report

## Protocol

- Geometry is computed once per view, seed, week, and feature group.
- LR, LightGBM, and RF share the selected records, record order, scaler, and transformed matrices.
- Same-family rate uses k=10 for Figure 2. Other k values are retained only as sensitivity outputs.
- Self-neighbors are excluded by explicit row-index comparison.
- All panels and both views use the global axis range `{limits}`.

## Before/after summary

- Geometry rank cells changed: {changed_geometry_ranks}
- Macro-F1 rank cells changed: {changed_f1_ranks}
- Complete values and deltas: `before_after_comparison.csv`

## Manuscript update locations

1. Figure 2 panels in `sections/4_Experimental_Protocol.tex`.
2. Section 5.1 values, rankings, and interpretation.
3. Appendix A family-aligned sensitivity text and any corresponding table.
4. Any table or figure that pairs family Macro-F1 with same-family rate@10.

Claims should be updated only after reviewing the complete comparison table.
"""
    (result_dir / "result_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
