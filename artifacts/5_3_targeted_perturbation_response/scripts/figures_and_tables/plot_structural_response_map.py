#!/usr/bin/env python3
"""
Build the Figure 6 structural response maps and their Appendix Figure D.1
Wild (U)/Unpacked (B) counterparts.

Each figure:
- 1 row x 3 panels: Wild (U), Wild (B), Unpacked (B)
- x-axis: Decision-Level AUC Degradation
- y-axis: Change in Local Class Mixing (Delta Mix@10)
- color: feature subset (Header / Imports)
- point size: masking ratio (1% / 5% / 10%)
- important masking only
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42  # embed fonts as TrueType
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import numpy as np

TITLE_FONT_SIZE = 24
AXIS_LABEL_FONT_SIZE = 19
TICK_FONT_SIZE = 19
LEGEND_FONT_SIZE = 17

SETTINGS = ["Wild (U)", "Wild (B)", "Unpacked (B)"]
SETTING_SLUGS = {
    "Wild (U)": "wild_u",
    "Wild (B)": "wild_b",
    "Unpacked (B)": "unpacked_b",
}
LR_TO_CANONICAL = {
    "Baseline": "Wild (U)",
    "Controlled": "Wild (B)",
    "Robustness": "Unpacked (B)",
}
SUBSET_COLORS = {
    "Header": "#C44536",
    "Imports": "#2F6BFF",
}
STRENGTH_LABELS = {
    "1%": 100,
    "5%": 200,
    "10%": 320,
}
STRENGTH_FLOAT_TO_LABEL = {
    0.01: "1%",
    0.05: "5%",
    0.1: "10%",
}
FEATURE_TO_LABEL = {
    "header": "Header",
    "imports": "Imports",
}


def style_axis(ax) -> None:
    ax.tick_params(axis="x", labelsize=TICK_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.xaxis.grid(True, linestyle="-", linewidth=0.8, alpha=0.22)
    ax.yaxis.grid(True, linestyle="-", linewidth=0.8, alpha=0.28)
    ax.set_axisbelow(True)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_lr_points(path: Path) -> List[Dict[str, object]]:
    rows = read_csv(path)
    points: List[Dict[str, object]] = []
    for row in rows:
        points.append(
            {
                "setting": LR_TO_CANONICAL[row["Setting"]],
                "subset": row["Subset"],
                "strength": row["Strength"],
                "auc_drop": abs(float(row["Delta AUC (Imp)"])),
                "delta_mix": float(row["Delta Mix (Imp)"]),
            }
        )
    return points


def aggregate_lgbm_file(path: Path, setting: str) -> List[Dict[str, object]]:
    grouped: Dict[tuple, Dict[str, float]] = defaultdict(lambda: {"auc": 0.0, "mix": 0.0, "n": 0})
    for row in read_csv(path):
        if row["feature_group"] not in FEATURE_TO_LABEL:
            continue
        if row["perturbation_type"] != "important":
            continue
        strength = float(row["strength"])
        if strength not in STRENGTH_FLOAT_TO_LABEL:
            continue
        key = (FEATURE_TO_LABEL[row["feature_group"]], STRENGTH_FLOAT_TO_LABEL[strength])
        grouped[key]["auc"] += max(0.0, -float(row["delta_auc_mean"]))
        grouped[key]["mix"] += float(row["delta_mix_at_k_mean"])
        grouped[key]["n"] += 1

    points: List[Dict[str, object]] = []
    for (subset, strength), agg in grouped.items():
        n = max(int(agg["n"]), 1)
        points.append(
            {
                "setting": setting,
                "subset": subset,
                "strength": strength,
                "auc_drop": agg["auc"] / n,
                "delta_mix": agg["mix"] / n,
            }
        )
    return points


def load_lgbm_points(baseline_csv: Path, controlled_csv: Path, robustness_csv: Path) -> List[Dict[str, object]]:
    points: List[Dict[str, object]] = []
    points.extend(aggregate_lgbm_file(baseline_csv, "Wild (U)"))
    points.extend(aggregate_lgbm_file(controlled_csv, "Wild (B)"))
    points.extend(aggregate_lgbm_file(robustness_csv, "Unpacked (B)"))
    return points


def plot_single_view(
    points: List[Dict[str, object]],
    output_path: Path,
    model_label: str,
    setting: str,
) -> None:
    # Height (not just the 5.1 originally used) matters here: with the current
    # two-line y-label, a shorter initial canvas gets the label's top edge
    # clipped during the first render pass, before bbox_inches="tight" can
    # crop - matplotlib doesn't re-render at a larger canvas to avoid this.
    fig, ax = plt.subplots(1, 1, figsize=(6.2, 6.2))
    x_values = [float(p["auc_drop"]) for p in points]
    y_values = [float(p["delta_mix"]) for p in points]
    x_max = max(x_values) if x_values else 0.2
    x_pad = max(0.01, x_max * 0.12)
    y_max = max(y_values) if y_values else 0.01
    y_pad = max(0.0004, y_max * 0.18)
    y_min = 0.0 if model_label == "LR" else min(-0.0015, min(y_values) if y_values else 0.0)
    y_max_plot = y_max + y_pad
    x_max_plot = x_max + x_pad

    setting_points = [p for p in points if p["setting"] == setting]
    for subset in ["Header", "Imports"]:
        subset_points = [p for p in setting_points if p["subset"] == subset]
        for point in subset_points:
            ax.scatter(
                float(point["auc_drop"]),
                float(point["delta_mix"]),
                s=STRENGTH_LABELS[str(point["strength"])],
                color=SUBSET_COLORS[str(point["subset"])],
                marker="o",
                alpha=0.9,
                edgecolors="white",
                linewidths=0.9,
                zorder=3,
            )

    style_axis(ax)
    ax.set_xlim(0.0, x_max_plot)
    ax.set_xticks(np.linspace(0.0, x_max_plot, 3))
    ax.set_ylim(y_min, y_max_plot)
    ax.set_ylabel("Change in Local\n" + r"Class Mixing ($\Delta$Mix@10)", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_xlabel("Decision-Level\nAUC Degradation", fontsize=AXIS_LABEL_FONT_SIZE, labelpad=8)

    subset_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=SUBSET_COLORS[name], markersize=10, label=name)
        for name in ["Header", "Imports"]
    ]
    strength_handles = [
        plt.scatter([], [], s=size, color="#777777", alpha=0.75, label=label)
        for label, size in STRENGTH_LABELS.items()
    ]

    # Fix the axes' position first so the legends below can be centered on
    # the plot area itself, not on the full figure canvas (the y-axis label
    # eats into the left margin, so figure-fraction x=0.5 sits left of the
    # plot's true center).
    # Matches the RF sibling panel's (plot_rf_wild_b_masking_figures.py:
    # plot_structural_response) rect/pad/anchor values exactly, so the two
    # legend rows sit at the same visual spacing below the plot across all
    # three models.
    fig.tight_layout(rect=(0.10, 0.18, 1.02, 0.98))
    pos = ax.get_position()
    axes_center_x = (pos.x0 + pos.x1) / 2

    legend_subsets = fig.legend(
        handles=subset_handles,
        labels=[h.get_label() for h in subset_handles],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(axes_center_x, 0.06),
        fontsize=LEGEND_FONT_SIZE,
        columnspacing=1.8,
        handletextpad=0.6,
    )
    fig.add_artist(legend_subsets)

    fig.legend(
        handles=strength_handles,
        labels=[h.get_label() for h in strength_handles],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(axes_center_x, -0.01),
        fontsize=LEGEND_FONT_SIZE,
        columnspacing=1.5,
        handletextpad=0.8,
    )

    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Figure 6 structural response maps and their Appendix Figure D.1 Wild (U)/Unpacked (B) counterparts.")
    parser.add_argument("--lr-csv", type=Path, required=True)
    parser.add_argument("--lgbm-baseline-csv", type=Path, required=True)
    parser.add_argument("--lgbm-controlled-csv", type=Path, required=True)
    parser.add_argument("--lgbm-robustness-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def output_name(setting: str, model_slug: str) -> str:
    # Matches the manuscript's own two naming conventions: Wild (B) is the
    # main-text Figure 6 panel ("structural_response_map_masking_wild_b_{model}.pdf");
    # Wild (U) and Unpacked (B) are the Appendix Figure D.1 companions, named
    # "app_structural_response_map_{model}_{slug}.pdf" to match the appendix's
    # own naming convention.
    if setting == "Wild (B)":
        return f"structural_response_map_masking_wild_b_{model_slug}.pdf"
    return f"app_structural_response_map_{model_slug}_{SETTING_SLUGS[setting]}.pdf"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lr_points = load_lr_points(args.lr_csv)
    lgbm_points = load_lgbm_points(
        args.lgbm_baseline_csv,
        args.lgbm_controlled_csv,
        args.lgbm_robustness_csv,
    )
    for setting in SETTINGS:
        plot_single_view(
            lr_points,
            args.output_dir / output_name(setting, "lr"),
            "LR",
            setting,
        )
        plot_single_view(
            lgbm_points,
            args.output_dir / output_name(setting, "lightgbm"),
            "LightGBM",
            setting,
        )
    print(f"Saved Figure 6 (Wild (B)) and Appendix Figure D.1 (Wild (U), Unpacked (B)) structural response maps to {args.output_dir}")


if __name__ == "__main__":
    main()
