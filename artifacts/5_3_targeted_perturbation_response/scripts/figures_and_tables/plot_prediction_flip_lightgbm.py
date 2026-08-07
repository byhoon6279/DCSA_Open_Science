#!/usr/bin/env python3
"""
Build single-view LightGBM manuscript figures for RQ2-3.

Outputs one PDF per evaluation setting:
- RQ2-3_prediction_level_instability_lightgbm_wild_u.pdf
- prediction_flip_rate_masking_wild_b_lightgbm.pdf
- RQ2-3_prediction_level_instability_lightgbm_unpacked_b.pdf
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42  # embed fonts as TrueType
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

TITLE_FONT_SIZE = 24
AXIS_LABEL_FONT_SIZE = 19
TICK_FONT_SIZE = 19
LEGEND_FONT_SIZE = 17

SETTINGS = ["Wild (U)", "Wild (B)", "Unpacked (B)"]
SETTING_TO_SLUG = {
    "Wild (U)": "wild_u",
    "Wild (B)": "wild_b",
    "Unpacked (B)": "unpacked_b",
}
SUBSETS = ["Header", "Imports"]
SUBSET_COLORS = {
    "Header": "#C44536",
    "Imports": "#2F6BFF",
}
SUBSET_MARKERS = {
    "Header": "s",
    "Imports": "o",
}
SUBSET_KEY_TO_LABEL = {
    "header": "Header",
    "imports": "Imports",
}
SUBSET_OFFSETS = {
    "Header": -0.05,
    "Imports": 0.05,
}
STRENGTHS = ["1%", "5%", "10%"]
STRENGTH_FLOATS = [0.01, 0.05, 0.10]


def style_axis(ax) -> None:
    ax.tick_params(axis="x", labelsize=TICK_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
    ax.xaxis.label.set_size(AXIS_LABEL_FONT_SIZE)
    ax.yaxis.label.set_size(AXIS_LABEL_FONT_SIZE)
    ax.xaxis.grid(True, linestyle="-", linewidth=0.9, alpha=0.4)
    ax.yaxis.grid(True, linestyle="-", linewidth=0.9, alpha=0.5)
    ax.set_axisbelow(True)


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def aggregate_rows(rows: Iterable[Dict[str, str]], setting_label: str) -> List[Dict[str, str]]:
    grouped: Dict[tuple, List[float]] = defaultdict(list)
    for row in rows:
        feature_group = row["feature_group"]
        perturbation_type = row["perturbation_type"]
        strength = float(row["strength"])
        if feature_group not in SUBSET_KEY_TO_LABEL:
            continue
        if perturbation_type not in {"important", "random"}:
            continue
        if strength not in STRENGTH_FLOATS:
            continue
        grouped[(feature_group, perturbation_type, strength)].append(float(row["flip_rate_mean"]))

    output_rows: List[Dict[str, str]] = []
    for feature_group, subset_label in SUBSET_KEY_TO_LABEL.items():
        for strength_float, strength_label in zip(STRENGTH_FLOATS, STRENGTHS):
            imp_values = grouped[(feature_group, "important", strength_float)]
            rnd_values = grouped[(feature_group, "random", strength_float)]
            output_rows.append(
                {
                    "Setting": setting_label,
                    "Subset": subset_label,
                    "Strength": strength_label,
                    "Flip Rate (Imp)": f"{sum(imp_values) / len(imp_values):.3f}",
                    "Flip Rate (Rand)": f"{sum(rnd_values) / len(rnd_values):.3f}",
                }
            )
    return output_rows


def build_lookup(rows: List[Dict[str, str]]) -> Dict[tuple, Dict[str, str]]:
    return {(row["Setting"], row["Subset"], row["Strength"]): row for row in rows}


def compute_ylim(rows: List[Dict[str, str]], setting: str) -> tuple[float, float]:
    lookup = build_lookup(rows)
    values: List[float] = []
    for subset in SUBSETS:
        for strength in STRENGTHS:
            row = lookup[(setting, subset, strength)]
            values.append(float(row["Flip Rate (Imp)"]))
            values.append(float(row["Flip Rate (Rand)"]))
    ymax = max(values) if values else 0.1
    y_bottom = 0.0
    y_top = math.ceil((ymax + 0.01) * 20) / 20
    return y_bottom, max(y_top, 0.05)


def plot_single_view(rows: List[Dict[str, str]], output_path: Path, setting: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(5.8, 4.8))
    lookup = build_lookup(rows)
    x = list(range(len(STRENGTHS)))
    y_bottom, y_top = compute_ylim(rows, setting)

    for subset in SUBSETS:
        color = SUBSET_COLORS[subset]
        marker = SUBSET_MARKERS[subset]
        offset = SUBSET_OFFSETS[subset]
        important_values = [
            float(lookup[(setting, subset, strength)]["Flip Rate (Imp)"])
            for strength in STRENGTHS
        ]
        random_values = [
            float(lookup[(setting, subset, strength)]["Flip Rate (Rand)"])
            for strength in STRENGTHS
        ]
        x_positions = [value + offset for value in x]
        ax.plot(x_positions, important_values, color=color, marker=marker, linewidth=2.8, linestyle="-", zorder=3)
        ax.plot(
            x_positions,
            random_values,
            color=color,
            marker=marker,
            linewidth=1.8,
            linestyle="--",
            alpha=0.35,
            zorder=2,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(STRENGTHS)
    ax.set_xlabel("Masking Ratio")
    ax.set_ylabel("Flip Rate")
    ax.set_ylim(y_bottom, y_top)
    ax.yaxis.set_major_locator(MultipleLocator(0.05 if y_top <= 0.30 else 0.1))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f" if y_top <= 0.30 else "%.1f"))
    style_axis(ax)

    subset_handles = [
        Line2D([0], [0], color=SUBSET_COLORS[s], lw=2.8, marker=SUBSET_MARKERS[s], label=s)
        for s in SUBSETS
    ]
    style_handles = [
        Line2D([0], [0], color="black", lw=2.8, linestyle="-", label="Important"),
        Line2D([0], [0], color="black", lw=1.8, linestyle="--", alpha=0.35, label="Random"),
    ]
    legend_subsets = fig.legend(
        subset_handles,
        [h.get_label() for h in subset_handles],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
        fontsize=LEGEND_FONT_SIZE,
        columnspacing=1.6,
        handlelength=2.2,
    )
    fig.add_artist(legend_subsets)
    fig.legend(
        style_handles,
        [h.get_label() for h in style_handles],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.09),
        fontsize=LEGEND_FONT_SIZE,
        columnspacing=1.8,
        handlelength=2.6,
    )
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build single-view LightGBM manuscript figures for RQ2-3.")
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--controlled-csv", type=Path, required=True)
    parser.add_argument("--robustness-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: List[Dict[str, str]] = []
    rows.extend(aggregate_rows(read_rows(args.baseline_csv), "Wild (U)"))
    rows.extend(aggregate_rows(read_rows(args.controlled_csv), "Wild (B)"))
    rows.extend(aggregate_rows(read_rows(args.robustness_csv), "Unpacked (B)"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for setting in SETTINGS:
        plot_single_view(
            rows,
            args.output_dir / f"RQ2-3_prediction_level_instability_lightgbm_{SETTING_TO_SLUG[setting]}.pdf",
            setting,
        )
    print(f"Saved LightGBM manuscript figures to {args.output_dir}")


if __name__ == "__main__":
    main()
