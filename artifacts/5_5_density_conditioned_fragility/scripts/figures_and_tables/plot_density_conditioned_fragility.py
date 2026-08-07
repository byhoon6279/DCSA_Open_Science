#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42  # embed fonts as TrueType
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

TITLE_FONT_SIZE = 24
AXIS_LABEL_FONT_SIZE = 19
TICK_FONT_SIZE = 19
LEGEND_FONT_SIZE = 17

DENSITY_ORDER = ["high_density", "mid_density", "low_density"]
DENSITY_LABELS = {
    "high_density": "High",
    "mid_density": "Mid",
    "low_density": "Low",
}
DENSITY_COLORS = {
    "high_density": "#2F6BFF",
    "mid_density": "#C44536",
    "low_density": "#6C757D",
}
DENSITY_MARKERS = {
    "high_density": "o",
    "mid_density": "s",
    "low_density": "^",
}
DENSITY_OFFSETS = {
    "high_density": -0.08,
    "mid_density": 0.0,
    "low_density": 0.08,
}
SUBSETS = ["header", "imports"]
SUBSET_LABELS = {
    "header": "Header",
    "imports": "Imports",
}
PERTURBATION_ORDER = ["important", "random"]
STRENGTH_VALUES = [0.01, 0.05, 0.1]
STRENGTH_LABELS = {
    0.01: "1%",
    0.05: "5%",
    0.1: "10%",
}


def style_axis(ax) -> None:
    ax.tick_params(axis="x", labelsize=TICK_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
    ax.xaxis.label.set_size(AXIS_LABEL_FONT_SIZE)
    ax.yaxis.label.set_size(AXIS_LABEL_FONT_SIZE)
    ax.xaxis.grid(True, linestyle="-", linewidth=0.9, alpha=0.4)
    ax.yaxis.grid(True, linestyle="-", linewidth=0.9, alpha=0.5)
    ax.set_axisbelow(True)


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_lookup(rows: List[Dict[str, str]]) -> Dict[tuple, Dict[str, str]]:
    lookup: Dict[tuple, Dict[str, str]] = {}
    for row in rows:
        key = (
            row["feature_group"],
            row["density_bin"],
            row["perturbation_type"],
            round(float(row["strength"]), 5),
        )
        lookup[key] = row
    return lookup


def metric_bounds(
    rows: List[Dict[str, str]],
    important_col: str,
    random_col: str,
) -> tuple[float, float]:
    ymin = None
    ymax = None
    for row in rows:
        for col in (important_col, random_col):
            val = float(row[col])
            ymin = val if ymin is None else min(ymin, val)
            ymax = val if ymax is None else max(ymax, val)
    assert ymin is not None and ymax is not None
    return ymin, ymax


def plot_metric_figure(
    rows: List[Dict[str, str]],
    important_col: str,
    random_col: str,
    ylabel: str,
    output_stem: str,
    output_dir: Path,
    name_prefix: str,
) -> None:
    lookup = build_lookup(rows)
    x = list(range(len(STRENGTH_VALUES)))

    ymin, ymax = metric_bounds(rows, important_col, random_col)
    margin = max((ymax - ymin) * 0.15, 0.001)

    for subset in SUBSETS:
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 5.2), sharex=True, sharey=True)
        for density in DENSITY_ORDER:
            color = DENSITY_COLORS[density]
            marker = DENSITY_MARKERS[density]
            offset = DENSITY_OFFSETS[density]

            important_values = [
                float(lookup[(subset, density, "important", strength)][important_col])
                for strength in STRENGTH_VALUES
            ]
            random_values = [
                float(lookup[(subset, density, "random", strength)][random_col])
                for strength in STRENGTH_VALUES
            ]
            x_positions = [value + offset for value in x]

            ax.plot(
                x_positions,
                important_values,
                color=color,
                marker=marker,
                linewidth=2.8,
                linestyle="-",
                zorder=3,
            )
            ax.plot(
                x_positions,
                random_values,
                color=color,
                marker=marker,
                linewidth=1.9,
                linestyle="--",
                alpha=0.35,
                zorder=2,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([STRENGTH_LABELS[v] for v in STRENGTH_VALUES])
        ax.set_xlabel("Masking Ratio")
        style_axis(ax)
        ax.set_ylim(ymin - margin, ymax + margin)
        ax.set_ylabel(ylabel)

        density_handles = [
            Line2D(
                [0],
                [0],
                color=DENSITY_COLORS[density],
                lw=2.8,
                marker=DENSITY_MARKERS[density],
                label=DENSITY_LABELS[density],
            )
            for density in DENSITY_ORDER
        ]
        style_handles = [
            Line2D([0], [0], color="black", lw=2.8, linestyle="-", label="Important"),
            Line2D([0], [0], color="black", lw=1.9, linestyle="--", alpha=0.5, label="Random"),
        ]

        density_legend = fig.legend(
            density_handles,
            [handle.get_label() for handle in density_handles],
            loc="lower center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 0.01),
            fontsize=LEGEND_FONT_SIZE,
        )
        style_legend = fig.legend(
            style_handles,
            [handle.get_label() for handle in style_handles],
            loc="lower center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, -0.045),
            fontsize=LEGEND_FONT_SIZE,
        )
        fig.add_artist(density_legend)
        fig.add_artist(style_legend)
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        stem = f"{name_prefix}_{output_stem}_{subset}" if name_prefix else f"{output_stem}_{subset}"
        fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build manuscript-oriented RQ3 Block 2 figures.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name-prefix", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_metric_figure(
        rows,
        important_col="flip_rate_mean_mean",
        random_col="flip_rate_mean_mean",
        ylabel="Flip Rate",
        output_stem="figure_density_fragility_flip_rate",
        output_dir=args.output_dir,
        name_prefix=args.name_prefix,
    )

    plot_metric_figure(
        rows,
        important_col="auc_drop_mean_mean",
        random_col="auc_drop_mean_mean",
        ylabel="AUC Degradation",
        output_stem="figure_density_fragility_auc_drop",
        output_dir=args.output_dir,
        name_prefix=args.name_prefix,
    )


if __name__ == "__main__":
    main()
