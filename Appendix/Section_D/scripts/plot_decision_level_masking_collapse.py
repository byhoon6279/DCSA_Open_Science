#!/usr/bin/env python3
"""
Build the Appendix Figure D.1 decision-level collapse panels for Wild (U) and
Unpacked (B) (2 models x 2 settings = 4 PDFs).

This is the pre-journal-restyling version of the decision-level collapse
plotter. The Wild (B) main-text figure was later restyled into what is now
`plot_decision_level_masking_collapse_v2.py` in this package, but Wild (U)
and Unpacked (B) were never restyled - this script remains their only
generator, so it is kept for those two settings only (Wild (B) output is
intentionally not produced here to avoid a stale-styled duplicate of what
`_v2.py` already produces).

Each figure:
- 2 panels: Header / Imports
- x-axis: masking ratio
- y-axis: Normalized AUC degradation
- color: Wild (U), Wild (B), Unpacked (B)
- linestyle: important vs random

Inputs are per-setting aggregate_results.csv files. Values are averaged over
test weeks for each (feature_group, perturbation_type, strength) combination.
Normalized degradation is computed as:
    AUC drop / (original AUC - 0.5)
so the drop is normalized by the available margin above chance.
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
SETTING_COLORS = {
    "Wild (U)": "#6C757D",
    "Wild (B)": "#C44536",
    "Unpacked (B)": "#2F6BFF",
}
SETTING_MARKERS = {
    "Wild (U)": "o",
    "Wild (B)": "s",
    "Unpacked (B)": "^",
}
SETTING_OFFSETS = {
    "Wild (U)": -0.08,
    "Wild (B)": 0.0,
    "Unpacked (B)": 0.08,
}
SUBSETS = ["Header", "Imports"]
FEATURE_TO_LABEL = {
    "header": "Header",
    "imports": "Imports",
}
STRENGTHS = ["1%", "5%", "10%"]
STRENGTH_FLOATS = [0.01, 0.05, 0.10]
SUBSET_COLORS = {
    "Header": "#2F6BFF",
    "Imports": "#C44536",
}
SUBSET_MARKERS = {
    "Header": "s",
    "Imports": "o",
}
SUBSET_OFFSETS = {
    "Header": -0.05,
    "Imports": 0.05,
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
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def aggregate_setting_rows(rows: Iterable[Dict[str, str]], setting_label: str) -> List[Dict[str, str]]:
    grouped: Dict[tuple, List[float]] = defaultdict(list)
    for row in rows:
        feature_group = row["feature_group"]
        perturbation_type = row["perturbation_type"]
        strength = float(row["strength"])
        if feature_group not in FEATURE_TO_LABEL:
            continue
        if perturbation_type not in {"important", "random"}:
            continue
        if strength not in STRENGTH_FLOATS:
            continue
        auc_drop = max(0.0, -float(row["delta_auc_mean"]))
        original_auc = float(row["auc_original"])
        available_margin = max(original_auc - 0.5, 1e-8)
        relative_drop = auc_drop / available_margin
        grouped[(feature_group, perturbation_type, strength)].append(relative_drop)

    output_rows: List[Dict[str, str]] = []
    for feature_group, subset_label in FEATURE_TO_LABEL.items():
        for strength_float, strength_label in zip(STRENGTH_FLOATS, STRENGTHS):
            imp_values = grouped[(feature_group, "important", strength_float)]
            rnd_values = grouped[(feature_group, "random", strength_float)]
            output_rows.append(
                {
                    "Setting": setting_label,
                    "Subset": subset_label,
                    "Strength": strength_label,
                    "Normalized AUC Degradation (Imp)": f"{sum(imp_values) / len(imp_values):.3f}",
                    "Normalized AUC Degradation (Rand)": f"{sum(rnd_values) / len(rnd_values):.3f}",
                }
            )
    return output_rows


def build_lookup(rows: List[Dict[str, str]]) -> Dict[tuple, Dict[str, str]]:
    return {(row["Setting"], row["Subset"], row["Strength"]): row for row in rows}


def plot_row(
    axes: List[plt.Axes],
    lookup: Dict[tuple, Dict[str, str]],
    settings: List[str],
) -> tuple[float, float]:
    x = list(range(len(STRENGTHS)))
    ymin = None
    ymax = None
    for subset in SUBSETS:
        for setting in settings:
            for strength in STRENGTHS:
                row = lookup[(setting, subset, strength)]
                for col in ("Normalized AUC Degradation (Imp)", "Normalized AUC Degradation (Rand)"):
                    value = float(row[col])
                    ymin = value if ymin is None else min(ymin, value)
                    ymax = value if ymax is None else max(ymax, value)

    assert ymin is not None and ymax is not None
    margin = max((ymax - ymin) * 0.15, 0.01)

    for ax, subset in zip(axes, SUBSETS):
        for setting in settings:
            color = SETTING_COLORS[setting]
            marker = SETTING_MARKERS[setting]
            offset = SETTING_OFFSETS[setting]
            important_values = [
                float(lookup[(setting, subset, strength)]["Normalized AUC Degradation (Imp)"])
                for strength in STRENGTHS
            ]
            random_values = [
                float(lookup[(setting, subset, strength)]["Normalized AUC Degradation (Rand)"])
                for strength in STRENGTHS
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
                linewidth=1.8,
                linestyle="--",
                alpha=0.35,
                zorder=2,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(STRENGTHS)
        ax.set_xlabel("Masking Ratio")
        style_axis(ax)

    return max(0.0, ymin - margin), ymax + margin


def compute_setting_ylim(rows: List[Dict[str, str]], setting: str) -> tuple[float, float]:
    lookup = build_lookup(rows)
    values: List[float] = []
    for subset in SUBSETS:
        for strength in STRENGTHS:
            row = lookup[(setting, subset, strength)]
            values.append(float(row["Normalized AUC Degradation (Imp)"]))
            values.append(float(row["Normalized AUC Degradation (Rand)"]))
    ymax = max(values) if values else 0.1
    y_bottom = 0.0
    y_top = math.ceil((ymax + 0.01) * 20) / 20
    return y_bottom, max(y_top, 0.05)


def plot_single_model_setting(
    rows: List[Dict[str, str]],
    output_path: Path,
    model_label: str,
    setting: str,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(5.8, 4.8))
    lookup = build_lookup(rows)
    x = list(range(len(STRENGTHS)))
    y_bottom, y_top = compute_setting_ylim(rows, setting)

    for subset in SUBSETS:
        color = SUBSET_COLORS[subset]
        marker = SUBSET_MARKERS[subset]
        offset = SUBSET_OFFSETS[subset]
        important_values = [
            float(lookup[(setting, subset, strength)]["Normalized AUC Degradation (Imp)"])
            for strength in STRENGTHS
        ]
        random_values = [
            float(lookup[(setting, subset, strength)]["Normalized AUC Degradation (Rand)"])
            for strength in STRENGTHS
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
            linewidth=1.8,
            linestyle="--",
            alpha=0.35,
            zorder=2,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(STRENGTHS)
    ax.set_xlabel("Masking Ratio")
    ax.set_ylabel("Normalized AUC Degradation")
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
    parser = argparse.ArgumentParser(description="Build the LR and LightGBM decision-level masking panels for Appendix Figure D.1.")
    parser.add_argument("--lr-baseline-csv", type=Path, required=True)
    parser.add_argument("--lr-controlled-csv", type=Path, required=True)
    parser.add_argument("--lr-robustness-csv", type=Path, required=True)
    parser.add_argument("--lgbm-baseline-csv", type=Path, required=True)
    parser.add_argument("--lgbm-controlled-csv", type=Path, required=True)
    parser.add_argument("--lgbm-robustness-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lr_rows: List[Dict[str, str]] = []
    lr_rows.extend(aggregate_setting_rows(read_rows(args.lr_baseline_csv), "Wild (U)"))
    lr_rows.extend(aggregate_setting_rows(read_rows(args.lr_controlled_csv), "Wild (B)"))
    lr_rows.extend(aggregate_setting_rows(read_rows(args.lr_robustness_csv), "Unpacked (B)"))

    lgbm_rows: List[Dict[str, str]] = []
    lgbm_rows.extend(aggregate_setting_rows(read_rows(args.lgbm_baseline_csv), "Wild (U)"))
    lgbm_rows.extend(aggregate_setting_rows(read_rows(args.lgbm_controlled_csv), "Wild (B)"))
    lgbm_rows.extend(aggregate_setting_rows(read_rows(args.lgbm_robustness_csv), "Unpacked (B)"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for setting, slug in [("Wild (U)", "wild_u"), ("Unpacked (B)", "unpacked_b")]:
        plot_single_model_setting(
            lr_rows,
            args.output_dir / f"decision_level_masking_collapse_{slug}_lr.pdf",
            "LR",
            setting,
        )
        plot_single_model_setting(
            lgbm_rows,
            args.output_dir / f"decision_level_masking_collapse_{slug}_lightgbm.pdf",
            "LightGBM",
            setting,
        )
    print(f"Saved Appendix Figure D.1 decision-level collapse panels (Wild (U), Unpacked (B)) to {args.output_dir}")


if __name__ == "__main__":
    main()
