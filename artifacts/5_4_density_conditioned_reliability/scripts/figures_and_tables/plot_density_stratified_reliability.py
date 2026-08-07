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
import numpy as np


DENSITY_ORDER = ["high_density", "mid_density", "low_density"]
DENSITY_LABELS = {
    "high_density": "High",
    "mid_density": "Mid",
    "low_density": "Low",
}
FEATURE_ORDER = ["all", "header", "section", "imports", "strings"]
FEATURE_LABELS = {
    "all": "All",
    "header": "Header",
    "section": "Section",
    "imports": "Imports",
    "strings": "Strings",
}
METRIC_COLORS = {
    "auc": "#1f4e79",
    "mix_at_10": "#c75b12",
}
DENSITY_COLORS = {
    "high_density": "#2F6BFF",
    "mid_density": "#C44536",
    "low_density": "#6C757D",
}

TITLE_FONT_SIZE = 24
AXIS_LABEL_FONT_SIZE = 19
TICK_FONT_SIZE = 19
LEGEND_FONT_SIZE = 17


def style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["grid.alpha"] = 0.45
    plt.rcParams["grid.linewidth"] = 0.9


def style_axis(ax) -> None:
    ax.tick_params(axis="x", labelsize=TICK_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
    ax.xaxis.label.set_size(AXIS_LABEL_FONT_SIZE)
    ax.yaxis.label.set_size(AXIS_LABEL_FONT_SIZE)
    ax.xaxis.grid(True, linestyle="-", linewidth=0.9, alpha=0.4)
    ax.yaxis.grid(True, linestyle="-", linewidth=0.9, alpha=0.5)
    ax.set_axisbelow(True)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_metric_summary(path: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    rows = read_csv(path)
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for row in rows:
        density = row["density_bin"]
        feature = row["feature_group"]
        out.setdefault(density, {})[feature] = {
            "auc_mean": float(row["auc_mean"]) if row["auc_mean"] else float("nan"),
            "mix_at_10_mean": float(row["mix_at_10_mean"]) if row["mix_at_10_mean"] else float("nan"),
            "js_divergence_mean": float(row["js_divergence_mean"]) if row["js_divergence_mean"] else float("nan"),
        }
    return out


def load_ranking_summary(path: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    rows = read_csv(path)
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for row in rows:
        density = row["density_bin"]
        feature = row["feature_group"]
        out.setdefault(density, {})[feature] = {
            "auc_rank_mean": float(row["auc_rank_mean"]) if row["auc_rank_mean"] else float("nan"),
            "mix_at_10_rank_mean": float(row["mix_at_10_rank_mean"]) if row["mix_at_10_rank_mean"] else float("nan"),
            "js_divergence_rank_mean": float(row["js_divergence_rank_mean"]) if row["js_divergence_rank_mean"] else float("nan"),
            "auc_vs_mix_conflicts_mean": float(row["auc_vs_mix_conflicts_mean"]) if row["auc_vs_mix_conflicts_mean"] else float("nan"),
            "auc_vs_js_conflicts_mean": float(row["auc_vs_js_conflicts_mean"]) if row["auc_vs_js_conflicts_mean"] else float("nan"),
        }
    return out


def plot_metric_lines(metric_summary: Dict[str, Dict[str, Dict[str, float]]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), sharex=True)
    x = np.arange(len(FEATURE_ORDER))

    panels = [
        ("auc_mean", "AUC", axes[0], "AUC (higher is better)"),
        ("mix_at_10_mean", "Mix@10", axes[1], "Mix@10 (lower is better)"),
    ]

    for metric_key, _title, ax, subtitle in panels:
        for density in DENSITY_ORDER:
            y = [
                metric_summary.get(density, {}).get(feature, {}).get(metric_key, float("nan"))
                for feature in FEATURE_ORDER
            ]
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=2.6,
                markersize=7,
                label=DENSITY_LABELS[density],
                color=DENSITY_COLORS[density],
            )
        ax.set_xticks(x)
        ax.set_xticklabels([FEATURE_LABELS[feature] for feature in FEATURE_ORDER], rotation=0)
        ax.set_xlabel("Feature Subset")
        ax.set_ylabel(subtitle)
        style_axis(ax)

    handles, labels = axes[1].get_legend_handles_labels()
    if axes[1].legend_ is not None:
        axes[1].legend_.remove()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        fontsize=LEGEND_FONT_SIZE,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    stem = output_dir / "figure_density_reliability_metric_lines"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    plot_single_metric_panel(
        metric_summary=metric_summary,
        metric_key="auc_mean",
        y_label="AUC (higher is better)",
        output_stem=output_dir / "figure_density_reliability_auc",
    )
    plot_single_metric_panel(
        metric_summary=metric_summary,
        metric_key="mix_at_10_mean",
        y_label="Mix@10 (lower is better)",
        output_stem=output_dir / "figure_density_reliability_mix_at_10",
    )


def plot_single_metric_panel(
    metric_summary: Dict[str, Dict[str, Dict[str, float]]],
    metric_key: str,
    y_label: str,
    output_stem: Path,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6.8, 5.4))
    x = np.arange(len(FEATURE_ORDER))

    for density in DENSITY_ORDER:
        y = [
            metric_summary.get(density, {}).get(feature, {}).get(metric_key, float("nan"))
            for feature in FEATURE_ORDER
        ]
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.6,
            markersize=7,
            label=DENSITY_LABELS[density],
            color=DENSITY_COLORS[density],
        )

    ax.set_xticks(x)
    ax.set_xticklabels([FEATURE_LABELS[feature] for feature in FEATURE_ORDER], rotation=0)
    ax.set_xlabel("Feature Subset")
    ax.set_ylabel(y_label)
    style_axis(ax)

    handles, labels = ax.get_legend_handles_labels()
    if ax.legend_ is not None:
        ax.legend_.remove()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        fontsize=LEGEND_FONT_SIZE,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_rankings(ranking_summary: Dict[str, Dict[str, Dict[str, float]]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.6), sharey=True)
    x = np.arange(len(FEATURE_ORDER))

    panels = [
        ("auc_rank_mean", "AUC Rank"),
        ("mix_at_10_rank_mean", "Mix@10 Rank"),
        ("js_divergence_rank_mean", "JS Rank"),
    ]

    for metric_key, title, ax in zip([p[0] for p in panels], [p[1] for p in panels], axes):
        for density in DENSITY_ORDER:
            y = [
                ranking_summary.get(density, {}).get(feature, {}).get(metric_key, float("nan"))
                for feature in FEATURE_ORDER
            ]
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=2.4,
                markersize=7,
                label=DENSITY_LABELS[density],
                color=DENSITY_COLORS[density],
            )
        ax.set_title(title, fontsize=TITLE_FONT_SIZE)
        ax.set_xticks(x)
        ax.set_xticklabels([FEATURE_LABELS[feature] for feature in FEATURE_ORDER], rotation=0)
        ax.set_xlabel("Feature Subset")
        ax.invert_yaxis()
        style_axis(ax)

    axes[0].set_ylabel("Rank (1 = best)")
    handles, labels = axes[-1].get_legend_handles_labels()
    if axes[-1].legend_ is not None:
        axes[-1].legend_.remove()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        fontsize=LEGEND_FONT_SIZE,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    stem = output_dir / "figure_density_reliability_rankings"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_conflict_table(ranking_summary: Dict[str, Dict[str, Dict[str, float]]], output_dir: Path) -> None:
    rows: List[Dict[str, object]] = []
    for density in DENSITY_ORDER:
        features = ranking_summary.get(density, {})
        if not features:
            continue
        representative = next(iter(features.values()))
        rows.append(
            {
                "density_bin": DENSITY_LABELS[density],
                "auc_vs_mix_conflicts_mean": representative.get("auc_vs_mix_conflicts_mean", float("nan")),
                "auc_vs_js_conflicts_mean": representative.get("auc_vs_js_conflicts_mean", float("nan")),
            }
        )
    if not rows:
        return
    out_path = output_dir / "table_density_reliability_conflicts.csv"
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot RQ3 Block 1 density-stratified reliability figures.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--name-prefix",
        type=str,
        default="",
        help="Optional filename prefix, e.g. 'block1_lr_controlled_'.",
    )
    parser.add_argument(
        "--skip-conflict-table",
        action="store_true",
        help="Skip writing the CSV conflict table. Useful for manuscript-figure-only directories.",
    )
    parser.add_argument(
        "--single-panels-only",
        action="store_true",
        help="Generate only the standalone AUC and Mix@10 PDFs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_summary = load_metric_summary(args.input_dir / "aggregate_metric_rows.csv")
    ranking_summary = load_ranking_summary(args.input_dir / "aggregate_ranking_rows.csv")
    if args.single_panels_only:
        plot_single_metric_panel(
            metric_summary=metric_summary,
            metric_key="auc_mean",
            y_label="AUC (higher is better)",
            output_stem=args.output_dir / f"{args.name_prefix}figure_density_reliability_auc",
        )
        plot_single_metric_panel(
            metric_summary=metric_summary,
            metric_key="mix_at_10_mean",
            y_label="Mix@10 (lower is better)",
            output_stem=args.output_dir / f"{args.name_prefix}figure_density_reliability_mix_at_10",
        )
        return
    if args.name_prefix:
        global_prefix = args.name_prefix
        original_metric = plot_metric_lines
        original_rank = plot_rankings
        original_table = write_conflict_table

        def plot_metric_lines_with_prefix(metric_summary, output_dir):
            plot_metric_lines(metric_summary, output_dir)
            for stem in [
                "figure_density_reliability_metric_lines.pdf",
                "figure_density_reliability_auc.pdf",
                "figure_density_reliability_mix_at_10.pdf",
            ]:
                source = output_dir / stem
                if source.exists():
                    source.rename(output_dir / f"{global_prefix}{stem}")

        def plot_rankings_with_prefix(ranking_summary, output_dir):
            plot_rankings(ranking_summary, output_dir)
            source = output_dir / "figure_density_reliability_rankings.pdf"
            if source.exists():
                source.rename(output_dir / f"{global_prefix}figure_density_reliability_rankings.pdf")

        def write_conflict_table_with_prefix(ranking_summary, output_dir):
            write_conflict_table(ranking_summary, output_dir)
            source = output_dir / "table_density_reliability_conflicts.csv"
            if source.exists():
                source.rename(output_dir / f"{global_prefix}table_density_reliability_conflicts.csv")

        plot_metric_lines_with_prefix(metric_summary, args.output_dir)
        plot_rankings_with_prefix(ranking_summary, args.output_dir)
        if not args.skip_conflict_table:
            write_conflict_table_with_prefix(ranking_summary, args.output_dir)
    else:
        plot_metric_lines(metric_summary, args.output_dir)
        plot_rankings(ranking_summary, args.output_dir)
        if not args.skip_conflict_table:
            write_conflict_table(ranking_summary, args.output_dir)


if __name__ == "__main__":
    main()
