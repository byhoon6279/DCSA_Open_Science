"""
Generate comparison figures for taxonomy-aligned family baselines.

This script compares:
1. In-the-wild family-aligned baseline
2. Unpacked-controlled family-aligned baseline

Figures:
1. same_family_rate@k by feature subset (original vs unpacked)
2. macro-F1 vs same_family_rate@k scatter (two panels)
3. family_silhouette by feature subset (original vs unpacked)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["axes.labelsize"] = 19
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16


FEATURE_ORDER = ["all", "header", "section", "strings", "imports"]
LABEL_MAP = {
    "all": "All",
    "header": "Header",
    "section": "Section",
    "strings": "Strings",
    "imports": "Imports",
}
POINT_COLOR_MAP = {
    "all": "#264653",
    "header": "#2A9D8F",
    "section": "#E9C46A",
    "strings": "#F4A261",
    "imports": "#E76F51",
}
LINE_COLOR_MAP = {
    "original": "#2F6BFF",
    "unpacked": "#D94841",
}
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "results" / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-summary-csv", type=Path, required=True)
    parser.add_argument("--unpacked-summary-csv", type=Path, required=True)
    parser.add_argument("--original-by-k-csv", type=Path, required=True)
    parser.add_argument("--unpacked-by-k-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--purity-k", type=int, default=10)
    parser.add_argument("--scatter-xmin", type=float)
    parser.add_argument("--scatter-xmax", type=float)
    parser.add_argument("--scatter-ymin", type=float)
    parser.add_argument("--scatter-ymax", type=float)
    parser.add_argument("--annotation-fontsize", type=int, default=14)
    parser.add_argument("--png-also", action="store_true")
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def load_summary_metrics(path: Path) -> dict[str, dict[str, float]]:
    rows = read_csv_rows(path)
    metrics: dict[str, dict[str, float]] = {feature: {} for feature in FEATURE_ORDER}
    for row in rows:
        if row["scope"] != "metric":
            continue
        feature_group = row["feature_group"]
        if feature_group not in FEATURE_ORDER:
            continue
        metrics[feature_group][row["metric"]] = float(row["mean"])
    return metrics


def load_purity_at_k(path: Path, k: int) -> dict[str, float]:
    rows = read_csv_rows(path)
    values: dict[str, float] = {}
    for row in rows:
        if int(row["k"]) != k:
            continue
        if row["scope"] != "metric" or row["metric"] != "same_family_rate_at_k":
            continue
        feature_group = row["feature_group"]
        if feature_group not in FEATURE_ORDER:
            continue
        values[feature_group] = float(row["mean"])
    missing = [feature for feature in FEATURE_ORDER if feature not in values]
    if missing:
        raise ValueError(f"Missing same_family_rate_at_k for k={k}: {missing}")
    return values


def style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.45)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_figure_1(
    output_dir: Path,
    original_purity: dict[str, float],
    unpacked_purity: dict[str, float],
    purity_k: int,
    png_also: bool = False,
) -> None:
    x = np.arange(len(FEATURE_ORDER))
    labels = [LABEL_MAP[feature] for feature in FEATURE_ORDER]

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    ax.plot(
        x,
        [original_purity[feature] for feature in FEATURE_ORDER],
        marker="o",
        linewidth=2.4,
        markersize=7,
        color=LINE_COLOR_MAP["original"],
        label="Wild (U)",
    )
    ax.plot(
        x,
        [unpacked_purity[feature] for feature in FEATURE_ORDER],
        marker="s",
        linewidth=2.4,
        markersize=7,
        color=LINE_COLOR_MAP["unpacked"],
        label="Unpacked (B)",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Feature Subset")
    ax.set_ylabel(f"Same-Family Rate@{purity_k}")
    ax.set_ylim(0.0, 1.0)
    style_axes(ax)
    ax.grid(axis="y", linestyle="-", linewidth=0.9, alpha=0.5)
    ax.grid(axis="x", linestyle="-", linewidth=0.9, alpha=0.4)
    legend = ax.legend(loc="lower right", frameon=True, fontsize=14, fancybox=True, framealpha=0.95)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#d0d0d0")
    legend.get_frame().set_linewidth(0.9)

    fig.tight_layout()
    if png_also:
        fig.savefig(output_dir / "figure1_local_family_alignment_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "figure1_local_family_alignment_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def annotate_subset_points(
    ax: plt.Axes,
    x_values: list[float],
    y_values: list[float],
    annotation_fontsize: int,
    setting_name: str,
    is_lightgbm: bool,
    is_rf: bool = False,
) -> None:
    for feature, x_value, y_value in zip(FEATURE_ORDER, x_values, y_values):
        ax.scatter(x_value, y_value, s=70, color=POINT_COLOR_MAP[feature], zorder=3)
        if is_lightgbm and setting_name == "wild" and feature == "header":
            offset = (-92, 0)
        elif is_lightgbm and setting_name == "unpacked" and feature == "header":
            offset = (0, -24)
        elif is_rf and setting_name == "wild" and feature == "header":
            # RF's Header and Section points sit almost on top of each other
            # under Wild (B); the default right-side offset collides with
            # the Section label above it, so drop Header to the lower-left instead.
            offset = (-58, -22)
        elif setting_name == "unpacked" and feature == "header":
            offset = (10, 0)
        elif setting_name == "wild" and feature == "header":
            offset = (10, 0)
        elif setting_name == "unpacked" and feature == "all":
            offset = (0, -18)
        else:
            offset = (5, 5)
        ax.annotate(
            LABEL_MAP[feature],
            (x_value, y_value),
            textcoords="offset points",
            xytext=offset,
            fontsize=annotation_fontsize,
        )


def plot_figure_2(
    output_dir: Path,
    original_summary: dict[str, dict[str, float]],
    unpacked_summary: dict[str, dict[str, float]],
    original_purity: dict[str, float],
    unpacked_purity: dict[str, float],
    purity_k: int,
    annotation_fontsize: int = 14,
    scatter_xmin: float | None = None,
    scatter_xmax: float | None = None,
    scatter_ymin: float | None = None,
    scatter_ymax: float | None = None,
    png_also: bool = False,
) -> None:
    settings = [
        ("wild", original_summary, original_purity),
        ("unpacked", unpacked_summary, unpacked_purity),
    ]
    is_lightgbm = "lightgbm" in str(output_dir).lower()
    is_rf = "rf" in str(output_dir).lower()

    if None not in (scatter_xmin, scatter_xmax, scatter_ymin, scatter_ymax):
        x_low = float(scatter_xmin)
        x_high = float(scatter_xmax)
        y_low = float(scatter_ymin)
        y_high = float(scatter_ymax)
    else:
        x_min = min(
            min(summary[feature]["family_macro_f1"] for feature in FEATURE_ORDER)
            for _, summary, _ in settings
        )
        x_max = max(
            max(summary[feature]["family_macro_f1"] for feature in FEATURE_ORDER)
            for _, summary, _ in settings
        )
        y_min = min(min(purity[feature] for feature in FEATURE_ORDER) for _, _, purity in settings)
        y_max = max(max(purity[feature] for feature in FEATURE_ORDER) for _, _, purity in settings)
        x_pad = (x_max - x_min) * 0.08 if not np.isclose(x_min, x_max) else 0.03
        y_pad = (y_max - y_min) * 0.08 if not np.isclose(y_min, y_max) else 0.03
        x_low = x_min - x_pad
        x_high = x_max + x_pad
        y_low = y_min - y_pad
        y_high = y_max + y_pad

    for setting_name, summary, purity in settings:
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        x_values = [summary[feature]["family_macro_f1"] for feature in FEATURE_ORDER]
        y_values = [purity[feature] for feature in FEATURE_ORDER]
        annotate_subset_points(
            ax,
            x_values,
            y_values,
            annotation_fontsize=annotation_fontsize,
            setting_name=setting_name,
            is_lightgbm=is_lightgbm,
            is_rf=is_rf,
        )
        ax.set_xlim(x_low, x_high)
        ax.set_ylim(y_low, y_high)
        ref_min = max(ax.get_xlim()[0], ax.get_ylim()[0])
        ref_max = min(ax.get_xlim()[1], ax.get_ylim()[1])
        if ref_max > ref_min:
            ax.plot(
                [ref_min, ref_max],
                [ref_min, ref_max],
                linestyle="--",
                linewidth=1.1,
                color="#888888",
                alpha=0.8,
                zorder=1,
            )
        style_axes(ax)
        ax.set_xlabel("Classification Performance (Macro-F1)")
        ax.set_ylabel(f"Same-Family Rate@{purity_k}")
        ax.grid(axis="y", linestyle="-", linewidth=0.9, alpha=0.5)
        ax.grid(axis="x", linestyle="-", linewidth=0.9, alpha=0.4)

        fig.tight_layout()
        if png_also:
            fig.savefig(
                output_dir / f"figure2_classification_vs_structure_scatter_{setting_name}.png",
                dpi=300,
                bbox_inches="tight",
            )
        fig.savefig(
            output_dir / f"figure2_classification_vs_structure_scatter_{setting_name}.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_figure_3(
    output_dir: Path,
    original_summary: dict[str, dict[str, float]],
    unpacked_summary: dict[str, dict[str, float]],
    png_also: bool = False,
) -> None:
    x = np.arange(len(FEATURE_ORDER))
    labels = [LABEL_MAP[feature] for feature in FEATURE_ORDER]

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    ax.plot(
        x,
        [original_summary[feature]["family_silhouette"] for feature in FEATURE_ORDER],
        marker="o",
        linewidth=2.4,
        markersize=7,
        color=LINE_COLOR_MAP["original"],
        label="In-the-wild",
    )
    ax.plot(
        x,
        [unpacked_summary[feature]["family_silhouette"] for feature in FEATURE_ORDER],
        marker="s",
        linewidth=2.4,
        markersize=7,
        color=LINE_COLOR_MAP["unpacked"],
        label="Unpacked-controlled",
    )
    ax.axhline(0.0, color="#888888", linewidth=1.0, linestyle=":")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Feature Subset")
    ax.set_ylabel("Family Silhouette")
    ax.set_title("Global Family Structure Across Feature Subsets")
    style_axes(ax)
    ax.legend(frameon=False)

    fig.tight_layout()
    if png_also:
        fig.savefig(output_dir / "figure3_global_family_structure_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "figure3_global_family_structure_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    original_summary = load_summary_metrics(args.original_summary_csv)
    unpacked_summary = load_summary_metrics(args.unpacked_summary_csv)
    original_purity = load_purity_at_k(args.original_by_k_csv, args.purity_k)
    unpacked_purity = load_purity_at_k(args.unpacked_by_k_csv, args.purity_k)

    plot_figure_1(args.output_dir, original_purity, unpacked_purity, args.purity_k, png_also=args.png_also)
    plot_figure_2(
        args.output_dir,
        original_summary,
        unpacked_summary,
        original_purity,
        unpacked_purity,
        args.purity_k,
        annotation_fontsize=args.annotation_fontsize,
        scatter_xmin=args.scatter_xmin,
        scatter_xmax=args.scatter_xmax,
        scatter_ymin=args.scatter_ymin,
        scatter_ymax=args.scatter_ymax,
        png_also=args.png_also,
    )
    plot_figure_3(args.output_dir, original_summary, unpacked_summary, png_also=args.png_also)


if __name__ == "__main__":
    main()
