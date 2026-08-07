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
import seaborn as sns


DEFAULT_INPUT = (
    Path(__file__).resolve().parents[3]
    / "results"
    / "LR"
    / "win32_all_train_all_test_balanced_test"
    / "balanced_test_aggregate_summary.csv"
)
FEATURE_ORDER = ["all", "header", "section", "strings", "imports"]
FEATURE_COLORS = {
    "all": "#1f4e79",
    "header": "#2a7f62",
    "section": "#c75b12",
    "strings": "#8b1e3f",
    "imports": "#5c6b73",
}


def style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["axes.labelsize"] = 19
    plt.rcParams["xtick.labelsize"] = 16
    plt.rcParams["ytick.labelsize"] = 16


def load_summary(path: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    table: Dict[str, Dict[str, Dict[str, float]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            feature = row["feature_group"]
            metric = row["metric"]
            table.setdefault(feature, {})[metric] = {
                "scope": row["scope"],
                "mean": float(row["mean"]),
                "std": float(row["std"]),
                "min": float(row["min"]),
                "max": float(row["max"]),
            }
    return table


def default_output_dir(input_path: Path) -> Path:
    return input_path.parent


def normalize(values: List[float], invert: bool = False) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < 1e-12:
        out = np.ones_like(arr)
    else:
        out = (arr - lo) / (hi - lo)
    if invert:
        out = 1.0 - out
    return out


def save_auc_js_scatter(summary: Dict[str, Dict[str, Dict[str, float]]], out_path: Path, png_also: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 5.8))

    mix_values = [summary[group]["mix_at_k"]["mean"] for group in FEATURE_ORDER]
    mix_sizes = 400 + 1800 * normalize(mix_values)
    auc_values = [summary[group]["auc"]["mean"] for group in FEATURE_ORDER]
    js_values = [summary[group]["js_divergence"]["mean"] for group in FEATURE_ORDER]

    for idx, group in enumerate(FEATURE_ORDER):
        auc = summary[group]["auc"]["mean"]
        js = summary[group]["js_divergence"]["mean"]
        size = float(mix_sizes[idx])
        ax.scatter(
            js,
            auc,
            s=size,
            color=FEATURE_COLORS[group],
            alpha=0.85,
            edgecolor="white",
            linewidth=1.2,
        )
        if group == "strings":
            label = "strings\n(high AUC, near-zero JS)"
            offset = (10, 10)
            weight = "bold"
        elif group == "imports":
            label = group
            offset = (16, 12)
            weight = "normal"
        else:
            label = group
            offset = (8, 6)
            weight = "normal"
        ax.annotate(
            label,
            (js, auc),
            xytext=offset,
            textcoords="offset points",
            fontsize=14,
            fontweight=weight,
        )

    ax.set_xlabel("JS Divergence")
    ax.set_ylabel("AUC")
    x_min = min(-0.01, min(js_values) - 0.005)
    x_max = max(0.135, max(js_values) + 0.015)
    y_min = min(0.89, min(auc_values) - 0.015)
    y_max = min(1.0, max(0.985, max(auc_values) + 0.012))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, axis="y", linestyle="-", linewidth=0.9, alpha=0.5)
    ax.grid(True, axis="x", linestyle="-", linewidth=0.9, alpha=0.4)
    ax.text(
        0.98,
        0.03,
        "Bubble size: Mix@10\n(larger = higher local mixing)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=13,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#d0d0d0", "alpha": 0.9},
    )
    fig.tight_layout()
    if png_also:
        fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_rank_dumbbell(summary: Dict[str, Dict[str, Dict[str, float]]], out_path: Path, png_also: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    y = np.arange(len(FEATURE_ORDER))

    for idx, group in enumerate(FEATURE_ORDER):
        auc_rank = summary[group]["auc_rank"]["mean"]
        js_rank = summary[group]["js_divergence_rank"]["mean"]
        ax.plot([auc_rank, js_rank], [idx, idx], color="#b7b7b7", linewidth=2.0, zorder=1)
        ax.scatter(auc_rank, idx, s=90, color="#1f4e79", zorder=2, label="AUC Rank" if idx == 0 else None)
        ax.scatter(js_rank, idx, s=90, color="#8b1e3f", zorder=2, label="JS Rank" if idx == 0 else None)

    ax.set_yticks(y)
    ax.set_yticklabels([group.title() for group in FEATURE_ORDER])
    ax.set_xlabel("Mean Rank")
    ax.set_xlim(0.7, 5.3)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.invert_yaxis()
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.14))
    fig.tight_layout()
    if png_also:
        fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot RQ1 representation-level mismatch figures from aggregate_summary.csv.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--auc-js-stem",
        type=str,
        default="representation_mismatch_auc_vs_js",
        help="Filename stem for the AUC-vs-JS scatter figure.",
    )
    parser.add_argument(
        "--rank-stem",
        type=str,
        default="representation_mismatch_rank_dumbbell",
        help="Filename stem for the rank dumbbell figure.",
    )
    parser.add_argument(
        "--png-also",
        action="store_true",
        help="Also export PNG files alongside PDFs.",
    )
    args = parser.parse_args()

    style()
    summary = load_summary(args.input)
    output_dir = args.output_dir or default_output_dir(args.input)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_auc_js_scatter(summary, output_dir / args.auc_js_stem, png_also=args.png_also)
    save_rank_dumbbell(summary, output_dir / args.rank_stem, png_also=args.png_also)

    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
