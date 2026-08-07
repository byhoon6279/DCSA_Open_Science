#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42  # embed fonts as TrueType
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parents[2]
FIGURE_DIR = BASE_DIR / "results" / "manuscript_figures"
TABLE_DIR = BASE_DIR / "results" / "manuscript_tables"

SETTINGS = {
    "original_mixed": BASE_DIR / "results" / "experiment_runs" / "win32_all_train_all_test" / "original_aggregate_summary.csv",
    "mixed_balanced": BASE_DIR / "results" / "experiment_runs" / "win32_all_train_all_test_balanced_test" / "balanced_test_aggregate_summary.csv",
    "unpacked_balanced": BASE_DIR / "results" / "experiment_runs" / "win32_all_train_all_test_unpacked_balanced_test" / "unpacked_balanced_aggregate_summary.csv",
}
FEATURE_ORDER = ["all", "header", "section", "strings", "imports"]
SETTING_ORDER = ["original_mixed", "mixed_balanced", "unpacked_balanced"]
SETTING_LABELS = {
    "original_mixed": "Wild (U)",
    "mixed_balanced": "Wild (B)",
    "unpacked_balanced": "Unpacked (B)",
}
SETTING_COLORS = {
    "original_mixed": "#1f4e79",
    "mixed_balanced": "#c75b12",
    "unpacked_balanced": "#2a7f62",
}


def style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["axes.labelsize"] = 19
    plt.rcParams["xtick.labelsize"] = 16
    plt.rcParams["ytick.labelsize"] = 16


def load_summary(path: Path) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["scope"] != "metric":
                continue
            feature = row["feature_group"]
            metric = row["metric"]
            out.setdefault(feature, {})[metric] = float(row["mean"])
    return out


def write_comparison_table(
    summaries: Dict[str, Dict[str, Dict[str, float]]],
    out_path: Path,
) -> Path:
    fieldnames = [
        "feature_subset",
        "original_mixed_auc",
        "original_mixed_acc",
        "original_mixed_mix_at_10",
        "original_mixed_js_divergence",
        "mixed_balanced_auc",
        "mixed_balanced_acc",
        "mixed_balanced_mix_at_10",
        "mixed_balanced_js_divergence",
        "unpacked_balanced_auc",
        "unpacked_balanced_acc",
        "unpacked_balanced_mix_at_10",
        "unpacked_balanced_js_divergence",
    ]
    rows = []
    for feature in FEATURE_ORDER:
        row = {"feature_subset": feature}
        for setting in SETTING_ORDER:
            metrics = summaries[setting][feature]
            row[f"{setting}_auc"] = round(metrics["auc"], 4)
            row[f"{setting}_acc"] = round(metrics["acc"], 4)
            row[f"{setting}_mix_at_10"] = round(metrics["mix_at_k"], 4)
            row[f"{setting}_js_divergence"] = round(metrics["js_divergence"], 4)
        rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def save_auc_comparison_figure(
    summaries: Dict[str, Dict[str, Dict[str, float]]],
    out_path: Path,
) -> Path:
    x = np.arange(len(FEATURE_ORDER))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.6, 4.9))

    for idx, setting in enumerate(SETTING_ORDER):
        offsets = (-width, 0.0, width)
        auc_values = [summaries[setting][feature]["auc"] for feature in FEATURE_ORDER]
        ax.bar(
            x + offsets[idx],
            auc_values,
            width=width,
            color=SETTING_COLORS[setting],
            label=SETTING_LABELS[setting],
        )

    ax.set_xticks(x)
    ax.set_xticklabels([feature.title() for feature in FEATURE_ORDER], rotation=-20, ha="left")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.90, 1.0)
    ax.grid(True, axis="y", linestyle="-", linewidth=0.9, alpha=0.5)
    ax.grid(True, axis="x", linestyle="-", linewidth=0.9, alpha=0.4)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=1)
    fig.tight_layout(rect=(0.0, 0.0, 0.84, 1.0))

    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_path.with_suffix(".png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a three-setting RQ1 hierarchy figure/table from aggregate summaries.")
    parser.add_argument("--original", type=Path, default=SETTINGS["original_mixed"])
    parser.add_argument("--balanced", type=Path, default=SETTINGS["mixed_balanced"])
    parser.add_argument("--unpacked", type=Path, default=SETTINGS["unpacked_balanced"])
    parser.add_argument(
        "--figure-stem",
        type=Path,
        default=FIGURE_DIR / "feature_subset_auc_hierarchy_lr",
        help="Output path without extension for the hierarchy figure.",
    )
    parser.add_argument(
        "--table-path",
        type=Path,
        default=TABLE_DIR / "table_rq1_three_setting_comparison.csv",
        help="Output CSV path for the three-setting comparison table.",
    )
    args = parser.parse_args()

    style()
    paths = {
        "original_mixed": args.original,
        "mixed_balanced": args.balanced,
        "unpacked_balanced": args.unpacked,
    }
    summaries = {setting: load_summary(path) for setting, path in paths.items()}
    write_comparison_table(summaries, args.table_path)
    save_auc_comparison_figure(summaries, args.figure_stem)
    print(f"Saved RQ1 figures/tables to {BASE_DIR}")


if __name__ == "__main__":
    main()
