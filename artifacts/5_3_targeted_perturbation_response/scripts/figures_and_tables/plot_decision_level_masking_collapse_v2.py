#!/usr/bin/env python3
"""
Build the LR and LightGBM decision-level masking panels for Figure 5.

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

Each mean point additionally gets a 95% CI error bar: for the matching
sibling trial_results.csv (seed/week-level rows next to the aggregate CSV),
important-mask values and week-averaged random-repeat values are averaged
over weeks per seed (5 seed-level values each), then percentile-bootstrapped
(10,000 resamples). This mirrors the seed-level unit of analysis used by the
Appendix E paired-difference bootstrap, but is computed independently on the
important and random arms separately rather than on their paired difference.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42  # embed fonts as TrueType
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260805

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
    "Header": "#C44536",
    "Imports": "#2F6BFF",
}
SUBSET_MARKERS = {
    "Header": "s",
    "Imports": "o",
}
SUBSET_OFFSETS = {
    "Header": -0.05,
    "Imports": 0.05,
}
LABEL_TO_FEATURE = {label: group for group, label in FEATURE_TO_LABEL.items()}
RANDOM_ALPHA = 0.55
DECISION_FIGURE_SIZE = (6.8, 5.0)
DECISION_SUBPLOTS = {
    "left": 0.21,
    "bottom": 0.40,
    "right": 0.98,
    "top": 0.95,
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


def bootstrap_ci(
    values: List[float],
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    resample_means = []
    for _ in range(n_resamples):
        resample_means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    resample_means.sort()
    lower_idx = int(round(0.025 * (n_resamples - 1)))
    upper_idx = int(round(0.975 * (n_resamples - 1)))
    return resample_means[lower_idx], resample_means[upper_idx]


def seed_level_normalized_drops(
    trial_results_path: Path,
    feature_group: str,
    strength: float,
    perturbation_type: str,
) -> List[float]:
    """Five seed-level normalized-AUC-degradation values: within each seed,
    average over the 12 test weeks; within each seed-week, average repeat
    trials (only relevant for perturbation_type='random', which has multiple
    repeats per seed-week — 'important' has exactly one)."""
    seed_week_values: Dict[tuple, List[float]] = defaultdict(list)
    for row in read_rows(trial_results_path):
        if row["feature_group"] != feature_group or row["perturbation_type"] != perturbation_type:
            continue
        if not math.isclose(float(row["strength"]), strength, rel_tol=0.0, abs_tol=1e-12):
            continue
        delta_auc = float(row["delta_auc"])
        baseline_auc = float(row["auc"]) - delta_auc
        margin = max(baseline_auc - 0.5, 1e-8)
        normalized_drop = max(0.0, -delta_auc) / margin
        seed_week_values[(int(row["seed"]), row["test_week"])].append(normalized_drop)

    seed_to_week_means: Dict[int, List[float]] = defaultdict(list)
    for (seed, _week), values in seed_week_values.items():
        seed_to_week_means[seed].append(sum(values) / len(values))

    assert len(seed_to_week_means) == 5, (
        f"expected 5 seeds for {feature_group}/{strength}/{perturbation_type}, "
        f"got {len(seed_to_week_means)}"
    )
    assert all(len(weeks) == 12 for weeks in seed_to_week_means.values()), (
        f"expected 12 weeks per seed for {feature_group}/{strength}/{perturbation_type}, "
        f"got {[len(weeks) for weeks in seed_to_week_means.values()]}"
    )
    return [sum(values) / len(values) for _, values in sorted(seed_to_week_means.items())]


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
    trial_results_path: Path,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=DECISION_FIGURE_SIZE)
    x = list(range(len(STRENGTHS)))
    y_bottom, y_top = compute_setting_ylim(rows, setting)

    all_ci_highs: List[float] = []
    for subset in SUBSETS:
        color = SUBSET_COLORS[subset]
        marker = SUBSET_MARKERS[subset]
        offset = SUBSET_OFFSETS[subset]

        # Point and CI must share the same estimand: both are computed from
        # the same unrounded seed-level values (not the 3-decimal-rounded
        # aggregate_results.csv figures used elsewhere), so the point always
        # falls inside its own CI by construction.
        important_seed_lists = [
            seed_level_normalized_drops(trial_results_path, LABEL_TO_FEATURE[subset], strength_float, "important")
            for strength_float in STRENGTH_FLOATS
        ]
        random_seed_lists = [
            seed_level_normalized_drops(trial_results_path, LABEL_TO_FEATURE[subset], strength_float, "random")
            for strength_float in STRENGTH_FLOATS
        ]
        important_values = [sum(values) / len(values) for values in important_seed_lists]
        random_values = [sum(values) / len(values) for values in random_seed_lists]
        important_cis = [bootstrap_ci(values) for values in important_seed_lists]
        random_cis = [bootstrap_ci(values) for values in random_seed_lists]
        all_ci_highs.extend(high for _low, high in important_cis)
        all_ci_highs.extend(high for _low, high in random_cis)

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
        ax.errorbar(
            x_positions,
            important_values,
            yerr=[
                [v - lo for v, (lo, _hi) in zip(important_values, important_cis)],
                [hi - v for v, (_lo, hi) in zip(important_values, important_cis)],
            ],
            fmt="none",
            ecolor=color,
            elinewidth=1.4,
            capsize=3,
            zorder=4,
        )
        ax.plot(
            x_positions,
            random_values,
            color=color,
            marker=marker,
            linewidth=1.8,
            linestyle="--",
            alpha=RANDOM_ALPHA,
            zorder=2,
        )
        ax.errorbar(
            x_positions,
            random_values,
            yerr=[
                [v - lo for v, (lo, _hi) in zip(random_values, random_cis)],
                [hi - v for v, (_lo, hi) in zip(random_values, random_cis)],
            ],
            fmt="none",
            ecolor=color,
            elinewidth=1.2,
            capsize=3,
            alpha=RANDOM_ALPHA,
            zorder=1,
        )

    if all_ci_highs:
        ci_ymax = math.ceil((max(all_ci_highs) + 0.01) * 20) / 20
        y_top = max(y_top, ci_ymax)

    ax.set_xticks(x)
    ax.set_xticklabels(STRENGTHS)
    ax.set_xlabel("Masking Ratio")
    ax.set_ylabel("Normalized AUC\nDegradation")
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
        Line2D([0], [0], color="black", lw=1.8, linestyle="--", alpha=RANDOM_ALPHA, label="Random"),
    ]
    legend_subsets = fig.legend(
        subset_handles,
        [h.get_label() for h in subset_handles],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.15),
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
        bbox_to_anchor=(0.5, 0.05),
        fontsize=LEGEND_FONT_SIZE,
        columnspacing=1.8,
        handlelength=2.6,
    )
    fig.subplots_adjust(**DECISION_SUBPLOTS)
    fig.savefig(output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the LR and LightGBM decision-level masking panels for Figure 5.")
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
    plot_single_model_setting(
        lr_rows,
        args.output_dir / "decision_level_masking_collapse_wild_b_lr.pdf",
        "LR",
        "Wild (B)",
        trial_results_path=args.lr_controlled_csv.parent / "trial_results.csv",
    )
    plot_single_model_setting(
        lgbm_rows,
        args.output_dir / "decision_level_masking_collapse_wild_b_lightgbm.pdf",
        "LightGBM",
        "Wild (B)",
        trial_results_path=args.lgbm_controlled_csv.parent / "trial_results.csv",
    )
    print(f"Saved Figure 5 plots to {args.output_dir}")


if __name__ == "__main__":
    main()
