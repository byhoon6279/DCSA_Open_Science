#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42  # embed fonts as TrueType
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import numpy as np

TITLE_FONT_SIZE = 24
AXIS_LABEL_FONT_SIZE = 19
TICK_FONT_SIZE = 19
LEGEND_FONT_SIZE = 17

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260805

plt.rcParams.update({
    "font.size": TICK_FONT_SIZE,
    "axes.labelsize": AXIS_LABEL_FONT_SIZE,
    "xtick.labelsize": TICK_FONT_SIZE,
    "ytick.labelsize": TICK_FONT_SIZE,
    "legend.fontsize": LEGEND_FONT_SIZE,
})

SUBSETS = ["Header", "Imports"]
FEATURE_TO_LABEL = {
    "header": "Header",
    "imports": "Imports",
}
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
LABEL_TO_FEATURE = {label: group for group, label in FEATURE_TO_LABEL.items()}
RANDOM_ALPHA = 0.55
STRENGTHS = ["1%", "5%", "10%"]
STRENGTH_FLOATS = [0.01, 0.05, 0.10]
STRENGTH_LABELS = {
    "1%": 100,
    "5%": 200,
    "10%": 320,
}
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


def aggregate_auc_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
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
        grouped[(feature_group, perturbation_type, strength)].append(auc_drop / available_margin)

    output_rows: List[Dict[str, str]] = []
    for feature_group, subset_label in FEATURE_TO_LABEL.items():
        for strength_float, strength_label in zip(STRENGTH_FLOATS, STRENGTHS):
            imp_values = grouped[(feature_group, "important", strength_float)]
            rnd_values = grouped[(feature_group, "random", strength_float)]
            output_rows.append(
                {
                    "Subset": subset_label,
                    "Strength": strength_label,
                    "Normalized AUC Degradation (Imp)": f"{sum(imp_values) / len(imp_values):.3f}",
                    "Normalized AUC Degradation (Rand)": f"{sum(rnd_values) / len(rnd_values):.3f}",
                }
            )
    return output_rows


def aggregate_flip_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
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
        grouped[(feature_group, perturbation_type, strength)].append(float(row["flip_rate_mean"]))

    output_rows: List[Dict[str, str]] = []
    for feature_group, subset_label in FEATURE_TO_LABEL.items():
        for strength_float, strength_label in zip(STRENGTH_FLOATS, STRENGTHS):
            imp_values = grouped[(feature_group, "important", strength_float)]
            rnd_values = grouped[(feature_group, "random", strength_float)]
            output_rows.append(
                {
                    "Subset": subset_label,
                    "Strength": strength_label,
                    "Flip Rate (Imp)": f"{sum(imp_values) / len(imp_values):.3f}",
                    "Flip Rate (Rand)": f"{sum(rnd_values) / len(rnd_values):.3f}",
                }
            )
    return output_rows


def aggregate_structural_points(rows: Iterable[Dict[str, str]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple, Dict[str, float]] = defaultdict(lambda: {"auc": 0.0, "mix": 0.0, "n": 0.0})
    for row in rows:
        feature_group = row["feature_group"]
        perturbation_type = row["perturbation_type"]
        strength = float(row["strength"])
        if feature_group not in FEATURE_TO_LABEL:
            continue
        if perturbation_type != "important":
            continue
        if strength not in STRENGTH_FLOATS:
            continue
        key = (FEATURE_TO_LABEL[feature_group], f"{int(strength * 100)}%")
        grouped[key]["auc"] += max(0.0, -float(row["delta_auc_mean"]))
        grouped[key]["mix"] += float(row["delta_mix_at_k_mean"])
        grouped[key]["n"] += 1.0

    output: List[Dict[str, object]] = []
    for (subset, strength), agg in grouped.items():
        n = max(agg["n"], 1.0)
        output.append(
            {
                "subset": subset,
                "strength": strength,
                "auc_drop": agg["auc"] / n,
                "delta_mix": agg["mix"] / n,
            }
        )
    return output


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
    trials (only relevant for perturbation_type='random')."""
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


def build_lookup(rows: List[Dict[str, str]], important_col: str, random_col: str) -> Dict[tuple, Dict[str, float]]:
    return {
        (row["Subset"], row["Strength"]): {
            important_col: float(row[important_col]),
            random_col: float(row[random_col]),
        }
        for row in rows
    }


def compute_ylim(rows: List[Dict[str, str]], important_col: str, random_col: str) -> tuple[float, float]:
    lookup = build_lookup(rows, important_col, random_col)
    values: List[float] = []
    for subset in SUBSETS:
        for strength in STRENGTHS:
            row = lookup[(subset, strength)]
            values.append(row[important_col])
            values.append(row[random_col])
    ymax = max(values) if values else 0.1
    y_bottom = 0.0
    y_top = math.ceil((ymax + 0.01) * 20) / 20
    return y_bottom, max(y_top, 0.05)


def plot_single_view_lines(
    rows: List[Dict[str, str]],
    important_col: str,
    random_col: str,
    ylabel: str,
    output_path: Path,
    subset_colors: Dict[str, str] = SUBSET_COLORS,
    # "decision" matches decision_level_masking_collapse's fixed-margin layout
    # (DECISION_FIGURE_SIZE, subplots_adjust). "flip" matches the LR/LightGBM
    # prediction_flip_rate_masking_wild_b_{lr,lightgbm}.pdf sibling figures'
    # own layout (plot_prediction_flip_lr.py/plot_prediction_flip_lightgbm.py:
    # figsize (5.8, 4.8), tight_layout, bbox_inches="tight") so the RF panel
    # comes out the same page size/aspect ratio as those two.
    layout: str = "decision",
    trial_results_path: Optional[Path] = None,
) -> None:
    figsize = DECISION_FIGURE_SIZE if layout == "decision" else (5.8, 4.8)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    lookup = build_lookup(rows, important_col, random_col)
    x = list(range(len(STRENGTHS)))
    y_bottom, y_top = compute_ylim(rows, important_col, random_col)
    # Only the Figure 5 (decision-level AUC) panel gets the higher-visibility
    # random-line alpha and CI error bars; the Figure 7 flip-rate panel this
    # function also builds is intentionally left at its original 0.35/no-CI
    # styling since only Figure 5 uncertainty was requested.
    random_alpha = RANDOM_ALPHA if trial_results_path is not None else 0.35

    all_ci_highs: List[float] = []
    for subset in SUBSETS:
        color = subset_colors[subset]
        marker = SUBSET_MARKERS[subset]
        offset = SUBSET_OFFSETS[subset]

        important_cis = None
        random_cis = None
        if trial_results_path is not None:
            # Point and CI must share the same estimand: both computed from
            # the same unrounded seed-level values, not the 3-decimal-rounded
            # aggregate-CSV figures `lookup` holds (used for the flip-rate
            # panel, which has no seed-level CI).
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
        else:
            important_values = [lookup[(subset, strength)][important_col] for strength in STRENGTHS]
            random_values = [lookup[(subset, strength)][random_col] for strength in STRENGTHS]

        x_positions = [value + offset for value in x]
        ax.plot(x_positions, important_values, color=color, marker=marker, linewidth=2.8, linestyle="-", zorder=3)
        ax.plot(
            x_positions,
            random_values,
            color=color,
            marker=marker,
            linewidth=1.8,
            linestyle="--",
            alpha=random_alpha,
            zorder=2,
        )

        if important_cis is not None and random_cis is not None:
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
        y_top = max(y_top, math.ceil((max(all_ci_highs) + 0.01) * 20) / 20)

    ax.set_xticks(x)
    ax.set_xticklabels(STRENGTHS)
    ax.set_xlabel("Masking Ratio")
    if ylabel == "Normalized AUC Degradation":
        ax.set_ylabel("Normalized AUC\nDegradation")
    else:
        ax.set_ylabel(ylabel)
    ax.set_ylim(y_bottom, y_top)
    ax.yaxis.set_major_locator(MultipleLocator(0.05 if y_top <= 0.30 else 0.1))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f" if y_top <= 0.30 else "%.1f"))
    style_axis(ax)

    subset_handles = [
        Line2D([0], [0], color=subset_colors[s], lw=2.8, marker=SUBSET_MARKERS[s], label=s)
        for s in SUBSETS
    ]
    style_handles = [
        Line2D([0], [0], color="black", lw=2.8, linestyle="-", label="Important"),
        Line2D([0], [0], color="black", lw=1.8, linestyle="--", alpha=random_alpha, label="Random"),
    ]
    legend_anchors = (0.5, 0.15), (0.5, 0.05)
    if layout == "flip":
        legend_anchors = (0.5, -0.01), (0.5, -0.09)
    legend_subsets = fig.legend(
        subset_handles,
        [h.get_label() for h in subset_handles],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=legend_anchors[0],
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
        bbox_to_anchor=legend_anchors[1],
        fontsize=LEGEND_FONT_SIZE,
        columnspacing=1.8,
        handlelength=2.6,
    )
    if layout == "decision":
        fig.subplots_adjust(**DECISION_SUBPLOTS)
        fig.savefig(output_path)
    else:
        fig.tight_layout(rect=(0, 0.18, 1, 1))
        fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_structural_response(points: List[Dict[str, object]], output_path: Path) -> None:
    # Header/Imports swapped relative to the module-level SUBSET_COLORS so this
    # panel's colors match the LR/LightGBM structural-response-map panels in
    # the same manuscript figure (plot_structural_response_map.py: Header
    # "#C44536", Imports "#2F6BFF") - only this figure is affected, not the
    # decision-level-collapse/prediction-flip RF figures this script also builds.
    structural_subset_colors = {"Header": "#C44536", "Imports": "#2F6BFF"}
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 6.0))
    x_values = [float(p["auc_drop"]) for p in points]
    y_values = [float(p["delta_mix"]) for p in points]
    x_max = max(x_values) if x_values else 0.2
    x_pad = max(0.01, x_max * 0.12)
    y_max = max(y_values) if y_values else 0.01
    y_pad = max(0.0004, y_max * 0.18)
    y_min = min(-0.0015, min(y_values) if y_values else 0.0)

    for subset in SUBSETS:
        subset_points = [p for p in points if p["subset"] == subset]
        for point in subset_points:
            ax.scatter(
                float(point["auc_drop"]),
                float(point["delta_mix"]),
                s=STRENGTH_LABELS[str(point["strength"])],
                color=structural_subset_colors[str(point["subset"])],
                marker="o",
                alpha=0.9,
                edgecolors="white",
                linewidths=0.9,
                zorder=3,
            )

    style_axis(ax)
    ax.set_box_aspect(1)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    x_max_plot = x_max + x_pad
    ax.set_xlim(0.0, x_max_plot)
    ax.set_xticks(np.linspace(0.0, x_max_plot, 3))
    ax.set_ylim(y_min, y_max + y_pad)
    ax.set_ylabel("Change in Local\n" + r"Class Mixing ($\Delta$Mix@10)", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_xlabel("Decision-Level\nAUC Degradation", fontsize=AXIS_LABEL_FONT_SIZE, labelpad=8)

    subset_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=structural_subset_colors[name], markersize=10, label=name)
        for name in SUBSETS
    ]
    strength_handles = [
        plt.scatter([], [], s=size, color="#777777", alpha=0.75, label=label)
        for label, size in STRENGTH_LABELS.items()
    ]

    # Fix the axes' position first so the legends below can be centered on
    # the plot area itself, not on the full figure canvas (the y-axis label
    # eats into the left margin, so figure-fraction x=0.5 sits left of the
    # plot's true center).
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
    parser = argparse.ArgumentParser(description="Generate Wild (B) RF manuscript figures for Section 5.3.")
    parser.add_argument("--rq2-2-csv", type=Path, required=True)
    parser.add_argument("--rq2-3-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rq2_2_rows = read_rows(args.rq2_2_csv)
    rq2_3_rows = read_rows(args.rq2_3_csv)

    auc_rows = aggregate_auc_rows(rq2_2_rows)
    flip_rows = aggregate_flip_rows(rq2_3_rows)
    structural_points = aggregate_structural_points(rq2_2_rows)

    plot_single_view_lines(
        auc_rows,
        important_col="Normalized AUC Degradation (Imp)",
        random_col="Normalized AUC Degradation (Rand)",
        ylabel="Normalized AUC Degradation",
        output_path=args.output_dir / "decision_level_masking_collapse_wild_b_rf.pdf",
        # Header/Imports swapped relative to the module-level SUBSET_COLORS so
        # this figure matches its LR/LightGBM siblings (Header=#C44536,
        # Imports=#2F6BFF); the prediction-flip-rate figure below is now
        # overridden to the same scheme (see its own call below).
        subset_colors={"Header": "#C44536", "Imports": "#2F6BFF"},
        trial_results_path=args.rq2_2_csv.parent / "trial_results.csv",
    )
    plot_structural_response(
        structural_points,
        output_path=args.output_dir / "structural_response_map_masking_wild_b_rf.pdf",
    )
    plot_single_view_lines(
        flip_rows,
        important_col="Flip Rate (Imp)",
        random_col="Flip Rate (Rand)",
        ylabel="Flip Rate",
        output_path=args.output_dir / "prediction_flip_rate_masking_wild_b_rf.pdf",
        # Header=red/Imports=blue, matching this figure's LR/LightGBM siblings
        # and the other two RF panels this script builds.
        subset_colors={"Header": "#C44536", "Imports": "#2F6BFF"},
        # Match the LR/LightGBM siblings' page size/aspect ratio too (see
        # plot_single_view_lines' layout parameter above).
        layout="flip",
    )
    print(f"Saved RF Wild (B) manuscript figures to {args.output_dir}")


if __name__ == "__main__":
    main()
