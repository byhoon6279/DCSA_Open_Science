#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42  # embed fonts as TrueType
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

try:
    import umap
except ImportError as exc:  # pragma: no cover - CLI guard
    raise SystemExit(
        "umap-learn is required for this script. Install it in the active environment first."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[3]  # Appendix/Section_C/scripts/<file> -> parents[0]=scripts,[1]=Section_C,[2]=Appendix,[3]=package root
SECTION_ROOT = REPO_ROOT / "artifacts" / "5_1_representation_level_separability_disagreement"
_SHARED_LIB_DIR = REPO_ROOT / "artifacts" / "shared"
sys.path.insert(0, str(_SHARED_LIB_DIR))

from common import (  # type: ignore
    balance_samples,
    labels_array,
    load_samples_from_paths,
    scale_train_test,
    select_feature_group,
    week_paths,
)


FIGURE_DIR = Path(__file__).resolve().parents[1] / "results"
OUTPUT_NAME_TEMPLATE = "umap_feature_subset_{feature_group}_mixed_balanced.pdf"

# Only the Wild (B) setting is used by the manuscript (Appendix Figure C.1),
# split into one file per feature group via OUTPUT_NAME_TEMPLATE.
SETTING_PRESETS = {
    "mixed_balanced": {
        "results_json": SECTION_ROOT / "results" / "LR" / "win32_all_train_all_test_balanced_test" / "results.json",
        "lr_summary_csv": SECTION_ROOT / "results" / "LR" / "win32_all_train_all_test_balanced_test" / "balanced_test_aggregate_summary.csv",
        "lgbm_summary_csv": SECTION_ROOT / "results" / "LightGBM" / "win32_all_train_all_test_lightgbm_balanced_test" / "balanced_test_aggregate_summary_lightgbm.csv",
        "output_dir": FIGURE_DIR,
        "view_label": "Wild (B)",
    },
}

FEATURE_GROUPS = ["all", "header", "section", "imports"]
CLASS_LABELS = {
    0: "Benign",
    1: "Malware",
}
CLASS_COLORS = {
    0: "#1f77b4",
    1: "#d62728",
}

UMAP_PARAMS = {
    "n_neighbors": 25,
    "min_dist": 0.18,
    "metric": "euclidean",
    "n_components": 2,
    "random_state": 42,
}


def style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["axes.labelsize"] = 25
    plt.rcParams["xtick.labelsize"] = 22
    plt.rcParams["ytick.labelsize"] = 22


def load_results_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    config = payload["config"] if "config" in payload and isinstance(payload["config"], dict) else payload
    data_root = Path(config["data_root"])
    if data_root.exists():
        return config
    if not data_root.is_absolute():
        candidates = [(path.parent / data_root).resolve(), (REPO_ROOT / data_root).resolve()]
        trimmed_parts = list(data_root.parts)
        while trimmed_parts and trimmed_parts[0] in {".", ".."}:
            trimmed_parts.pop(0)
        if trimmed_parts:
            candidates.append((REPO_ROOT / Path(*trimmed_parts)).resolve())
    else:
        # Old configs record an absolute EMBER2024
        # path (e.g. /redacted-local-path/EMBER2024/dataset/features). The
        # public package expects the EMBER2024 weekly .jsonl files under
        # Data/ when feature-level reconstruction is required, with no
        # dataset/features subfolder, so re-root at the "EMBER2024" path
        # segment instead of trying to resolve the rest.
        candidates = []
        if "EMBER2024" in data_root.parts:
            candidates.append((REPO_ROOT / "Data").resolve())
        candidates.append(data_root)
    chosen = next((candidate for candidate in candidates if candidate.exists()), candidates[-1])
    config["data_root"] = str(chosen)
    return config


def load_metric_summary(path: Path) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["scope"] != "metric":
                continue
            summary.setdefault(row["feature_group"], {})[row["metric"]] = float(row["mean"])
    return summary


def resolve_test_paths(config: dict, weeks: Sequence[str] | None) -> List[Path]:
    data_root = Path(config["data_root"])
    platform = str(config["platform"])
    chosen_weeks = list(weeks) if weeks else list(config["test_weeks"])
    return week_paths(data_root, platform, chosen_weeks, "test")


def sample_balanced_pool(
    test_paths: Sequence[Path],
    packer_filter: str,
    seed: int,
    max_per_class: int,
) -> List[object]:
    pooled_samples = load_samples_from_paths(test_paths, packer_filter=packer_filter)
    if not pooled_samples:
        raise ValueError("No samples were loaded for the requested weeks and packer filter.")
    sampled = balance_samples(pooled_samples, seed=seed, max_per_class=max_per_class)
    if not sampled:
        raise ValueError("Balanced sampling produced an empty sample set.")
    return sampled


def compute_embedding(
    features: np.ndarray,
    random_state: int,
    n_neighbors: int,
    min_dist: float,
) -> np.ndarray:
    X_scaled, _ = scale_train_test(features, features)
    reducer = umap.UMAP(
        **{
            **UMAP_PARAMS,
            "random_state": random_state,
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
        }
    )
    return reducer.fit_transform(X_scaled)


def build_panel_payloads(
    samples: Sequence[object],
    lr_summary: Dict[str, Dict[str, float]],
    lgbm_summary: Dict[str, Dict[str, float]],
    seed: int,
    n_neighbors: int,
    min_dist: float,
) -> Dict[str, Dict[str, np.ndarray | float]]:
    labels = labels_array(samples)
    payloads: Dict[str, Dict[str, np.ndarray | float]] = {}

    for feature_group in FEATURE_GROUPS:
        features = select_feature_group(samples, feature_group)
        payloads[feature_group] = {
            "embedding": compute_embedding(
                features,
                random_state=seed,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
            ),
            "labels": labels,
            "lr_auc": lr_summary[feature_group]["auc"],
            "lgbm_auc": lgbm_summary[feature_group]["auc"],
            "mix_at_k": lr_summary[feature_group]["mix_at_k"],
            "js_divergence": lr_summary[feature_group]["js_divergence"],
        }
    return payloads


METRIC_BOX_CORNERS = {
    "upper_left": {"x": 0.03, "y": 0.97, "va": "top", "ha": "left"},
    # "imports" panel's cluster sits in the upper-left, so its metric box
    # moves to the lower-right corner to avoid overlapping the scatter.
    "lower_right": {"x": 0.97, "y": 0.03, "va": "bottom", "ha": "right"},
    # "section" panel's cluster sits in the upper-left, so its metric box
    # moves to the upper-right corner to avoid overlapping the scatter.
    "upper_right": {"x": 0.97, "y": 0.97, "va": "top", "ha": "right"},
}


def add_metric_box(
    ax: plt.Axes,
    payload: Dict[str, np.ndarray | float],
    mode: str,
    corner: str = "upper_left",
) -> None:
    if mode == "dual_auc":
        text = "\n".join(
            [
                f"LR AUC: {payload['lr_auc']:.3f}",
                f"LGBM AUC: {payload['lgbm_auc']:.3f}",
                f"Mix@10: {payload['mix_at_k']:.3f}",
                f"JS: {payload['js_divergence']:.3f}",
            ]
        )
    elif mode == "lr_single_auc":
        text = "\n".join(
            [
                f"AUC: {payload['lr_auc']:.3f}",
                f"Mix@10: {payload['mix_at_k']:.3f}",
                f"JS: {payload['js_divergence']:.3f}",
            ]
        )
    elif mode == "lgbm_single_auc":
        text = "\n".join(
            [
                f"AUC: {payload['lgbm_auc']:.3f}",
                f"Mix@10: {payload['mix_at_k']:.3f}",
                f"JS: {payload['js_divergence']:.3f}",
            ]
        )
    else:
        text = "\n".join(
            [
                f"AUC: {payload['lr_auc']:.3f}",
                f"Mix@10: {payload['mix_at_k']:.3f}",
                f"JS: {payload['js_divergence']:.3f}",
            ]
        )
    position = METRIC_BOX_CORNERS[corner]
    ax.text(
        position["x"],
        position["y"],
        text,
        transform=ax.transAxes,
        va=position["va"],
        ha=position["ha"],
        fontsize=19,
        bbox={
            "boxstyle": "round,pad=0.32",
            "facecolor": "white",
            "edgecolor": "#d0d0d0",
            "alpha": 0.95,
        },
    )


def draw_panel(
    ax: plt.Axes,
    payload: Dict[str, np.ndarray | float],
    show_ylabel: bool,
    metric_box_mode: str,
    metric_box_corner: str = "upper_left",
) -> None:
    embedding = payload["embedding"]
    labels = payload["labels"]

    for class_id in [0, 1]:
        mask = labels == class_id
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=18,
            alpha=0.58,
            c=CLASS_COLORS[class_id],
            label=CLASS_LABELS[class_id],
            edgecolors="none",
            rasterized=True,
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2" if show_ylabel else "")
    ax.grid(True, axis="y", linestyle="-", linewidth=0.9, alpha=0.5)
    ax.grid(True, axis="x", linestyle="-", linewidth=0.9, alpha=0.4)
    add_metric_box(ax, payload, mode=metric_box_mode, corner=metric_box_corner)


def save_panel_figure(
    payload: Dict[str, np.ndarray | float],
    output_path: Path,
    metric_box_mode: str,
    metric_box_corner: str = "upper_left",
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(7.6, 7.6))

    draw_panel(
        ax,
        payload,
        show_ylabel=True,
        metric_box_mode=metric_box_mode,
        metric_box_corner=metric_box_corner,
    )

    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        fontsize=30,
        markerscale=3.0,
        scatterpoints=1,
        frameon=False,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")
    fig.tight_layout(pad=1.4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the Appendix Figure C.1 UMAP feature-subset comparison panels."
    )
    parser.add_argument(
        "--setting-preset",
        choices=list(SETTING_PRESETS.keys()),
        default="mixed_balanced",
        help="Evaluation setting; only `mixed_balanced` (Wild (B)) is used by the manuscript.",
    )
    parser.add_argument("--results-json", type=Path)
    parser.add_argument("--lr-summary-csv", type=Path)
    parser.add_argument("--lgbm-summary-csv", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to write one PDF per feature group into (default: results/figures/).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help=(
            "Override the EMBER2024 feature data_root recorded in results.json, for "
            "machines where the dataset lives at a different absolute path."
        ),
    )
    parser.add_argument(
        "--weeks",
        nargs="*",
        help="Optional explicit test weeks. Defaults to all test weeks in results.json.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=1200,
        help="Maximum number of benign/malware samples to pool for the UMAP figure.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-neighbors", type=int, default=25)
    parser.add_argument("--min-dist", type=float, default=0.18)
    parser.add_argument("--view-label", type=str)
    parser.add_argument(
        "--metric-box-mode",
        choices=["geometry_only", "dual_auc", "lr_single_auc", "lgbm_single_auc"],
        default="geometry_only",
        help=(
            "Use `geometry_only` for the paper main figure, `lr_single_auc` or "
            "`lgbm_single_auc` for model-specific companion figures, or `dual_auc` "
            "for an appendix robustness view."
        ),
    )
    args = parser.parse_args()

    style()
    preset = SETTING_PRESETS[args.setting_preset]
    results_json = args.results_json or preset["results_json"]
    lr_summary_csv = args.lr_summary_csv or preset["lr_summary_csv"]
    lgbm_summary_csv = args.lgbm_summary_csv or preset["lgbm_summary_csv"]
    output_dir = args.output_dir or preset["output_dir"]
    view_label = args.view_label or preset["view_label"]

    config = load_results_config(results_json)
    if args.data_root:
        config["data_root"] = str(args.data_root)
    lr_summary = load_metric_summary(lr_summary_csv)
    lgbm_summary = load_metric_summary(lgbm_summary_csv)
    chosen_weeks = list(args.weeks) if args.weeks else list(config["test_weeks"])
    test_paths = resolve_test_paths(config, chosen_weeks)
    samples = sample_balanced_pool(
        test_paths=test_paths,
        packer_filter=str(config.get("packer_filter", "all")),
        seed=args.seed,
        max_per_class=args.max_per_class,
    )
    labels = labels_array(samples)
    per_class_count = int(min(np.sum(labels == 0), np.sum(labels == 1)))
    payloads = build_panel_payloads(
        samples,
        lr_summary=lr_summary,
        lgbm_summary=lgbm_summary,
        seed=args.seed,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
    )
    for feature_group in FEATURE_GROUPS:
        output_path = output_dir / OUTPUT_NAME_TEMPLATE.format(feature_group=feature_group)
        if feature_group == "imports":
            metric_box_corner = "lower_right"
        elif feature_group == "section":
            metric_box_corner = "upper_right"
        else:
            metric_box_corner = "upper_left"
        save_panel_figure(
            payloads[feature_group],
            output_path=output_path,
            metric_box_mode=args.metric_box_mode,
            metric_box_corner=metric_box_corner,
        )
        print(f"Saved UMAP figure to {output_path}")


if __name__ == "__main__":
    main()
