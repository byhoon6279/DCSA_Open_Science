#!/usr/bin/env python3
"""
Density-conditioned PE-inspired intervention probe for Section 5.5.

Reuses the PE-inspired intervention implementation from Section 5.3:
- compute density in the same representation space
- compare pooled evaluation against low-density-only evaluation
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors


REPO_ROOT = Path(__file__).resolve().parents[4]
RQ2_4_PATH = (
    REPO_ROOT
    / "artifacts"
    / "5_3_targeted_perturbation_response"
    / "pe_inspired_feature_intervention"
    / "scripts"
    / "run_pe_feature_intervention_validation.py"
)


def load_rq2_4_module():
    spec = importlib.util.spec_from_file_location("rq2_4_pe", RQ2_4_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {RQ2_4_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rq2 = load_rq2_4_module()


def compute_density_scores(X_train_scaled: np.ndarray, X_test_scaled: np.ndarray, k_density: int) -> np.ndarray:
    effective_k = max(1, min(int(k_density), len(X_train_scaled)))
    nn = NearestNeighbors(n_neighbors=effective_k, metric="cosine")
    nn.fit(np.asarray(X_train_scaled, dtype=np.float64))
    distances, _ = nn.kneighbors(np.asarray(X_test_scaled, dtype=np.float64))
    return -distances.mean(axis=1)


def assign_density_bins(scores: np.ndarray) -> tuple[np.ndarray, Dict[str, float]]:
    q25 = float(np.quantile(scores, 0.25))
    q75 = float(np.quantile(scores, 0.75))
    bins = np.full(len(scores), "mid_density", dtype=object)
    bins[scores <= q25] = "low_density"
    bins[scores >= q75] = "high_density"
    return bins, {"q25": q25, "q75": q75}


def evaluate_subset(
    *,
    y_subset: np.ndarray,
    baseline_proba_subset: np.ndarray,
    perturbed_proba_subset: np.ndarray,
    baseline_auc_subset: float,
    decision_threshold: float,
    boundary_quantiles: Sequence[float],
) -> Dict[str, float]:
    malware_mask = y_subset == 1
    if not np.any(malware_mask):
        raise ValueError("Subset must contain at least one malware sample.")
    metrics = rq2.evaluate_metrics(
        baseline_proba_full=baseline_proba_subset,
        perturbed_proba_full=perturbed_proba_subset,
        baseline_proba_malware=baseline_proba_subset[malware_mask],
        perturbed_proba_malware=perturbed_proba_subset[malware_mask],
        y_test=y_subset,
        decision_threshold=decision_threshold,
        boundary_quantiles=boundary_quantiles,
    )
    metrics["baseline_auc"] = baseline_auc_subset
    metrics["delta_auc"] = float(metrics["auc"] - baseline_auc_subset)
    return metrics


def summarize_scope_row(
    *,
    seed: int,
    feature_group: str,
    operator: str,
    selection_type: str,
    selection_mode: str,
    selection_value: float,
    trial_id: int,
    n_selected_features: int,
    density_scope: str,
    density_bin: str,
    y_subset: np.ndarray,
    baseline_proba_subset: np.ndarray,
    perturbed_proba_subset: np.ndarray,
    baseline_auc_subset: float,
    decision_threshold: float,
    boundary_quantiles: Sequence[float],
) -> Dict[str, object]:
    metrics = evaluate_subset(
        y_subset=y_subset,
        baseline_proba_subset=baseline_proba_subset,
        perturbed_proba_subset=perturbed_proba_subset,
        baseline_auc_subset=baseline_auc_subset,
        decision_threshold=decision_threshold,
        boundary_quantiles=boundary_quantiles,
    )
    return {
        "seed": seed,
        "feature_group": feature_group,
        "operator": operator,
        "selection_type": selection_type,
        "selection_mode": selection_mode,
        "selection_value": selection_value,
        "trial_id": trial_id,
        "density_scope": density_scope,
        "density_bin": density_bin,
        "n_samples": int(len(y_subset)),
        "n_malware": int((y_subset == 1).sum()),
        "n_benign": int((y_subset == 0).sum()),
        "n_selected_features": n_selected_features,
        **metrics,
    }


def run_from_config(config: dict) -> dict:
    data_root = Path(config["data_root"])
    platform = str(config["platform"])
    packer_filter = str(config.get("packer_filter", "all"))
    train_weeks = list(config["train_weeks"])
    test_weeks = list(config["test_weeks"])
    seeds = [int(seed) for seed in config["seeds"]]
    feature_groups = list(config["feature_groups"])
    random_trials = int(config["random_trials"])
    decision_threshold = float(config.get("decision_threshold", 0.5))
    boundary_quantiles = [float(value) for value in config.get("boundary_quantiles", [0.1, 0.2])]
    balance_train = bool(config.get("balance_train", True))
    balance_test = bool(config.get("balance_test", True))
    max_train_per_class = config.get("max_train_per_class")
    max_test_per_class = config.get("max_test_per_class")
    model_type = str(config.get("model_type", "lightgbm"))
    importance_method = str(config.get("importance_method", "permutation"))
    candidate_profile = str(config.get("candidate_profile", "strict"))
    validation_fraction = float(config.get("validation_fraction", 0.2))
    permutation_repeats = int(config.get("permutation_repeats", 3))
    lgbm_n_jobs = int(config.get("lgbm_n_jobs", 2))
    omp_num_threads = int(config.get("omp_num_threads", 1))
    k_density = int(config.get("k_density", 10))
    density_bins_to_report = list(
        config.get(
            "density_bins_to_report",
            ["high_density", "mid_density", "low_density"],
        )
    )
    save_density_rows = bool(config.get("save_density_rows", False))

    if model_type == "lightgbm":
        rq2.configure_thread_env(omp_num_threads)

    train_paths = rq2.week_paths(data_root, platform, train_weeks, "train")
    all_train_samples = rq2.load_samples_from_paths(train_paths, packer_filter=packer_filter)
    test_samples_by_week = {
        week: rq2.load_samples_from_jsonl(
            rq2.week_paths(data_root, platform, [week], "test")[0],
            packer_filter=packer_filter,
        )
        for week in test_weeks
    }

    summary_rows: List[Dict[str, object]] = []
    density_rows: List[Dict[str, object]] = []

    for seed in seeds:
        train_samples = all_train_samples
        if balance_train:
            train_samples = rq2.balance_samples(all_train_samples, seed=seed, max_per_class=max_train_per_class)
        y_train = rq2.labels_array(train_samples)

        pooled_test_samples: List[object] = []
        for week_idx, week in enumerate(test_weeks):
            test_samples = test_samples_by_week[week]
            if balance_test:
                test_samples = rq2.balance_samples(
                    test_samples,
                    seed=seed + week_idx + 1,
                    max_per_class=max_test_per_class,
                )
            pooled_test_samples.extend(test_samples)

        y_test = rq2.labels_array(pooled_test_samples)

        for group in feature_groups:
            X_train_raw = rq2.select_feature_group(train_samples, group)
            X_test_raw = rq2.select_feature_group(pooled_test_samples, group)

            if model_type == "lightgbm" and importance_method == "permutation":
                train_idx, val_idx = rq2.split_train_validation_indices(
                    n_samples=len(train_samples),
                    y_train=y_train,
                    seed=seed,
                    validation_fraction=validation_fraction,
                )
                X_train_fit_raw = X_train_raw[train_idx]
                y_train_fit = y_train[train_idx]
                X_val_raw = X_train_raw[val_idx]
                y_val = y_train[val_idx]
            else:
                X_train_fit_raw = X_train_raw
                y_train_fit = y_train
                X_val_raw = None
                y_val = None

            scaler = rq2.fit_scaler(X_train_fit_raw)
            X_train_scaled = rq2.transform_with_scaler(scaler, X_train_fit_raw)
            X_test_scaled = rq2.transform_with_scaler(scaler, X_test_raw)
            clf = rq2.fit_classifier(
                X_train_scaled,
                y_train_fit,
                seed=seed,
                model_type=model_type,
                lgbm_n_jobs=lgbm_n_jobs,
            )
            X_val_scaled = rq2.transform_with_scaler(scaler, X_val_raw) if X_val_raw is not None else None
            importance_scores = rq2.compute_feature_importance(
                clf=clf,
                model_type=model_type,
                importance_method=importance_method,
                X_val_scaled=X_val_scaled,
                y_val=y_val,
                seed=seed,
                permutation_repeats=permutation_repeats,
            )
            stats = rq2.compute_distribution_stats(X_train_fit_raw, y_train_fit)
            candidate_indices = rq2.candidate_indices_for_group(group, candidate_profile, X_train_raw.shape[1])
            selection_levels = rq2.selection_levels_for_group(group, config, len(candidate_indices))
            operators = rq2.operators_for_group(group, config)

            baseline_scaled = np.asarray(X_test_scaled, dtype=np.float64)
            baseline_proba_full = rq2.positive_proba(clf, baseline_scaled)
            malware_mask = y_test == 1
            malware_indices = np.flatnonzero(malware_mask)
            baseline_malware_raw = np.asarray(X_test_raw[malware_mask], dtype=np.float64)
            baseline_malware_scaled = np.asarray(baseline_scaled[malware_mask], dtype=np.float64)

            density_scores = compute_density_scores(X_train_scaled, baseline_scaled, k_density=k_density)
            density_bins, thresholds = assign_density_bins(density_scores)
            density_masks = {
                density_bin: density_bins == density_bin
                for density_bin in density_bins_to_report
            }
            baseline_scope_cache = {
                "pooled": {
                    "density_bin": "all",
                    "mask": np.ones(len(y_test), dtype=bool),
                    "baseline_auc": float(roc_auc_score(y_test, baseline_proba_full)),
                }
            }
            for density_bin, mask in density_masks.items():
                y_subset = y_test[mask]
                if len(np.unique(y_subset)) < 2:
                    continue
                baseline_scope_cache[density_bin] = {
                    "density_bin": density_bin,
                    "mask": mask,
                    "baseline_auc": float(roc_auc_score(y_subset, baseline_proba_full[mask])),
                }

            if save_density_rows:
                for sample_index, sample in enumerate(pooled_test_samples):
                    density_rows.append(
                        {
                            "seed": seed,
                            "feature_group": group,
                            "sample_index": sample_index,
                            "sha256": getattr(sample, "sha256", ""),
                            "label": int(getattr(sample, "label", -1)),
                            "density_score": float(density_scores[sample_index]),
                            "density_bin": str(density_bins[sample_index]),
                            "density_q25": thresholds["q25"],
                            "density_q75": thresholds["q75"],
                        }
                    )

            for operator in operators:
                for selection in selection_levels:
                    selection_mode = str(selection["selection_mode"])
                    selection_value = float(selection["selection_value"])
                    n_select = int(selection["n_select"])
                    important_indices = rq2.select_top_indices_within_pool(
                        importance_scores,
                        candidate_indices,
                        n_select,
                    )
                    important_rng = np.random.default_rng(
                        rq2.intervention_seed(
                            seed=seed,
                            week_idx=0,
                            group=group,
                            operator=operator,
                            selection_value=selection_value,
                            trial_id=0,
                            selection_type="important",
                        )
                    )
                    important_scaled = rq2.build_perturbed_malware_scaled(
                        baseline_malware_raw=baseline_malware_raw,
                        baseline_malware_scaled=baseline_malware_scaled,
                        selected_indices=important_indices,
                        operator=operator,
                        stats=stats,
                        scaler=scaler,
                        rng=important_rng,
                    )
                    important_proba_malware = rq2.positive_proba(clf, important_scaled)
                    important_proba_full = np.array(baseline_proba_full, copy=True)
                    important_proba_full[malware_indices] = important_proba_malware

                    summary_rows.append(
                        summarize_scope_row(
                            seed=seed,
                            feature_group=group,
                            operator=operator,
                            selection_type="important",
                            selection_mode=selection_mode,
                            selection_value=selection_value,
                            trial_id=0,
                            n_selected_features=len(important_indices),
                            density_scope="pooled",
                            density_bin="all",
                            y_subset=y_test,
                            baseline_proba_subset=baseline_proba_full,
                            perturbed_proba_subset=important_proba_full,
                            baseline_auc_subset=float(baseline_scope_cache["pooled"]["baseline_auc"]),
                            decision_threshold=decision_threshold,
                            boundary_quantiles=boundary_quantiles,
                        )
                    )
                    for density_bin in density_bins_to_report:
                        scope = baseline_scope_cache.get(density_bin)
                        if scope is None:
                            continue
                        mask = np.asarray(scope["mask"], dtype=bool)
                        summary_rows.append(
                            summarize_scope_row(
                                seed=seed,
                                feature_group=group,
                                operator=operator,
                                selection_type="important",
                                selection_mode=selection_mode,
                                selection_value=selection_value,
                                trial_id=0,
                                n_selected_features=len(important_indices),
                                density_scope="density_restricted",
                                density_bin=density_bin,
                                y_subset=y_test[mask],
                                baseline_proba_subset=baseline_proba_full[mask],
                                perturbed_proba_subset=important_proba_full[mask],
                                baseline_auc_subset=float(scope["baseline_auc"]),
                                decision_threshold=decision_threshold,
                                boundary_quantiles=boundary_quantiles,
                            )
                        )

                    for trial_id in range(1, random_trials + 1):
                        random_rng = np.random.default_rng(
                            rq2.intervention_seed(
                                seed=seed,
                                week_idx=0,
                                group=group,
                                operator=operator,
                                selection_value=selection_value,
                                trial_id=trial_id,
                                selection_type="random",
                            )
                        )
                        random_indices = rq2.random_indices_from_pool(candidate_indices, n_select, random_rng)
                        random_scaled = rq2.build_perturbed_malware_scaled(
                            baseline_malware_raw=baseline_malware_raw,
                            baseline_malware_scaled=baseline_malware_scaled,
                            selected_indices=random_indices,
                            operator=operator,
                            stats=stats,
                            scaler=scaler,
                            rng=random_rng,
                        )
                        random_proba_malware = rq2.positive_proba(clf, random_scaled)
                        random_proba_full = np.array(baseline_proba_full, copy=True)
                        random_proba_full[malware_indices] = random_proba_malware

                        summary_rows.append(
                            summarize_scope_row(
                                seed=seed,
                                feature_group=group,
                                operator=operator,
                                selection_type="random",
                                selection_mode=selection_mode,
                                selection_value=selection_value,
                                trial_id=trial_id,
                                n_selected_features=n_select,
                                density_scope="pooled",
                                density_bin="all",
                                y_subset=y_test,
                                baseline_proba_subset=baseline_proba_full,
                                perturbed_proba_subset=random_proba_full,
                                baseline_auc_subset=float(baseline_scope_cache["pooled"]["baseline_auc"]),
                                decision_threshold=decision_threshold,
                                boundary_quantiles=boundary_quantiles,
                            )
                        )
                        for density_bin in density_bins_to_report:
                            scope = baseline_scope_cache.get(density_bin)
                            if scope is None:
                                continue
                            mask = np.asarray(scope["mask"], dtype=bool)
                            summary_rows.append(
                                summarize_scope_row(
                                    seed=seed,
                                    feature_group=group,
                                    operator=operator,
                                    selection_type="random",
                                    selection_mode=selection_mode,
                                    selection_value=selection_value,
                                    trial_id=trial_id,
                                    n_selected_features=n_select,
                                    density_scope="density_restricted",
                                    density_bin=density_bin,
                                    y_subset=y_test[mask],
                                    baseline_proba_subset=baseline_proba_full[mask],
                                    perturbed_proba_subset=random_proba_full[mask],
                                    baseline_auc_subset=float(scope["baseline_auc"]),
                                    decision_threshold=decision_threshold,
                                    boundary_quantiles=boundary_quantiles,
                                )
                            )

    return {
        "config": config,
        "summary_rows": summary_rows,
        "density_rows": density_rows,
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        import csv

        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_results(results: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "summary_rows.csv", results["summary_rows"])
    write_csv(output_dir / "density_rows.csv", results["density_rows"])
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(results["config"], handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the density-conditioned PE-inspired intervention probe for Section 5.5.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    data_root = Path(config["data_root"])
    if not data_root.is_absolute():
        candidates = [
            (args.config.parent / data_root).resolve(),
            (Path.cwd() / data_root).resolve(),
            (REPO_ROOT / data_root).resolve(),
        ]
        for candidate in candidates:
            if candidate.exists():
                config["data_root"] = str(candidate)
                break
        else:
            config["data_root"] = str(candidates[-1])
    results = run_from_config(config)
    save_results(results, args.output_dir)


if __name__ == "__main__":
    main()
