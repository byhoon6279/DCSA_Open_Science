#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import multiprocessing as mp
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[4]
_SHARED_LIB_DIR = REPO_ROOT / "artifacts" / "shared"
if str(_SHARED_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARED_LIB_DIR))

from common import (
    FEATURE_GROUPS,
    balance_samples,
    feature_rankings,
    labels_array,
    load_samples_from_jsonl,
    load_samples_from_paths,
    select_feature_group,
    stabilize_features,
    summarize,
    week_paths,
)


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "results" / "feature_subset_generalization_with_rf"

warnings.filterwarnings(
    "ignore",
    message="divide by zero encountered in matmul",
    category=RuntimeWarning,
    module=r"sklearn\.linear_model\._linear_loss",
)
warnings.filterwarnings(
    "ignore",
    message=r"X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
    module=r"sklearn\.utils\.validation",
)


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fit_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    model_type: str,
    lightgbm_n_jobs: int | None = None,
    rf_n_estimators: int = 500,
    rf_n_jobs: int = -1,
) -> Any:
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train)
    if model_type == "logistic_regression":
        clf = LogisticRegression(
            solver="lbfgs",
            class_weight="balanced",
            max_iter=5000,
            random_state=seed,
        )
    elif model_type == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "lightgbm is not installed. Install it first, then rerun with "
                "`model_type=lightgbm` or `--model lightgbm`."
            ) from exc
        clf = LGBMClassifier(
            objective="binary",
            class_weight="balanced",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1 if lightgbm_n_jobs is None else int(lightgbm_n_jobs),
            verbosity=-1,
        )
    elif model_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=int(rf_n_estimators),
            random_state=seed,
            n_jobs=int(rf_n_jobs),
            class_weight=None,
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    clf.fit(X_train, y_train)
    return clf


def positive_proba(clf: Any, X: np.ndarray) -> np.ndarray:
    proba = clf.predict_proba(X)
    if proba.ndim != 2 or proba.shape[1] < 2:
        raise ValueError("Classifier must expose binary predict_proba output.")
    return proba[:, 1]


def compute_mix_at_k_values(X: np.ndarray, y: np.ndarray, k_values: Sequence[int]) -> Dict[int, float]:
    if not k_values:
        return {}
    if len(X) <= 1:
        return {int(k): 0.0 for k in k_values}
    max_k = max(int(k) for k in k_values)
    effective_max_k = max(1, min(max_k, len(X) - 1))
    nn = NearestNeighbors(n_neighbors=effective_max_k + 1, metric="euclidean")
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    neighbor_labels = y[indices[:, 1:]]
    return {
        int(k): float((neighbor_labels[:, : max(1, min(int(k), len(X) - 1))] != y[:, None]).mean())
        for k in k_values
    }


def js_divergence_feature_mass(X_raw: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    a = np.asarray(X_raw, dtype=float)
    a = np.clip(a, 0.0, None)

    c0 = a[y == 0].sum(axis=0)
    c1 = a[y == 1].sum(axis=0)

    p = c0 + eps
    q = c1 + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    kl_pm = float(np.sum(p * (np.log(p) - np.log(m))))
    kl_qm = float(np.sum(q * (np.log(q) - np.log(m))))
    return 0.5 * (kl_pm + kl_qm)


def safe_silhouette(X: np.ndarray, y: np.ndarray) -> float:
    if len(X) < 3 or np.unique(y).size < 2:
        return float("nan")
    try:
        return float(silhouette_score(X, y, metric="euclidean"))
    except ValueError:
        return float("nan")


def fit_train_scaler(X_train_raw: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    X_train = stabilize_features(X_train_raw)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = np.clip(np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0), -100.0, 100.0)
    return X_train_scaled, scaler


def transform_with_scaler(X_raw: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    X = stabilize_features(X_raw)
    X_scaled = scaler.transform(X)
    return np.clip(np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0), -100.0, 100.0)


def class_counts(samples: Sequence) -> Dict[str, int]:
    return {
        "n_samples": len(samples),
        "n_benign": int(sum(sample.label == 0 for sample in samples)),
        "n_malware": int(sum(sample.label == 1 for sample in samples)),
    }


def prepare_group_model(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    model_type: str,
    lightgbm_n_jobs: int | None = None,
    rf_n_estimators: int = 500,
    rf_n_jobs: int = -1,
) -> Tuple[Any, np.ndarray, StandardScaler]:
    X_train, scaler = fit_train_scaler(X_train_raw)
    clf = fit_classifier(
        X_train,
        y_train,
        seed,
        model_type=model_type,
        lightgbm_n_jobs=lightgbm_n_jobs,
        rf_n_estimators=rf_n_estimators,
        rf_n_jobs=rf_n_jobs,
    )
    return clf, X_train, scaler


def evaluate_group_with_model(
    clf: Any,
    scaler: StandardScaler,
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    X_test = transform_with_scaler(X_test_raw, scaler)
    proba = positive_proba(clf, X_test)
    pred = (proba >= 0.5).astype(int)
    score_std = float(np.std(proba))
    if np.unique(y_test).size < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y_test, proba))
    return {
        "acc": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "auc": auc,
        "positive_rate": float(pred.mean()),
        "score_std": score_std,
        "degenerate_prediction": float(score_std < 1e-8),
        "silhouette": safe_silhouette(X_test, y_test),
        "js_divergence": js_divergence_feature_mass(X_test_raw, y_test),
    }, X_test, y_test


def build_run_result(
    seed: int,
    mix_k: int,
    week: str,
    test_samples: List,
    base_metrics: Dict[str, Dict[str, float]],
    mix_values_by_group: Dict[str, Dict[int, float]],
    feature_groups: Sequence[str],
) -> Dict[str, object]:
    per_group: Dict[str, Dict[str, float]] = {}
    for group in feature_groups:
        metrics = dict(base_metrics[group])
        metrics["mix_at_k"] = mix_values_by_group[group][mix_k]
        per_group[group] = metrics

    auc_ranks = feature_rankings({group: metrics["auc"] for group, metrics in per_group.items()})
    acc_ranks = feature_rankings({group: metrics["acc"] for group, metrics in per_group.items()})
    mix_ranks = feature_rankings({group: metrics["mix_at_k"] for group, metrics in per_group.items()}, higher_is_better=False)
    silhouette_ranks = feature_rankings({group: metrics["silhouette"] for group, metrics in per_group.items()})
    js_ranks = feature_rankings({group: metrics["js_divergence"] for group, metrics in per_group.items()})

    return {
        "seed": seed,
        "mix_k": mix_k,
        "test_week": week,
        "n_test_samples": len(test_samples),
        **class_counts(test_samples),
        "per_group": per_group,
        "auc_rank": auc_ranks,
        "acc_rank": acc_ranks,
        "mix_at_k_rank": mix_ranks,
        "silhouette_rank": silhouette_ranks,
        "js_divergence_rank": js_ranks,
    }


def summarize_results(
    run_results: List[Dict[str, object]],
    feature_groups: Sequence[str],
) -> Tuple[Dict[str, Dict[str, dict]], Dict[str, Dict[str, dict]]]:
    aggregate_metrics: Dict[str, Dict[str, dict]] = {}
    aggregate_ranks: Dict[str, Dict[str, dict]] = {
        "auc_rank": {},
        "acc_rank": {},
        "mix_at_k_rank": {},
        "silhouette_rank": {},
        "js_divergence_rank": {},
    }
    for group in feature_groups:
        aggregate_metrics[group] = {
            metric: summarize([run["per_group"][group][metric] for run in run_results])
            for metric in [
                "acc",
                "f1",
                "auc",
                "positive_rate",
                "score_std",
                "degenerate_prediction",
                "mix_at_k",
                "silhouette",
                "js_divergence",
            ]
        }
        aggregate_ranks["auc_rank"][group] = summarize([run["auc_rank"][group] for run in run_results])
        aggregate_ranks["acc_rank"][group] = summarize([run["acc_rank"][group] for run in run_results])
        aggregate_ranks["mix_at_k_rank"][group] = summarize([run["mix_at_k_rank"][group] for run in run_results])
        aggregate_ranks["silhouette_rank"][group] = summarize([run["silhouette_rank"][group] for run in run_results])
        aggregate_ranks["js_divergence_rank"][group] = summarize([run["js_divergence_rank"][group] for run in run_results])
    return aggregate_metrics, aggregate_ranks


def summarize_results_by_key(
    run_results: List[Dict[str, object]],
    key: str,
    feature_groups: Sequence[str],
) -> Dict[str, Dict[str, Dict[str, dict]]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for run in run_results:
        grouped[str(run[key])] = grouped.setdefault(str(run[key]), []) + [run]

    out: Dict[str, Dict[str, Dict[str, dict]]] = {}
    for key_value, rows in grouped.items():
        metrics, ranks = summarize_results(rows, feature_groups)
        out[key_value] = {"aggregate_metrics": metrics, "aggregate_ranks": ranks}
    return out


def feature_views_by_group(samples: Sequence, feature_groups: Sequence[str]) -> Dict[str, np.ndarray]:
    return {
        group: select_feature_group(samples, group)
        for group in feature_groups
    }


def run_seed_experiment(
    seed: int,
    all_train_samples: Sequence,
    test_weeks: Sequence[str],
    test_samples_by_week: Dict[str, List],
    cached_unbalanced_test_views_by_week: Dict[str, Dict[str, np.ndarray]],
    cached_unbalanced_test_labels_by_week: Dict[str, np.ndarray],
    balance_train: bool,
    balance_test: bool,
    max_train_per_class: int | None,
    max_test_per_class: int | None,
    model_type: str,
    lightgbm_n_jobs: int | None,
    rf_n_estimators: int,
    rf_n_jobs: int,
    mix_k_values: Sequence[int],
    feature_groups: Sequence[str],
) -> Dict[str, object]:
    train_samples = list(all_train_samples)
    if balance_train:
        train_samples = balance_samples(all_train_samples, seed=seed, max_per_class=max_train_per_class)
    train_sample_count = class_counts(train_samples)
    y_train = labels_array(train_samples)
    train_features_by_group = feature_views_by_group(train_samples, feature_groups)
    prepared_models_by_group: Dict[str, Tuple[Any, np.ndarray, StandardScaler]] = {}
    for group in feature_groups:
        prepared_models_by_group[group] = prepare_group_model(
            train_features_by_group[group],
            y_train,
            seed=seed,
            model_type=model_type,
            lightgbm_n_jobs=lightgbm_n_jobs,
            rf_n_estimators=rf_n_estimators,
            rf_n_jobs=rf_n_jobs,
        )

    run_results: List[Dict[str, object]] = []
    test_counts_by_week: Dict[str, Dict[str, int]] = {}
    for week_idx, week in enumerate(test_weeks):
        test_samples = test_samples_by_week[week]
        if balance_test:
            test_samples = balance_samples(
                test_samples,
                seed=seed + week_idx + 1,
                max_per_class=max_test_per_class,
            )
        test_counts_by_week[week] = class_counts(test_samples)

        base_metrics: Dict[str, Dict[str, float]] = {}
        mix_values_by_group: Dict[str, Dict[int, float]] = {}
        if balance_test:
            y_test = labels_array(test_samples)
            test_features_by_group = feature_views_by_group(test_samples, feature_groups)
        else:
            y_test = cached_unbalanced_test_labels_by_week[week]
            test_features_by_group = cached_unbalanced_test_views_by_week[week]
        for group in feature_groups:
            X_test_raw = test_features_by_group[group]
            clf, _, scaler = prepared_models_by_group[group]
            metrics, X_test, _ = evaluate_group_with_model(
                clf,
                scaler,
                X_test_raw,
                y_test,
            )
            base_metrics[group] = metrics
            mix_values_by_group[group] = compute_mix_at_k_values(X_test, y_test, mix_k_values)

        for mix_k in mix_k_values:
            run_results.append(
                build_run_result(
                    seed=seed,
                    mix_k=mix_k,
                    week=week,
                    test_samples=test_samples,
                    base_metrics=base_metrics,
                    mix_values_by_group=mix_values_by_group,
                    feature_groups=feature_groups,
                )
            )

    return {
        "seed": seed,
        "run_results": run_results,
        "train_sample_count": train_sample_count,
        "test_counts_by_week": test_counts_by_week,
    }


def run_from_config(config: dict) -> dict:
    data_root = Path(config["data_root"])
    platform = config["platform"]
    train_weeks = config["train_weeks"]
    test_weeks = config["test_weeks"]
    packer_filter = str(config.get("packer_filter", "all"))
    model_type = str(config.get("model_type", "logistic_regression"))
    lightgbm_n_jobs = config.get("lightgbm_n_jobs")
    seeds = list(config.get("seeds", [config.get("seed", 42)]))
    seeds = [int(seed) for seed in seeds]
    balance_train = bool(config.get("balance_train", True))
    balance_test = bool(config.get("balance_test", False))
    max_train_per_class = config.get("max_train_per_class")
    max_test_per_class = config.get("max_test_per_class")
    mix_k_values = list(config.get("mix_k_values", [config.get("mix_k", 10)]))
    mix_k_values = [int(k) for k in mix_k_values]
    requested_groups = list(config.get("feature_groups", FEATURE_GROUPS))
    feature_groups = [group for group in FEATURE_GROUPS if group in requested_groups]
    if not feature_groups:
        raise ValueError("feature_groups must contain at least one supported feature group.")
    num_workers = int(config.get("num_workers", 1))
    show_progress = bool(config.get("show_progress", True))
    rf_n_estimators = int(config.get("rf_n_estimators", 500))
    rf_n_jobs = int(config.get("rf_n_jobs", -1))

    train_paths = week_paths(data_root, platform, train_weeks, "train")
    all_train_samples = load_samples_from_paths(train_paths, packer_filter=packer_filter)
    test_samples_by_week: Dict[str, List] = {}
    week_iter = tqdm(test_weeks, desc="Loading test weeks", leave=False) if show_progress else test_weeks
    for week in week_iter:
        test_path = week_paths(data_root, platform, [week], "test")[0]
        test_samples_by_week[week] = load_samples_from_jsonl(test_path, packer_filter=packer_filter)
    cached_unbalanced_test_views_by_week: Dict[str, Dict[str, np.ndarray]] = {}
    cached_unbalanced_test_labels_by_week: Dict[str, np.ndarray] = {}
    if not balance_test:
        for week, samples in test_samples_by_week.items():
            cached_unbalanced_test_views_by_week[week] = feature_views_by_group(samples, feature_groups)
            cached_unbalanced_test_labels_by_week[week] = labels_array(samples)

    run_results: List[Dict[str, object]] = []
    train_sample_counts: Dict[int, Dict[str, int]] = {}
    test_sample_counts_by_week: Dict[str, Dict[str, int]] = {}
    if num_workers <= 1:
        seed_iter = tqdm(seeds, desc="Seeds") if show_progress else seeds
        seed_payloads = [
            run_seed_experiment(
                seed=seed,
                all_train_samples=all_train_samples,
                test_weeks=test_weeks,
                test_samples_by_week=test_samples_by_week,
                cached_unbalanced_test_views_by_week=cached_unbalanced_test_views_by_week,
                cached_unbalanced_test_labels_by_week=cached_unbalanced_test_labels_by_week,
                balance_train=balance_train,
                balance_test=balance_test,
                max_train_per_class=max_train_per_class,
                max_test_per_class=max_test_per_class,
                model_type=model_type,
                lightgbm_n_jobs=lightgbm_n_jobs,
                rf_n_estimators=rf_n_estimators,
                rf_n_jobs=rf_n_jobs,
                mix_k_values=mix_k_values,
                feature_groups=feature_groups,
            )
            for seed in seed_iter
        ]
    else:
        mp_ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=mp_ctx) as executor:
            seed_payloads = list(
                tqdm(
                    executor.map(
                        run_seed_experiment,
                        seeds,
                        [all_train_samples] * len(seeds),
                        [test_weeks] * len(seeds),
                        [test_samples_by_week] * len(seeds),
                        [cached_unbalanced_test_views_by_week] * len(seeds),
                        [cached_unbalanced_test_labels_by_week] * len(seeds),
                        [balance_train] * len(seeds),
                        [balance_test] * len(seeds),
                        [max_train_per_class] * len(seeds),
                        [max_test_per_class] * len(seeds),
                        [model_type] * len(seeds),
                        [lightgbm_n_jobs] * len(seeds),
                        [rf_n_estimators] * len(seeds),
                        [rf_n_jobs] * len(seeds),
                        [mix_k_values] * len(seeds),
                        [feature_groups] * len(seeds),
                    ),
                    total=len(seeds),
                    desc="Seeds",
                    disable=not show_progress,
                )
            )

    seed_payloads.sort(key=lambda payload: seeds.index(payload["seed"]))
    for payload in seed_payloads:
        seed = int(payload["seed"])
        train_sample_counts[seed] = payload["train_sample_count"]
        for week in test_weeks:
            test_sample_counts_by_week.setdefault(week, payload["test_counts_by_week"][week])
        run_results.extend(payload["run_results"])

    aggregate_metrics, aggregate_ranks = summarize_results(run_results, feature_groups)
    aggregate_by_seed = summarize_results_by_key(run_results, "seed", feature_groups)
    aggregate_by_mix_k = summarize_results_by_key(run_results, "mix_k", feature_groups)

    return {
        "config": {
            **config,
            "resolved_train_paths": [str(path) for path in train_paths],
            "packer_filter": packer_filter,
            "model_type": model_type,
            "lightgbm_n_jobs": lightgbm_n_jobs,
            "rf_n_estimators": rf_n_estimators,
            "rf_n_jobs": rf_n_jobs,
            "feature_groups": feature_groups,
            "seeds": seeds,
            "mix_k_values": mix_k_values,
            "num_workers": num_workers,
            "show_progress": show_progress,
            "train_sample_counts_by_seed": train_sample_counts,
            "test_sample_counts_by_week": test_sample_counts_by_week,
            "question": (
                "Does the relative separability pattern across feature subsets remain stable "
                "when the training set is pooled across multiple Win32 weeks and evaluation is repeated "
                "across test weeks, sampling seeds, and mix@k settings?"
            ),
        },
        "run_results": run_results,
        "aggregate_metrics": aggregate_metrics,
        "aggregate_ranks": aggregate_ranks,
        "aggregate_by_seed": aggregate_by_seed,
        "aggregate_by_mix_k": aggregate_by_mix_k,
    }


def weekly_rows(results: dict) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    feature_groups = results["config"].get("feature_groups", FEATURE_GROUPS)
    for week in results["run_results"]:
        row: Dict[str, object] = {
            "seed": week["seed"],
            "mix_k": week["mix_k"],
            "test_week": week["test_week"],
            "n_test_samples": week["n_test_samples"],
            "n_benign": week["n_benign"],
            "n_malware": week["n_malware"],
        }
        for group in feature_groups:
            metrics = week["per_group"][group]
            row[f"{group}_acc"] = metrics["acc"]
            row[f"{group}_f1"] = metrics["f1"]
            row[f"{group}_auc"] = metrics["auc"]
            row[f"{group}_positive_rate"] = metrics["positive_rate"]
            row[f"{group}_score_std"] = metrics["score_std"]
            row[f"{group}_degenerate_prediction"] = metrics["degenerate_prediction"]
            row[f"{group}_mix_at_k"] = metrics["mix_at_k"]
            row[f"{group}_silhouette"] = metrics["silhouette"]
            row[f"{group}_js_divergence"] = metrics["js_divergence"]
            row[f"{group}_auc_rank"] = week["auc_rank"][group]
            row[f"{group}_acc_rank"] = week["acc_rank"][group]
            row[f"{group}_mix_at_k_rank"] = week["mix_at_k_rank"][group]
            row[f"{group}_silhouette_rank"] = week["silhouette_rank"][group]
            row[f"{group}_js_divergence_rank"] = week["js_divergence_rank"][group]
        rows.append(row)
    return rows


def aggregate_rows(results: dict) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for group in results["config"].get("feature_groups", FEATURE_GROUPS):
        for metric, stats in results["aggregate_metrics"][group].items():
            rows.append(
                {
                    "scope": "metric",
                    "feature_group": group,
                    "metric": metric,
                    **stats,
                }
            )
        for metric, group_stats in results["aggregate_ranks"].items():
            rows.append(
                {
                    "scope": "rank",
                    "feature_group": group,
                    "metric": metric,
                    **group_stats[group],
                }
            )
    return rows


def aggregate_rows_by_grouping(grouped_results: Dict[str, Dict[str, Dict[str, dict]]], grouping_name: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    feature_groups = FEATURE_GROUPS
    if grouped_results:
        first_payload = next(iter(grouped_results.values()))
        feature_groups = list(first_payload["aggregate_metrics"].keys())
    for grouping_value, payload in grouped_results.items():
        for group in feature_groups:
            for metric, stats in payload["aggregate_metrics"][group].items():
                rows.append(
                    {
                        grouping_name: grouping_value,
                        "scope": "metric",
                        "feature_group": group,
                        "metric": metric,
                        **stats,
                    }
                )
            for rank_metric, rank_payload in payload["aggregate_ranks"].items():
                rows.append(
                    {
                        grouping_name: grouping_value,
                        "scope": "rank",
                        "feature_group": group,
                        "metric": rank_metric,
                        **rank_payload[group],
                    }
                )
    return rows


def print_summary(results: dict) -> None:
    print("RQ1 feature subset generalization")
    print(f"  model         = {results['config'].get('model_type', 'logistic_regression')}")
    print(f"  seeds         = {len(results['config']['seeds'])}")
    print(f"  mix@k values  = {results['config']['mix_k_values']}")
    print(f"  test weeks    = {len(results['config']['test_weeks'])}")
    print(f"  total runs    = {len(results['run_results'])}")
    print("")
    print("Aggregate AUC by feature group")
    for group in results["config"].get("feature_groups", FEATURE_GROUPS):
        auc_stats = results["aggregate_metrics"][group]["auc"]
        rank_stats = results["aggregate_ranks"]["auc_rank"][group]
        mix_stats = results["aggregate_metrics"][group]["mix_at_k"]
        score_std_stats = results["aggregate_metrics"][group]["score_std"]
        silhouette_stats = results["aggregate_metrics"][group]["silhouette"]
        js_stats = results["aggregate_metrics"][group]["js_divergence"]
        print(
            f"  {group:8s} AUC={auc_stats['mean']:.4f} +/- {auc_stats['std']:.4f} "
            f"mean_rank={rank_stats['mean']:.2f} "
            f"mix@k={mix_stats['mean']:.4f} "
            f"score_std={score_std_stats['mean']:.4f} "
            f"sil={silhouette_stats['mean']:.4f} "
            f"JS={js_stats['mean']:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RQ1 feature-subset generalization experiments across multiple weeks.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", choices=["logistic_regression", "lightgbm", "random_forest"], help="Override config model_type.")
    parser.add_argument("--balance-test", action="store_true", help="Override config to balance test samples 1:1.")
    parser.add_argument("--max-test-per-class", type=int, help="Override config max_test_per_class.")
    parser.add_argument("--num-workers", type=int, help="Override config num_workers.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    data_root = Path(config["data_root"])
    if not data_root.is_absolute():
        config["data_root"] = str((args.config.parent / data_root).resolve())
    if args.balance_test:
        config["balance_test"] = True
    if args.max_test_per_class is not None:
        config["max_test_per_class"] = int(args.max_test_per_class)
    if args.model is not None:
        config["model_type"] = args.model
    if args.num_workers is not None:
        config["num_workers"] = int(args.num_workers)
    if args.no_progress:
        config["show_progress"] = False
    results = run_from_config(config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    weekly = weekly_rows(results)
    aggregate = aggregate_rows(results)
    by_seed = aggregate_rows_by_grouping(results["aggregate_by_seed"], "seed")
    by_mix_k = aggregate_rows_by_grouping(results["aggregate_by_mix_k"], "mix_k")

    if weekly:
        write_csv(args.output_dir / "weekly_results.csv", list(weekly[0].keys()), weekly)
    write_csv(
        args.output_dir / "aggregate_summary.csv",
        ["scope", "feature_group", "metric", "mean", "std", "min", "max"],
        aggregate,
    )
    write_csv(
        args.output_dir / "aggregate_by_seed.csv",
        ["seed", "scope", "feature_group", "metric", "mean", "std", "min", "max"],
        by_seed,
    )
    write_csv(
        args.output_dir / "aggregate_by_mix_k.csv",
        ["mix_k", "scope", "feature_group", "metric", "mean", "std", "min", "max"],
        by_mix_k,
    )
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print_summary(results)
    print("")
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
