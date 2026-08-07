#!/usr/bin/env python3
"""
RQ2-3: Prediction Flip Analysis under Feature-Level Perturbation
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "artifacts" / "shared"))

from common import (  # type: ignore
    balance_samples,
    labels_array,
    load_samples_from_jsonl,
    load_samples_from_paths,
    select_feature_group,
    summarize,
    week_paths,
)

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - optional convenience dependency
    tqdm = None


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def log_progress(message: str) -> None:
    print(message, flush=True)


def stabilize_features(X: np.ndarray) -> np.ndarray:
    X = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -1e12, 1e12)
    return np.sign(X) * np.log1p(np.abs(X))


def fit_scaler(X_train_raw: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(stabilize_features(X_train_raw))
    return scaler


def transform_with_scaler(scaler: StandardScaler, X_raw: np.ndarray) -> np.ndarray:
    X_scaled = scaler.transform(stabilize_features(X_raw))
    return np.clip(np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0), -100.0, 100.0)


def zero_scaled_template(scaler: StandardScaler, n_features: int) -> np.ndarray:
    zero_row = np.zeros((1, n_features), dtype=np.float64)
    return transform_with_scaler(scaler, zero_row)[0]


def fit_classifier(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    model_type: str,
    lgbm_n_jobs: int,
    rf_n_estimators: int,
    rf_n_jobs: int,
) -> Any:
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
                "`model_type=lightgbm`."
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
            n_jobs=lgbm_n_jobs,
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

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        clf.fit(np.asarray(X_train_scaled, dtype=np.float64), np.asarray(y_train))
    return clf


def positive_proba(clf: Any, X: np.ndarray) -> np.ndarray:
    proba = clf.predict_proba(np.asarray(X, dtype=np.float64))
    if proba.ndim != 2 or proba.shape[1] < 2:
        raise ValueError("Classifier must expose binary predict_proba output.")
    return proba[:, 1]


def configure_thread_env(omp_num_threads: int) -> None:
    thread_value = str(omp_num_threads)
    os.environ["OMP_NUM_THREADS"] = thread_value
    os.environ["OPENBLAS_NUM_THREADS"] = thread_value
    os.environ["MKL_NUM_THREADS"] = thread_value
    os.environ["NUMEXPR_NUM_THREADS"] = thread_value
    os.environ["VECLIB_MAXIMUM_THREADS"] = thread_value


def select_top_indices(coef_abs: np.ndarray, strength: float) -> np.ndarray:
    n_features = len(coef_abs)
    n_mask = max(1, math.ceil(n_features * strength))
    ranked = np.argsort(-coef_abs, kind="stable")
    return ranked[:n_mask]


def split_train_validation_indices(
    *,
    n_samples: int,
    y_train: np.ndarray,
    seed: int,
    validation_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(n_samples)
    train_idx, val_idx = train_test_split(
        indices,
        test_size=validation_fraction,
        random_state=seed,
        stratify=y_train,
    )
    return np.asarray(train_idx), np.asarray(val_idx)


def compute_feature_importance(
    *,
    clf: Any,
    model_type: str,
    importance_method: str,
    X_val_scaled: np.ndarray | None,
    y_val: np.ndarray | None,
    seed: int,
    permutation_repeats: int,
) -> np.ndarray:
    if model_type == "logistic_regression":
        return np.abs(np.asarray(clf.coef_, dtype=np.float64).ravel())

    if model_type == "random_forest":
        if importance_method != "permutation":
            raise ValueError(
                "Random Forest in RQ2-3 currently supports importance_method='permutation' only."
            )
        if X_val_scaled is None or y_val is None:
            raise ValueError("Permutation importance requires a validation split.")
        result = permutation_importance(
            clf,
            np.asarray(X_val_scaled, dtype=np.float64),
            np.asarray(y_val),
            scoring="roc_auc",
            n_repeats=permutation_repeats,
            random_state=seed,
            n_jobs=1,
        )
        return np.asarray(result.importances_mean, dtype=np.float64)

    if model_type != "lightgbm":
        raise ValueError(f"Unsupported model_type: {model_type}")

    if importance_method == "permutation":
        if X_val_scaled is None or y_val is None:
            raise ValueError("Permutation importance requires a validation split.")
        result = permutation_importance(
            clf,
            np.asarray(X_val_scaled, dtype=np.float64),
            np.asarray(y_val),
            scoring="roc_auc",
            n_repeats=permutation_repeats,
            random_state=seed,
            n_jobs=1,
        )
        return np.asarray(result.importances_mean, dtype=np.float64)

    if importance_method == "gain":
        booster = getattr(clf, "booster_", None)
        if booster is not None:
            return np.asarray(booster.feature_importance(importance_type="gain"), dtype=np.float64)
        return np.asarray(getattr(clf, "feature_importances_"), dtype=np.float64)

    if importance_method == "shap":
        raise NotImplementedError(
            "importance_method='shap' is intentionally not implemented here yet. "
            "The current RQ2-3 LightGBM extension supports 'permutation' "
            "(default) and 'gain'."
        )

    raise ValueError(
        "Unsupported importance_method for LightGBM: "
        f"{importance_method}. Use 'permutation' or 'gain'."
    )


def apply_zero_mask(X: np.ndarray, indices: np.ndarray) -> np.ndarray:
    masked = np.array(X, copy=True)
    masked[:, indices] = 0.0
    return masked


def apply_zero_mask_scaled(
    X_scaled: np.ndarray,
    indices: np.ndarray,
    zero_scaled_values: np.ndarray,
) -> np.ndarray:
    masked = np.array(X_scaled, copy=True)
    masked[:, indices] = zero_scaled_values[indices]
    return masked


def trial_sort_key(row: Dict[str, object]) -> tuple:
    return (
        int(row["seed"]),
        str(row["feature_group"]),
        str(row["test_week"]),
        str(row["perturbation_type"]),
        float(row["strength"]),
        int(row["trial_id"]),
    )


def evaluate_flip_metrics(
    baseline_proba: np.ndarray,
    perturbed_proba: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    baseline_pred = (baseline_proba >= threshold).astype(int)
    perturbed_pred = (perturbed_proba >= threshold).astype(int)
    flips = baseline_pred != perturbed_pred

    benign_mask = y_true == 0
    malware_mask = y_true == 1

    benign_to_malware = flips & benign_mask & (perturbed_pred == 1)
    malware_to_benign = flips & malware_mask & (perturbed_pred == 0)

    baseline_margin = np.abs(baseline_proba - threshold)
    perturbed_margin = np.abs(perturbed_proba - threshold)

    return {
        "flip_rate": float(flips.mean()),
        "benign_to_malware_rate": float(benign_to_malware.mean()),
        "malware_to_benign_rate": float(malware_to_benign.mean()),
        "mean_abs_probability_shift": float(np.mean(np.abs(perturbed_proba - baseline_proba))),
        "mean_signed_probability_shift": float(np.mean(perturbed_proba - baseline_proba)),
        "mean_margin_shift": float(np.mean(perturbed_margin - baseline_margin)),
    }


def aggregate_rows(trial_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple, List[Dict[str, object]]] = {}
    for row in trial_rows:
        key = (
            row["feature_group"],
            row["test_week"],
            row["perturbation_type"],
            row["strength"],
        )
        grouped.setdefault(key, []).append(row)

    metric_names = [
        "flip_rate",
        "benign_to_malware_rate",
        "malware_to_benign_rate",
        "mean_abs_probability_shift",
        "mean_signed_probability_shift",
        "mean_margin_shift",
    ]

    aggregate: List[Dict[str, object]] = []
    for key, rows in sorted(grouped.items()):
        feature_group, test_week, perturbation_type, strength = key
        out: Dict[str, object] = {
            "feature_group": feature_group,
            "test_week": test_week,
            "perturbation_type": perturbation_type,
            "strength": strength,
            "n_trials": len(rows),
            "n_masked_features": rows[0]["n_masked_features"],
        }
        for metric in metric_names:
            stats = summarize([float(row[metric]) for row in rows])
            out[f"{metric}_mean"] = stats["mean"]
            out[f"{metric}_std"] = stats["std"]
            out[f"{metric}_min"] = stats["min"]
            out[f"{metric}_max"] = stats["max"]
        aggregate.append(out)
    return aggregate


def process_test_week(
    *,
    seed: int,
    group: str,
    week_idx: int,
    week: str,
    test_samples: List[object],
    scaler: StandardScaler,
    clf: Any,
    importance_scores: np.ndarray,
    zero_scaled_values: np.ndarray,
    strengths: List[float],
    random_trials: int,
    threshold: float,
) -> List[Dict[str, object]]:
    y_test = labels_array(test_samples)
    X_test_raw = select_feature_group(test_samples, group)
    baseline_scaled = transform_with_scaler(scaler, X_test_raw)
    baseline_proba = positive_proba(clf, baseline_scaled)

    rows: List[Dict[str, object]] = []
    baseline_metrics = evaluate_flip_metrics(baseline_proba, baseline_proba, y_test, threshold)
    rows.append(
        {
            "seed": seed,
            "feature_group": group,
            "test_week": week,
            "perturbation_type": "baseline",
            "strength": 0.0,
            "trial_id": 0,
            "n_masked_features": 0,
            **baseline_metrics,
        }
    )

    for strength in strengths:
        important_indices = select_top_indices(importance_scores, strength)
        important_scaled = apply_zero_mask_scaled(
            baseline_scaled,
            important_indices,
            zero_scaled_values,
        )
        important_proba = positive_proba(clf, important_scaled)
        important_metrics = evaluate_flip_metrics(baseline_proba, important_proba, y_test, threshold)
        rows.append(
            {
                "seed": seed,
                "feature_group": group,
                "test_week": week,
                "perturbation_type": "important",
                "strength": strength,
                "trial_id": 0,
                "n_masked_features": len(important_indices),
                **important_metrics,
            }
        )

        n_mask = len(important_indices)
        group_offset = sum(ord(ch) for ch in group)
        for trial_id in range(1, random_trials + 1):
            rng = np.random.default_rng(seed * 10_000 + week_idx * 1_000 + group_offset + trial_id)
            random_indices = rng.choice(len(importance_scores), size=n_mask, replace=False)
            random_scaled = apply_zero_mask_scaled(
                baseline_scaled,
                random_indices,
                zero_scaled_values,
            )
            random_proba = positive_proba(clf, random_scaled)
            random_metrics = evaluate_flip_metrics(baseline_proba, random_proba, y_test, threshold)
            rows.append(
                {
                    "seed": seed,
                    "feature_group": group,
                    "test_week": week,
                    "perturbation_type": "random",
                    "strength": strength,
                    "trial_id": trial_id,
                    "n_masked_features": n_mask,
                    **random_metrics,
                }
            )

    return rows


def run_from_config(config: dict) -> dict:
    data_root = Path(config["data_root"])
    platform = config["platform"]
    packer_filter = str(config.get("packer_filter", "all"))
    train_weeks = config["train_weeks"]
    test_weeks = config["test_weeks"]
    seeds = [int(seed) for seed in config["seeds"]]
    feature_groups = list(config["feature_groups"])
    strengths = [float(value) for value in config["perturbation_strengths"]]
    random_trials = int(config["random_trials"])
    threshold = float(config.get("decision_threshold", 0.5))
    balance_train = bool(config.get("balance_train", True))
    balance_test = bool(config.get("balance_test", False))
    max_train_per_class = config.get("max_train_per_class")
    max_test_per_class = config.get("max_test_per_class")
    model_type = str(config.get("model_type", "logistic_regression"))
    rf_n_estimators = int(config.get("rf_n_estimators", 500))
    if model_type == "lightgbm":
        num_workers = int(config.get("num_workers", 2))
        lgbm_n_jobs = int(config.get("lgbm_n_jobs", 2))
        omp_num_threads = int(config.get("omp_num_threads", 1))
        rf_n_jobs = int(config.get("rf_n_jobs", 1))
    elif model_type == "random_forest":
        num_workers = int(config.get("num_workers", 1))
        lgbm_n_jobs = int(config.get("lgbm_n_jobs", 1))
        omp_num_threads = int(config.get("omp_num_threads", 1))
        rf_n_jobs = int(config.get("rf_n_jobs", -1))
    else:
        num_workers = int(config.get("num_workers", max(1, min(4, os.cpu_count() or 1))))
        lgbm_n_jobs = int(config.get("lgbm_n_jobs", 1))
        omp_num_threads = int(config.get("omp_num_threads", 1))
        rf_n_jobs = int(config.get("rf_n_jobs", 1))
    importance_method = str(
        config.get(
            "importance_method",
            "coefficient" if model_type == "logistic_regression" else "permutation",
        )
    )
    validation_fraction = float(config.get("validation_fraction", 0.2))
    permutation_repeats = int(config.get("permutation_repeats", 3))
    needs_validation_split = (
        model_type in {"lightgbm", "random_forest"} and importance_method == "permutation"
    )
    if model_type == "logistic_regression" and importance_method != "coefficient":
        raise ValueError(
            "Logistic Regression in RQ2-3 currently requires "
            "importance_method='coefficient'."
        )
    if model_type == "random_forest" and importance_method != "permutation":
        raise ValueError(
            "Random Forest in RQ2-3 currently requires importance_method='permutation'."
        )
    if model_type == "lightgbm" and importance_method not in {"permutation", "gain", "shap"}:
        raise ValueError(
            "LightGBM in RQ2-3 supports importance_method in "
            "{'permutation', 'gain', 'shap'}."
        )
    if num_workers < 1 or lgbm_n_jobs < 1 or omp_num_threads < 1:
        raise ValueError("num_workers, lgbm_n_jobs, and omp_num_threads must all be >= 1.")
    if not (0.0 < validation_fraction < 1.0):
        raise ValueError("validation_fraction must be between 0 and 1.")
    if model_type == "lightgbm":
        configure_thread_env(omp_num_threads)

    train_paths = week_paths(data_root, platform, train_weeks, "train")
    all_train_samples = load_samples_from_paths(train_paths, packer_filter=packer_filter)
    test_samples_by_week = {
        week: load_samples_from_jsonl(
            week_paths(data_root, platform, [week], "test")[0],
            packer_filter=packer_filter,
        )
        for week in test_weeks
    }

    trial_rows: List[Dict[str, object]] = []

    total_jobs = len(seeds) * len(feature_groups)
    job_counter = 0

    for seed in seeds:
        train_samples = all_train_samples
        if balance_train:
            train_samples = balance_samples(all_train_samples, seed=seed, max_per_class=max_train_per_class)
        y_train = labels_array(train_samples)
        train_idx = np.arange(len(train_samples))
        val_idx = np.array([], dtype=int)
        if needs_validation_split:
            train_idx, val_idx = split_train_validation_indices(
                n_samples=len(train_samples),
                y_train=y_train,
                seed=seed,
                validation_fraction=validation_fraction,
            )

        for group in feature_groups:
            job_counter += 1
            log_progress(
                f"[RQ2-3] ({job_counter}/{total_jobs}) "
                f"seed={seed} group={group} | fitting {model_type} "
                f"(importance={importance_method})"
            )
            X_train_raw = select_feature_group(train_samples, group)
            if needs_validation_split:
                X_train_fit_raw = X_train_raw[train_idx]
                y_train_fit = y_train[train_idx]
                X_val_raw = X_train_raw[val_idx]
                y_val = y_train[val_idx]
            else:
                X_train_fit_raw = X_train_raw
                y_train_fit = y_train
                X_val_raw = None
                y_val = None

            scaler = fit_scaler(X_train_fit_raw)
            X_train_scaled = transform_with_scaler(scaler, X_train_fit_raw)
            clf = fit_classifier(
                X_train_scaled,
                y_train_fit,
                seed=seed,
                model_type=model_type,
                lgbm_n_jobs=lgbm_n_jobs,
                rf_n_estimators=rf_n_estimators,
                rf_n_jobs=rf_n_jobs,
            )
            X_val_scaled = transform_with_scaler(scaler, X_val_raw) if X_val_raw is not None else None
            importance_scores = compute_feature_importance(
                clf=clf,
                model_type=model_type,
                importance_method=importance_method,
                X_val_scaled=X_val_scaled,
                y_val=y_val,
                seed=seed,
                permutation_repeats=permutation_repeats,
            )
            log_progress(
                f"[RQ2-3] seed={seed} group={group} | model fitted, "
                f"processing {len(test_weeks)} test weeks with {num_workers} worker(s)"
            )
            zero_scaled_values = zero_scaled_template(scaler, X_train_raw.shape[1])

            week_tasks = []
            for week_idx, week in enumerate(test_weeks):
                test_samples = test_samples_by_week[week]
                if balance_test:
                    test_samples = balance_samples(
                        test_samples,
                        seed=seed + week_idx + 1,
                        max_per_class=max_test_per_class,
                    )
                week_tasks.append((week_idx, week, test_samples))

            if num_workers <= 1:
                iterator = week_tasks
                if tqdm is not None:
                    iterator = tqdm(
                        week_tasks,
                        desc=f"RQ2-3 seed={seed} group={group}",
                        leave=False,
                    )
                for week_idx, week, test_samples in iterator:
                    trial_rows.extend(
                        process_test_week(
                            seed=seed,
                            group=group,
                            week_idx=week_idx,
                            week=week,
                            test_samples=test_samples,
                            scaler=scaler,
                            clf=clf,
                            importance_scores=importance_scores,
                            zero_scaled_values=zero_scaled_values,
                            strengths=strengths,
                            random_trials=random_trials,
                            threshold=threshold,
                        )
                    )
            else:
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = [
                        executor.submit(
                            process_test_week,
                            seed=seed,
                            group=group,
                            week_idx=week_idx,
                            week=week,
                            test_samples=test_samples,
                            scaler=scaler,
                            clf=clf,
                            importance_scores=importance_scores,
                            zero_scaled_values=zero_scaled_values,
                            strengths=strengths,
                            random_trials=random_trials,
                            threshold=threshold,
                        )
                        for week_idx, week, test_samples in week_tasks
                    ]
                    completed = 0
                    progress = None
                    if tqdm is not None:
                        progress = tqdm(
                            total=len(futures),
                            desc=f"RQ2-3 seed={seed} group={group}",
                            leave=False,
                        )
                    for future in as_completed(futures):
                        trial_rows.extend(future.result())
                        completed += 1
                        if progress is not None:
                            progress.update(1)
                        elif completed == 1 or completed == len(futures) or completed % max(1, len(futures) // 4) == 0:
                            log_progress(
                                f"[RQ2-3] seed={seed} group={group} | "
                                f"completed {completed}/{len(futures)} weeks"
                            )
                    if progress is not None:
                        progress.close()

            log_progress(f"[RQ2-3] seed={seed} group={group} | done")

    trial_rows.sort(key=trial_sort_key)
    return {
        "config": config,
        "trial_rows": trial_rows,
        "aggregate_rows": aggregate_rows(trial_rows),
    }


def save_results(results: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_rows = results["trial_rows"]
    aggregate = results["aggregate_rows"]
    if trial_rows:
        write_csv(output_dir / "trial_results.csv", list(trial_rows[0].keys()), trial_rows)
    if aggregate:
        write_csv(output_dir / "aggregate_results.csv", list(aggregate[0].keys()), aggregate)
    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RQ2-3 prediction flip analysis.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=["logistic_regression", "lightgbm", "random_forest"],
        help="Override config model_type.",
    )
    parser.add_argument(
        "--importance-method",
        choices=["coefficient", "permutation", "gain", "shap"],
        help="Override config importance_method.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.model is not None:
        config["model_type"] = args.model
    if args.importance_method is not None:
        config["importance_method"] = args.importance_method
    data_root = Path(config["data_root"])
    if not data_root.is_absolute():
        repo_root = Path(__file__).resolve().parents[4]
        candidates = [
            (args.config.parent / data_root).resolve(),
            (Path.cwd() / data_root).resolve(),
            (repo_root / data_root).resolve(),
        ]
        for candidate in candidates:
            if candidate.exists():
                config["data_root"] = str(candidate)
                break
        else:
            config["data_root"] = str(candidates[-1])
    results = run_from_config(config)
    save_results(results, args.output_dir)
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
