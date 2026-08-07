#!/usr/bin/env python3
"""
RQ2-4: PE-Inspired Feature Intervention Validation
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "artifacts" / "shared"))

from common import (  # type: ignore
    DLL_CHARACTERISTICS,
    DOS_MEMBERS,
    IMAGE_CHARACTERISTICS,
    balance_samples,
    extract_feature_container,
    labels_array,
    load_samples_from_jsonl,
    load_samples_from_paths,
    matches_packer_filter,
    select_feature_group,
    summarize,
    week_paths,
)

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover
    tqdm = None


HEADER_FEATURE_NAMES = (
    [
        "timestamp",
        "number_of_sections",
        "number_of_symbols",
        "sizeof_optional_header",
        "pointer_to_symbol_table",
        "machine",
        "subsystem",
        "major_image_version",
        "minor_image_version",
        "major_linker_version",
        "minor_linker_version",
        "major_operating_system_version",
        "minor_operating_system_version",
        "major_subsystem_version",
        "minor_subsystem_version",
        "sizeof_code",
        "sizeof_headers",
        "sizeof_image",
        "sizeof_initialized_data",
        "sizeof_uninitialized_data",
        "sizeof_stack_reserve",
        "sizeof_stack_commit",
        "sizeof_heap_reserve",
        "sizeof_heap_commit",
        "address_of_entrypoint",
        "base_of_code",
        "image_base",
        "section_alignment",
        "checksum",
        "number_of_rvas_and_sizes",
    ]
    + [f"coff_characteristic::{name}" for name in IMAGE_CHARACTERISTICS]
    + [f"dll_characteristic::{name}" for name in DLL_CHARACTERISTICS]
    + [f"dos::{name}" for name in DOS_MEMBERS]
)

HEADER_STRICT_FEATURES = {
    "timestamp",
    "number_of_symbols",
    "pointer_to_symbol_table",
    "checksum",
    "dos::e_csum",
    "dos::e_oemid",
    "dos::e_oeminfo",
}

HEADER_WEAK_FEATURES = {
    "major_image_version",
    "minor_image_version",
    "major_linker_version",
    "minor_linker_version",
    "major_operating_system_version",
    "minor_operating_system_version",
    "major_subsystem_version",
    "minor_subsystem_version",
    "sizeof_headers",
    "sizeof_code",
    "sizeof_initialized_data",
    "sizeof_uninitialized_data",
}

IMPORT_LIBRARY_HASHER = FeatureHasher(256, input_type="string", alternate_sign=False)
IMPORT_API_HASHER = FeatureHasher(1024, input_type="string", alternate_sign=False)


@dataclass(frozen=True)
class DistributionStats:
    q05: np.ndarray
    q25: np.ndarray
    median: np.ndarray
    q75: np.ndarray
    q95: np.ndarray
    iqr: np.ndarray


@dataclass(frozen=True)
class ImportTokenInfo:
    libraries: tuple[str, ...]
    apis: tuple[str, ...]


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


def transform_selected_columns(scaler: StandardScaler, X_raw_subset: np.ndarray, indices: np.ndarray) -> np.ndarray:
    stabilized = stabilize_features(X_raw_subset)
    scaled = (stabilized - scaler.mean_[indices]) / scaler.scale_[indices]
    return np.clip(np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0), -100.0, 100.0)


def import_feature_label(index: int) -> str:
    if index == 0:
        return "import_count"
    if index == 1:
        return "library_count"
    if 2 <= index < 258:
        return f"library_hash_bin_{index - 2:03d}"
    return f"import_hash_bin_{index - 258:04d}"


def feature_label(group: str, index: int) -> str:
    if group == "header":
        return HEADER_FEATURE_NAMES[index]
    if group == "imports":
        return import_feature_label(index)
    return f"{group}_{index}"


def selected_feature_names(group: str, indices: np.ndarray) -> str:
    if len(indices) == 0:
        return ""
    return "|".join(feature_label(group, int(index)) for index in indices)


def fit_classifier(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    model_type: str,
    lgbm_n_jobs: int,
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
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        clf.fit(np.asarray(X_train_scaled, dtype=np.float64), np.asarray(y_train))
    return clf


def positive_proba(clf: Any, X: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
            category=UserWarning,
        )
        proba = clf.predict_proba(np.asarray(X, dtype=np.float64))
    proba = np.nan_to_num(np.asarray(proba, dtype=np.float64), nan=0.0, posinf=1.0, neginf=0.0)
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

    if model_type != "lightgbm":
        raise ValueError(f"Unsupported model_type: {model_type}")

    if importance_method == "permutation":
        if X_val_scaled is None or y_val is None:
            raise ValueError("Permutation importance requires a validation split.")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
                category=UserWarning,
            )
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

    raise ValueError(
        "Unsupported importance_method for LightGBM: "
        f"{importance_method}. Use 'permutation' or 'gain'."
    )


def compute_distribution_stats(X_raw: np.ndarray, y: np.ndarray) -> DistributionStats:
    benign = np.asarray(X_raw[y == 0], dtype=np.float64)
    if benign.size == 0:
        raise ValueError("Benign training samples are required for PE-inspired interventions.")
    q05 = np.percentile(benign, 5, axis=0)
    q25 = np.percentile(benign, 25, axis=0)
    median = np.percentile(benign, 50, axis=0)
    q75 = np.percentile(benign, 75, axis=0)
    q95 = np.percentile(benign, 95, axis=0)
    return DistributionStats(
        q05=q05.astype(np.float64),
        q25=q25.astype(np.float64),
        median=median.astype(np.float64),
        q75=q75.astype(np.float64),
        q95=q95.astype(np.float64),
        iqr=(q75 - q25).astype(np.float64),
    )


def header_candidate_indices(profile: str) -> np.ndarray:
    names = set(HEADER_STRICT_FEATURES)
    if profile == "strict_plus_weak":
        names |= HEADER_WEAK_FEATURES
    elif profile != "strict":
        raise ValueError(f"Unsupported header candidate profile: {profile}")
    return np.asarray(
        [idx for idx, name in enumerate(HEADER_FEATURE_NAMES) if name in names],
        dtype=int,
    )


def imports_candidate_indices(n_features: int) -> np.ndarray:
    return np.arange(n_features, dtype=int)


def candidate_indices_for_group(group: str, profile: str, n_features: int) -> np.ndarray:
    if group == "header":
        return header_candidate_indices(profile)
    if group == "imports":
        return imports_candidate_indices(n_features)
    raise ValueError(f"Unsupported feature_group for RQ2-4: {group}")


def load_import_token_lookup(paths: Sequence[Path], packer_filter: str) -> Dict[str, ImportTokenInfo]:
    lookup: Dict[str, ImportTokenInfo] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not matches_packer_filter(row, packer_filter):
                    continue
                sha256 = row.get("sha256")
                if not sha256:
                    continue
                raw = extract_feature_container(row)
                imports_obj = raw.get("imports") or {}
                if not isinstance(imports_obj, dict):
                    imports_obj = {}
                libraries = tuple({str(lib).lower() for lib in imports_obj.keys()})
                apis = tuple(
                    f"{str(lib).lower()}:{str(entry)}"
                    for lib, items in imports_obj.items()
                    for entry in (items or [])
                )
                lookup[str(sha256)] = ImportTokenInfo(libraries=libraries, apis=apis)
    return lookup


def batch_hash_bins(tokens: Sequence[str], hasher: FeatureHasher) -> Dict[str, int]:
    if not tokens:
        return {}
    matrix = hasher.transform([[token] for token in tokens])
    bins = np.asarray(matrix.argmax(axis=1)).ravel().astype(int)
    return {token: int(bin_idx) for token, bin_idx in zip(tokens, bins)}


def build_import_token_importance_rows(
    *,
    train_samples: Sequence[object],
    import_token_lookup: Dict[str, ImportTokenInfo],
    importance_scores: np.ndarray,
) -> List[Dict[str, object]]:
    library_doc_freq: Counter[str] = Counter()
    library_total_count: Counter[str] = Counter()
    api_doc_freq: Counter[str] = Counter()
    api_total_count: Counter[str] = Counter()

    for sample in train_samples:
        token_info = import_token_lookup.get(sample.sha256)
        if token_info is None:
            continue
        unique_libraries = set(token_info.libraries)
        unique_apis = set(token_info.apis)
        for token in unique_libraries:
            library_doc_freq[token] += 1
        for token in token_info.libraries:
            library_total_count[token] += 1
        for token in unique_apis:
            api_doc_freq[token] += 1
        for token in token_info.apis:
            api_total_count[token] += 1

    library_tokens = sorted(library_doc_freq)
    api_tokens = sorted(api_doc_freq)
    library_bins = batch_hash_bins(library_tokens, IMPORT_LIBRARY_HASHER)
    api_bins = batch_hash_bins(api_tokens, IMPORT_API_HASHER)

    library_collision_buckets: Dict[int, List[str]] = defaultdict(list)
    for token, bin_idx in library_bins.items():
        library_collision_buckets[bin_idx].append(token)
    api_collision_buckets: Dict[int, List[str]] = defaultdict(list)
    for token, bin_idx in api_bins.items():
        api_collision_buckets[bin_idx].append(token)

    rows: List[Dict[str, object]] = []
    for token in library_tokens:
        local_bin = library_bins[token]
        global_index = 2 + local_bin
        collisions = library_collision_buckets[local_bin]
        rows.append(
            {
                "token_type": "dll",
                "token": token,
                "global_feature_index": global_index,
                "feature_label": import_feature_label(global_index),
                "bin_importance": float(importance_scores[global_index]),
                "bin_importance_abs": float(abs(importance_scores[global_index])),
                "document_frequency": int(library_doc_freq[token]),
                "total_count": int(library_total_count[token]),
                "collision_count": len(collisions),
                "collision_example_tokens": "|".join(collisions[:10]),
            }
        )
    for token in api_tokens:
        local_bin = api_bins[token]
        global_index = 258 + local_bin
        collisions = api_collision_buckets[local_bin]
        rows.append(
            {
                "token_type": "api",
                "token": token,
                "global_feature_index": global_index,
                "feature_label": import_feature_label(global_index),
                "bin_importance": float(importance_scores[global_index]),
                "bin_importance_abs": float(abs(importance_scores[global_index])),
                "document_frequency": int(api_doc_freq[token]),
                "total_count": int(api_total_count[token]),
                "collision_count": len(collisions),
                "collision_example_tokens": "|".join(collisions[:10]),
            }
        )

    rows.sort(
        key=lambda row: (
            -float(row["bin_importance_abs"]),
            -int(row["document_frequency"]),
            str(row["token"]),
        )
    )
    return rows


def build_feature_importance_rows(
    *,
    group: str,
    importance_scores: np.ndarray,
    candidate_indices: np.ndarray,
) -> List[Dict[str, object]]:
    candidate_set = set(int(idx) for idx in candidate_indices)
    rows: List[Dict[str, object]] = []
    for index, score in enumerate(np.asarray(importance_scores, dtype=np.float64)):
        rows.append(
            {
                "feature_group": group,
                "feature_index": int(index),
                "feature_name": feature_label(group, int(index)),
                "importance": float(score),
                "importance_abs": float(abs(score)),
                "is_candidate": int(index) in candidate_set,
            }
        )
    rows.sort(key=lambda row: (-float(row["importance_abs"]), int(row["feature_index"])))
    return rows


def operators_for_group(group: str, config: dict) -> List[str]:
    if group == "header":
        return list(config.get("header_operators", ["header_benign_substitution"]))
    if group == "imports":
        return list(config.get("imports_operators", ["imports_benign_mass_injection"]))
    raise ValueError(f"Unsupported feature_group for RQ2-4: {group}")


def select_top_indices_within_pool(
    importance_scores: np.ndarray,
    candidate_indices: np.ndarray,
    n_select: int,
) -> np.ndarray:
    n_select = max(1, min(len(candidate_indices), int(n_select)))
    ranked_local = np.argsort(-importance_scores[candidate_indices], kind="stable")
    return candidate_indices[ranked_local[:n_select]]


def selection_levels_for_group(group: str, config: dict, n_candidate_features: int) -> List[Dict[str, object]]:
    if group == "header" and "header_selection_budgets" in config:
        budgets = [int(value) for value in config["header_selection_budgets"]]
        return [
            {
                "selection_mode": "budget",
                "selection_value": int(max(1, min(n_candidate_features, budget))),
                "selection_label": f"top_{int(max(1, min(n_candidate_features, budget)))}",
                "n_select": int(max(1, min(n_candidate_features, budget))),
            }
            for budget in budgets
        ]

    strengths = [float(value) for value in config["perturbation_strengths"]]
    return [
        {
            "selection_mode": "ratio",
            "selection_value": strength,
            "selection_label": f"{int(round(strength * 100))}pct",
            "n_select": int(max(1, min(n_candidate_features, math.ceil(n_candidate_features * strength)))),
        }
        for strength in strengths
    ]


def random_indices_from_pool(
    candidate_indices: np.ndarray,
    n_select: int,
    rng: np.random.Generator,
) -> np.ndarray:
    return np.asarray(rng.choice(candidate_indices, size=n_select, replace=False), dtype=int)


def intervention_seed(
    *,
    seed: int,
    week_idx: int,
    group: str,
    operator: str,
    selection_value: float,
    trial_id: int,
    selection_type: str,
) -> int:
    group_offset = sum(ord(ch) for ch in group)
    operator_offset = sum(ord(ch) for ch in operator)
    selection_offset = 0 if selection_type == "important" else 10_000_000
    strength_offset = int(round(selection_value * 10_000))
    return (
        seed * 100_000_000
        + week_idx * 1_000_000
        + group_offset * 10_000
        + operator_offset * 10
        + strength_offset
        + trial_id
        + selection_offset
    )


def apply_header_benign_substitution(
    X_raw_selected: np.ndarray,
    selected_indices: np.ndarray,
    stats: DistributionStats,
) -> np.ndarray:
    target = np.broadcast_to(stats.median[selected_indices], X_raw_selected.shape)
    return np.asarray(target, dtype=np.float64)


def apply_header_bounded_jitter(
    X_raw_selected: np.ndarray,
    selected_indices: np.ndarray,
    stats: DistributionStats,
    rng: np.random.Generator,
) -> np.ndarray:
    widths = np.maximum(stats.iqr[selected_indices], 1e-6)
    noise = rng.uniform(-0.25, 0.25, size=X_raw_selected.shape) * widths
    perturbed = X_raw_selected + noise
    lower = np.broadcast_to(stats.q05[selected_indices], X_raw_selected.shape)
    upper = np.broadcast_to(stats.q95[selected_indices], X_raw_selected.shape)
    return np.clip(perturbed, lower, upper)


def apply_imports_benign_mass_injection(
    X_raw_selected: np.ndarray,
    selected_indices: np.ndarray,
    stats: DistributionStats,
) -> np.ndarray:
    lower_target = np.broadcast_to(stats.q75[selected_indices], X_raw_selected.shape)
    upper_target = np.broadcast_to(stats.q95[selected_indices], X_raw_selected.shape)
    return np.minimum(np.maximum(X_raw_selected, lower_target), upper_target)


def build_perturbed_malware_scaled(
    *,
    baseline_malware_raw: np.ndarray,
    baseline_malware_scaled: np.ndarray,
    selected_indices: np.ndarray,
    operator: str,
    stats: DistributionStats,
    scaler: StandardScaler,
    rng: np.random.Generator,
) -> np.ndarray:
    perturbed_scaled = np.array(baseline_malware_scaled, copy=True)
    current_raw_selected = np.asarray(baseline_malware_raw[:, selected_indices], dtype=np.float64)

    if operator == "header_benign_substitution":
        new_raw_selected = apply_header_benign_substitution(current_raw_selected, selected_indices, stats)
    elif operator == "header_bounded_jitter":
        new_raw_selected = apply_header_bounded_jitter(current_raw_selected, selected_indices, stats, rng)
    elif operator == "imports_benign_mass_injection":
        new_raw_selected = apply_imports_benign_mass_injection(current_raw_selected, selected_indices, stats)
    else:
        raise ValueError(f"Unsupported operator: {operator}")

    perturbed_scaled[:, selected_indices] = transform_selected_columns(
        scaler,
        new_raw_selected,
        selected_indices,
    )
    return perturbed_scaled


def evaluate_metrics(
    *,
    baseline_proba_full: np.ndarray,
    perturbed_proba_full: np.ndarray,
    baseline_proba_malware: np.ndarray,
    perturbed_proba_malware: np.ndarray,
    y_test: np.ndarray,
    decision_threshold: float,
    boundary_quantiles: Sequence[float],
) -> Dict[str, float]:
    baseline_pred_malware = (baseline_proba_malware >= decision_threshold).astype(int)
    perturbed_pred_malware = (perturbed_proba_malware >= decision_threshold).astype(int)
    flips = baseline_pred_malware != perturbed_pred_malware
    malware_to_benign = (baseline_pred_malware == 1) & (perturbed_pred_malware == 0)
    baseline_margin = np.abs(baseline_proba_malware - decision_threshold)
    perturbed_margin = np.abs(perturbed_proba_malware - decision_threshold)

    out: Dict[str, float] = {
        "auc": float(roc_auc_score(y_test, perturbed_proba_full)),
        "flip_rate": float(flips.mean()),
        "malware_to_benign_rate": float(malware_to_benign.mean()),
        "mean_abs_probability_shift": float(np.mean(np.abs(perturbed_proba_malware - baseline_proba_malware))),
        "mean_signed_probability_shift": float(np.mean(perturbed_proba_malware - baseline_proba_malware)),
        "mean_margin_shift": float(np.mean(perturbed_margin - baseline_margin)),
    }

    order = np.argsort(baseline_margin, kind="stable")
    for quantile in boundary_quantiles:
        n_focus = max(1, math.ceil(len(order) * float(quantile)))
        focus_idx = order[:n_focus]
        key = int(round(float(quantile) * 100))
        out[f"boundary_low_{key}_flip_rate"] = float(flips[focus_idx].mean())
        out[f"boundary_low_{key}_malware_to_benign_rate"] = float(malware_to_benign[focus_idx].mean())

    return out


def trial_sort_key(row: Dict[str, object]) -> tuple:
    return (
        int(row["seed"]),
        str(row["feature_group"]),
        str(row["operator"]),
        str(row["test_week"]),
        str(row["selection_type"]),
        str(row["selection_mode"]),
        float(row["selection_value"]),
        int(row["trial_id"]),
    )


def aggregate_rows(trial_rows: List[Dict[str, object]], boundary_quantiles: Sequence[float]) -> List[Dict[str, object]]:
    grouped: Dict[tuple, List[Dict[str, object]]] = {}
    for row in trial_rows:
        key = (
            row["feature_group"],
            row["operator"],
            row["test_week"],
            row["selection_type"],
            row["selection_mode"],
            row["selection_value"],
        )
        grouped.setdefault(key, []).append(row)

    metric_names = [
        "auc",
        "delta_auc",
        "flip_rate",
        "malware_to_benign_rate",
        "mean_abs_probability_shift",
        "mean_signed_probability_shift",
        "mean_margin_shift",
    ]
    for quantile in boundary_quantiles:
        key = int(round(float(quantile) * 100))
        metric_names.append(f"boundary_low_{key}_flip_rate")
        metric_names.append(f"boundary_low_{key}_malware_to_benign_rate")

    aggregate: List[Dict[str, object]] = []
    for key, rows in sorted(grouped.items()):
        feature_group, operator, test_week, selection_type, selection_mode, selection_value = key
        out: Dict[str, object] = {
            "feature_group": feature_group,
            "operator": operator,
            "test_week": test_week,
            "selection_type": selection_type,
            "selection_mode": selection_mode,
            "selection_value": selection_value,
            "n_trials": len(rows),
            "n_selected_features": rows[0]["n_selected_features"],
            "n_candidate_features": rows[0]["n_candidate_features"],
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
    stats: DistributionStats,
    selection_levels: Sequence[Dict[str, object]],
    random_trials: int,
    decision_threshold: float,
    boundary_quantiles: Sequence[float],
    candidate_indices: np.ndarray,
    operators: Sequence[str],
) -> List[Dict[str, object]]:
    y_test = labels_array(test_samples)
    X_test_raw = select_feature_group(test_samples, group)
    baseline_scaled = transform_with_scaler(scaler, X_test_raw)
    baseline_proba_full = positive_proba(clf, baseline_scaled)
    malware_mask = y_test == 1
    malware_indices = np.flatnonzero(malware_mask)
    baseline_malware_raw = np.asarray(X_test_raw[malware_mask], dtype=np.float64)
    baseline_malware_scaled = np.asarray(baseline_scaled[malware_mask], dtype=np.float64)
    baseline_proba_malware = baseline_proba_full[malware_mask]
    baseline_auc = float(roc_auc_score(y_test, baseline_proba_full))

    rows: List[Dict[str, object]] = []
    for operator in operators:
        rows.append(
            {
                "seed": seed,
                "feature_group": group,
                "operator": operator,
                "test_week": week,
                "selection_type": "baseline",
                "selection_mode": "baseline",
                "selection_value": 0.0,
                "trial_id": 0,
                "n_selected_features": 0,
                "n_candidate_features": len(candidate_indices),
                "selected_feature_names": "",
                "auc": baseline_auc,
                "delta_auc": 0.0,
                "flip_rate": 0.0,
                "malware_to_benign_rate": 0.0,
                "mean_abs_probability_shift": 0.0,
                "mean_signed_probability_shift": 0.0,
                "mean_margin_shift": 0.0,
                **{
                    f"boundary_low_{int(round(float(q) * 100))}_flip_rate": 0.0
                    for q in boundary_quantiles
                },
                **{
                    f"boundary_low_{int(round(float(q) * 100))}_malware_to_benign_rate": 0.0
                    for q in boundary_quantiles
                },
            }
        )

        for selection in selection_levels:
            selection_mode = str(selection["selection_mode"])
            selection_value = float(selection["selection_value"])
            n_select = int(selection["n_select"])
            important_indices = select_top_indices_within_pool(importance_scores, candidate_indices, n_select)
            important_rng = np.random.default_rng(
                intervention_seed(
                    seed=seed,
                    week_idx=week_idx,
                    group=group,
                    operator=operator,
                    selection_value=selection_value,
                    trial_id=0,
                    selection_type="important",
                )
            )
            important_scaled = build_perturbed_malware_scaled(
                baseline_malware_raw=baseline_malware_raw,
                baseline_malware_scaled=baseline_malware_scaled,
                selected_indices=important_indices,
                operator=operator,
                stats=stats,
                scaler=scaler,
                rng=important_rng,
            )
            important_proba_malware = positive_proba(clf, important_scaled)
            important_proba_full = np.array(baseline_proba_full, copy=True)
            important_proba_full[malware_indices] = important_proba_malware
            important_metrics = evaluate_metrics(
                baseline_proba_full=baseline_proba_full,
                perturbed_proba_full=important_proba_full,
                baseline_proba_malware=baseline_proba_malware,
                perturbed_proba_malware=important_proba_malware,
                y_test=y_test,
                decision_threshold=decision_threshold,
                boundary_quantiles=boundary_quantiles,
            )
            rows.append(
                {
                    "seed": seed,
                    "feature_group": group,
                    "operator": operator,
                    "test_week": week,
                    "selection_type": "important",
                    "selection_mode": selection_mode,
                    "selection_value": selection_value,
                    "trial_id": 0,
                    "n_selected_features": len(important_indices),
                    "n_candidate_features": len(candidate_indices),
                    "selected_feature_names": selected_feature_names(group, important_indices),
                    **important_metrics,
                    "delta_auc": important_metrics["auc"] - baseline_auc,
                }
            )

            for trial_id in range(1, random_trials + 1):
                random_rng = np.random.default_rng(
                    intervention_seed(
                        seed=seed,
                        week_idx=week_idx,
                        group=group,
                        operator=operator,
                        selection_value=selection_value,
                        trial_id=trial_id,
                        selection_type="random",
                    )
                )
                random_indices = random_indices_from_pool(candidate_indices, n_select, random_rng)
                random_scaled = build_perturbed_malware_scaled(
                    baseline_malware_raw=baseline_malware_raw,
                    baseline_malware_scaled=baseline_malware_scaled,
                    selected_indices=random_indices,
                    operator=operator,
                    stats=stats,
                    scaler=scaler,
                    rng=random_rng,
                )
                random_proba_malware = positive_proba(clf, random_scaled)
                random_proba_full = np.array(baseline_proba_full, copy=True)
                random_proba_full[malware_indices] = random_proba_malware
                random_metrics = evaluate_metrics(
                    baseline_proba_full=baseline_proba_full,
                    perturbed_proba_full=random_proba_full,
                    baseline_proba_malware=baseline_proba_malware,
                    perturbed_proba_malware=random_proba_malware,
                    y_test=y_test,
                    decision_threshold=decision_threshold,
                    boundary_quantiles=boundary_quantiles,
                )
                rows.append(
                    {
                        "seed": seed,
                        "feature_group": group,
                        "operator": operator,
                        "test_week": week,
                        "selection_type": "random",
                        "selection_mode": selection_mode,
                        "selection_value": selection_value,
                        "trial_id": trial_id,
                        "n_selected_features": n_select,
                        "n_candidate_features": len(candidate_indices),
                        "selected_feature_names": selected_feature_names(group, random_indices),
                        **random_metrics,
                        "delta_auc": random_metrics["auc"] - baseline_auc,
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
    random_trials = int(config["random_trials"])
    decision_threshold = float(config.get("decision_threshold", 0.5))
    boundary_quantiles = [float(value) for value in config.get("boundary_quantiles", [0.1, 0.2])]
    balance_train = bool(config.get("balance_train", True))
    balance_test = bool(config.get("balance_test", True))
    max_train_per_class = config.get("max_train_per_class")
    max_test_per_class = config.get("max_test_per_class")
    model_type = str(config.get("model_type", "logistic_regression"))
    candidate_profile = str(config.get("candidate_profile", "strict"))

    if model_type == "lightgbm":
        num_workers = int(config.get("num_workers", 2))
        lgbm_n_jobs = int(config.get("lgbm_n_jobs", 2))
        omp_num_threads = int(config.get("omp_num_threads", 1))
    else:
        num_workers = int(config.get("num_workers", max(1, min(4, os.cpu_count() or 1))))
        lgbm_n_jobs = int(config.get("lgbm_n_jobs", 1))
        omp_num_threads = int(config.get("omp_num_threads", 1))

    importance_method = str(
        config.get(
            "importance_method",
            "coefficient" if model_type == "logistic_regression" else "permutation",
        )
    )
    validation_fraction = float(config.get("validation_fraction", 0.2))
    permutation_repeats = int(config.get("permutation_repeats", 3))

    if model_type == "logistic_regression" and importance_method != "coefficient":
        raise ValueError(
            "Logistic Regression in RQ2-4 currently requires "
            "importance_method='coefficient'."
        )
    if model_type == "lightgbm" and importance_method not in {"permutation", "gain"}:
        raise ValueError(
            "LightGBM in RQ2-4 supports importance_method in {'permutation', 'gain'}."
        )
    if num_workers < 1 or lgbm_n_jobs < 1 or omp_num_threads < 1:
        raise ValueError("num_workers, lgbm_n_jobs, and omp_num_threads must all be >= 1.")
    if not (0.0 < validation_fraction < 1.0):
        raise ValueError("validation_fraction must be between 0 and 1.")
    if model_type == "lightgbm":
        configure_thread_env(omp_num_threads)

    train_paths = week_paths(data_root, platform, train_weeks, "train")
    all_train_samples = load_samples_from_paths(train_paths, packer_filter=packer_filter)
    import_token_lookup: Dict[str, ImportTokenInfo] = {}
    if "imports" in feature_groups:
        import_token_lookup = load_import_token_lookup(train_paths, packer_filter)
    test_samples_by_week = {
        week: load_samples_from_jsonl(
            week_paths(data_root, platform, [week], "test")[0],
            packer_filter=packer_filter,
        )
        for week in test_weeks
    }

    trial_rows: List[Dict[str, object]] = []
    artifact_tables: Dict[str, List[Dict[str, object]]] = {}
    total_jobs = len(seeds) * len(feature_groups)
    job_counter = 0

    for seed in seeds:
        train_samples = all_train_samples
        if balance_train:
            train_samples = balance_samples(all_train_samples, seed=seed, max_per_class=max_train_per_class)
        y_train = labels_array(train_samples)
        train_idx = np.arange(len(train_samples))
        val_idx = np.array([], dtype=int)
        if model_type == "lightgbm" and importance_method == "permutation":
            train_idx, val_idx = split_train_validation_indices(
                n_samples=len(train_samples),
                y_train=y_train,
                seed=seed,
                validation_fraction=validation_fraction,
            )

        for group in feature_groups:
            job_counter += 1
            log_progress(
                f"[RQ2-4] ({job_counter}/{total_jobs}) "
                f"seed={seed} group={group} | fitting {model_type} "
                f"(importance={importance_method})"
            )
            X_train_raw = select_feature_group(train_samples, group)
            if model_type == "lightgbm" and importance_method == "permutation":
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
            stats = compute_distribution_stats(X_train_fit_raw, y_train_fit)
            candidate_indices = candidate_indices_for_group(group, candidate_profile, X_train_raw.shape[1])
            selection_levels = selection_levels_for_group(group, config, len(candidate_indices))
            operators = operators_for_group(group, config)
            artifact_tables[f"importance_reports/seed_{seed}_{group}_feature_importance.csv"] = (
                build_feature_importance_rows(
                    group=group,
                    importance_scores=importance_scores,
                    candidate_indices=candidate_indices,
                )
            )
            if group == "imports" and import_token_lookup:
                selected_train_samples = [train_samples[int(idx)] for idx in train_idx] if len(train_idx) else list(train_samples)
                artifact_tables[f"importance_reports/seed_{seed}_{group}_token_importance.csv"] = (
                    build_import_token_importance_rows(
                        train_samples=selected_train_samples,
                        import_token_lookup=import_token_lookup,
                        importance_scores=importance_scores,
                    )
                )
            log_progress(
                f"[RQ2-4] seed={seed} group={group} | "
                f"{len(candidate_indices)} candidate features, {len(operators)} operator(s), "
                f"{len(selection_levels)} selection level(s), "
                f"{len(test_weeks)} test weeks with {num_workers} worker(s)"
            )

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
                        desc=f"RQ2-4 seed={seed} group={group}",
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
                            stats=stats,
                            selection_levels=selection_levels,
                            random_trials=random_trials,
                            decision_threshold=decision_threshold,
                            boundary_quantiles=boundary_quantiles,
                            candidate_indices=candidate_indices,
                            operators=operators,
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
                            stats=stats,
                            selection_levels=selection_levels,
                            random_trials=random_trials,
                            decision_threshold=decision_threshold,
                            boundary_quantiles=boundary_quantiles,
                            candidate_indices=candidate_indices,
                            operators=operators,
                        )
                        for week_idx, week, test_samples in week_tasks
                    ]
                    completed = 0
                    progress = None
                    if tqdm is not None:
                        progress = tqdm(
                            total=len(futures),
                            desc=f"RQ2-4 seed={seed} group={group}",
                            leave=False,
                        )
                    for future in as_completed(futures):
                        trial_rows.extend(future.result())
                        completed += 1
                        if progress is not None:
                            progress.update(1)
                        elif completed == 1 or completed == len(futures) or completed % max(1, len(futures) // 4) == 0:
                            log_progress(
                                f"[RQ2-4] seed={seed} group={group} | "
                                f"completed {completed}/{len(futures)} weeks"
                            )
                    if progress is not None:
                        progress.close()

            log_progress(f"[RQ2-4] seed={seed} group={group} | done")

    trial_rows.sort(key=trial_sort_key)
    return {
        "config": config,
        "trial_rows": trial_rows,
        "aggregate_rows": aggregate_rows(trial_rows, boundary_quantiles),
        "artifact_tables": artifact_tables,
    }


def save_results(results: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_rows = results["trial_rows"]
    aggregate = results["aggregate_rows"]
    artifact_tables = results.get("artifact_tables", {})
    if trial_rows:
        write_csv(output_dir / "trial_results.csv", list(trial_rows[0].keys()), trial_rows)
    if aggregate:
        write_csv(output_dir / "aggregate_results.csv", list(aggregate[0].keys()), aggregate)
    for relative_path, rows in artifact_tables.items():
        if rows:
            write_csv(output_dir / relative_path, list(rows[0].keys()), rows)
    json_payload = {
        "config": results["config"],
        "trial_rows": trial_rows,
        "aggregate_rows": aggregate,
        "artifact_table_files": sorted(artifact_tables.keys()),
    }
    (output_dir / "results.json").write_text(json.dumps(json_payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RQ2-4 PE-inspired intervention validation.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=["logistic_regression", "lightgbm"],
        help="Override config model_type.",
    )
    parser.add_argument(
        "--importance-method",
        choices=["coefficient", "permutation", "gain"],
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
        repo_root = Path(__file__).resolve().parents[4]  # DCSA_Open_Science/ (was [3]="artifacts/", a pre-existing off-by-one)
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
