#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "artifacts" / "shared"))

from common import (  # type: ignore
    FEATURE_GROUPS,
    build_sample,
    feature_rankings,
    labels_array,
    matches_packer_filter,
    select_feature_group,
    stabilize_features,
    summarize,
    week_paths,
)

"""
RQ3 Block 1: Density-Stratified Reliability of Feature-Subset Separability

Main protocol:
- weekly files define only the temporal train/test boundary
- all train weeks are pooled into one train pool
- one balanced train subset is sampled per seed
- one model is trained per seed and feature subset
- all test weeks are pooled into one evaluation pool

Weekly information is preserved for composition checks and supplementary
robustness analysis, but not for week-by-week retraining.
"""

warnings.filterwarnings(
    "ignore",
    message="divide by zero encountered in matmul",
    category=RuntimeWarning,
    module=r"sklearn\..*",
)
warnings.filterwarnings(
    "ignore",
    message="overflow encountered in matmul",
    category=RuntimeWarning,
    module=r"sklearn\..*",
)
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in matmul",
    category=RuntimeWarning,
    module=r"sklearn\..*",
)
warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores",
    category=UserWarning,
    module=r"joblib\..*",
)
warnings.filterwarnings(
    "ignore",
    message=r"X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
    module=r"sklearn\.utils\.validation",
)


@dataclass(frozen=True)
class SampleRecord:
    sample: object
    packed: bool
    week: str


@dataclass(frozen=True)
class LabelRecord:
    """Cheap stand-in for a SampleRecord before the expensive feature vectors are built.

    balance_records only ever looks at `.label` (see balance_label_records below), so the
    full population can be scanned and balanced using just this lightweight record, and the
    costly build_sample() vectorization can be deferred to the small post-balancing subset.
    """

    file_path: Path
    line_index: int
    label: int
    sha256: str
    packed: bool
    week: str


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fit_scaler(X_train_raw: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(stabilize_features(X_train_raw))
    return scaler


def transform_with_scaler(scaler: StandardScaler, X_raw: np.ndarray) -> np.ndarray:
    X_scaled = scaler.transform(stabilize_features(X_raw))
    X_scaled = np.clip(np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0), -100.0, 100.0)
    return np.asarray(X_scaled, dtype=np.float32)


def fit_classifier(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    model_type: str,
    lgbm_n_jobs: int,
    rf_n_estimators: int,
    rf_n_jobs: int,
) -> Any:
    X_train_scaled = np.asarray(X_train_scaled, dtype=np.float32)
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
        clf.fit(X_train_scaled, y_train)
    return clf


def positive_proba(clf: Any, X_scaled: np.ndarray) -> np.ndarray:
    X_scaled = np.asarray(X_scaled, dtype=np.float32)
    proba = clf.predict_proba(X_scaled)
    if proba.ndim != 2 or proba.shape[1] < 2:
        raise ValueError("Classifier must expose binary predict_proba output.")
    return proba[:, 1]


def compute_auc(clf: Any, X_scaled: np.ndarray, y_true: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    proba = positive_proba(clf, X_scaled)
    return float(roc_auc_score(y_true, proba))


def compute_sample_mix_at_k(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    if len(X) <= 1:
        return np.zeros(len(X), dtype=np.float64)
    effective_k = max(1, min(k, len(X) - 1))
    nn = NearestNeighbors(n_neighbors=effective_k + 1, metric="euclidean")
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    neighbor_labels = y[indices[:, 1:]]
    return (neighbor_labels != y[:, None]).mean(axis=1)


def js_divergence_feature_mass(X_raw: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    if len(X_raw) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    a = np.asarray(X_raw, dtype=np.float32)
    a = np.clip(a, 0.0, None)

    c0 = a[y == 0].sum(axis=0)
    c1 = a[y == 1].sum(axis=0)
    if c0.sum() <= 0 or c1.sum() <= 0:
        return float("nan")

    p = c0 + eps
    q = c1 + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    kl_pm = float(np.sum(p * (np.log(p) - np.log(m))))
    kl_qm = float(np.sum(q * (np.log(q) - np.log(m))))
    return 0.5 * (kl_pm + kl_qm)


def _packed_from_row(row: dict) -> bool:
    packer = row.get("packer")
    if isinstance(packer, list):
        return len(packer) > 0
    if isinstance(packer, str):
        return bool(packer.strip())
    return bool(packer) if packer is not None else False


def load_label_records_from_jsonl(path: Path, packer_filter: str) -> List[LabelRecord]:
    """Same row survival/order as the old load_records_from_jsonl, minus the vectorization.

    This must accept/reject/order rows identically to what `build_sample(row, path) is None`
    used to do (label not in {0, 1} or missing sha256), so that the population balance_records
    later samples from is byte-for-byte the same population as before.
    """
    records: List[LabelRecord] = []
    week = path.stem.replace("_Win32_test", "").replace("_Win32_train", "")
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not matches_packer_filter(row, packer_filter):
                continue
            label = row.get("label")
            sha256 = row.get("sha256")
            if label not in {0, 1} or not sha256:
                continue
            records.append(
                LabelRecord(
                    file_path=path,
                    line_index=line_index,
                    label=int(label),
                    sha256=str(sha256),
                    packed=_packed_from_row(row),
                    week=week,
                )
            )
    return records


def load_label_records_from_paths(paths: Sequence[Path], packer_filter: str) -> List[LabelRecord]:
    rows: List[LabelRecord] = []
    for path in paths:
        rows.extend(load_label_records_from_jsonl(path, packer_filter))
    return rows


def balance_label_records(
    records: Sequence[LabelRecord],
    seed: int,
    max_per_class: int | None = None,
) -> List[LabelRecord]:
    """Identical selection logic to the old balance_records, over LabelRecord instead of
    SampleRecord. Only `.label` is inspected, so this produces the exact same selection
    (same rng call order: benign pick, then malware pick, then shuffle) as before."""
    benign = [record for record in records if record.label == 0]
    malware = [record for record in records if record.label == 1]
    target = min(len(benign), len(malware))
    if max_per_class is not None:
        target = min(target, max_per_class)
    rng = np.random.default_rng(seed)

    def pick(rows: List[LabelRecord]) -> List[LabelRecord]:
        if len(rows) <= target:
            return list(rows)
        idx = rng.choice(len(rows), size=target, replace=False)
        return [rows[i] for i in idx]

    balanced = pick(benign) + pick(malware)
    rng.shuffle(balanced)
    return balanced


def vectorize_selected(selected: Sequence[LabelRecord]) -> List[SampleRecord]:
    """Build full SampleRecords only for the given (already-balanced) selection.

    Re-reads each source file once, skipping straight past any line whose index isn't in
    the selected set for that file (cheap: no json.loads, no build_sample) so the expensive
    vectorization only ever runs on the small post-balancing subset, not the full corpus.
    """
    positions_by_file: Dict[Path, set] = {}
    for record in selected:
        positions_by_file.setdefault(record.file_path, set()).add(record.line_index)

    cache: Dict[tuple, object] = {}
    for file_path, positions in positions_by_file.items():
        with file_path.open("r", encoding="utf-8") as handle:
            for line_index, line in enumerate(handle):
                if line_index not in positions:
                    continue
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                sample = build_sample(row, file_path)
                if sample is None:
                    continue
                cache[(file_path, line_index)] = sample

    result: List[SampleRecord] = []
    for record in selected:
        sample = cache.get((record.file_path, record.line_index))
        if sample is None:
            raise RuntimeError(
                f"Selected record not found on re-read: {record.file_path}:{record.line_index}. "
                "The source file may have changed between the label pass and the vectorization pass."
            )
        result.append(SampleRecord(sample=sample, packed=record.packed, week=record.week))
    return result


def rank_conflict_count(rank_a: Dict[str, int], rank_b: Dict[str, int]) -> int:
    groups = sorted(rank_a.keys())
    conflicts = 0
    for i, left in enumerate(groups):
        for right in groups[i + 1 :]:
            order_a = rank_a[left] - rank_a[right]
            order_b = rank_b[left] - rank_b[right]
            if order_a == 0 or order_b == 0:
                continue
            if math.copysign(1, order_a) != math.copysign(1, order_b):
                conflicts += 1
    return conflicts


def compute_density_scores(
    X_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    k_density: int,
) -> np.ndarray:
    effective_k = max(1, min(k_density, len(X_train_scaled)))
    nn = NearestNeighbors(n_neighbors=effective_k, metric="cosine")
    nn.fit(X_train_scaled)
    distances, _ = nn.kneighbors(X_test_scaled)
    return -distances.mean(axis=1)


def assign_density_bins(
    scores: np.ndarray,
    *,
    low_quantile: float = 0.25,
    high_quantile: float = 0.75,
) -> tuple[np.ndarray, Dict[str, float]]:
    if not (0.0 < low_quantile < high_quantile < 1.0):
        raise ValueError("Density quantiles must satisfy 0 < low_quantile < high_quantile < 1.")
    q25 = float(np.quantile(scores, low_quantile))
    q75 = float(np.quantile(scores, high_quantile))
    bins = np.full(len(scores), "mid_density", dtype=object)
    bins[scores <= q25] = "low_density"
    bins[scores >= q75] = "high_density"
    return bins, {
        "q25": q25,
        "q75": q75,
        "low_quantile": low_quantile,
        "high_quantile": high_quantile,
        "low_threshold": q25,
        "high_threshold": q75,
    }


def as_samples(records: Sequence[SampleRecord]) -> List[object]:
    return [record.sample for record in records]


def composition_row(seed: int, density_bin: str, records: Sequence[SampleRecord]) -> Dict[str, object]:
    benign = sum(record.sample.label == 0 for record in records)
    malware = sum(record.sample.label == 1 for record in records)
    packed_malware = sum(record.sample.label == 1 and record.packed for record in records)
    packed_ratio = float(packed_malware / malware) if malware else float("nan")
    week_counts: Dict[str, int] = {}
    for record in records:
        week_counts[record.week] = week_counts.get(record.week, 0) + 1
    return {
        "seed": seed,
        "density_bin": density_bin,
        "n_samples": len(records),
        "n_benign": benign,
        "n_malware": malware,
        "benign_to_malware_ratio": float(benign / malware) if malware else float("nan"),
        "n_packed_malware": packed_malware,
        "packed_ratio": packed_ratio,
        "week_distribution_json": json.dumps(week_counts, sort_keys=True),
    }


def aggregate_rows(rows: List[Dict[str, object]], group_keys: Sequence[str], metric_keys: Sequence[str]) -> List[Dict[str, object]]:
    grouped: Dict[tuple, List[Dict[str, object]]] = {}
    for row in rows:
        key = tuple(row[key_name] for key_name in group_keys)
        grouped.setdefault(key, []).append(row)

    out: List[Dict[str, object]] = []
    for key, bucket in sorted(grouped.items()):
        row: Dict[str, object] = {name: value for name, value in zip(group_keys, key)}
        row["n_rows"] = len(bucket)
        for metric in metric_keys:
            values = [float(item[metric]) for item in bucket if not math.isnan(float(item[metric]))]
            if not values:
                row[f"{metric}_mean"] = float("nan")
                row[f"{metric}_std"] = float("nan")
                row[f"{metric}_min"] = float("nan")
                row[f"{metric}_max"] = float("nan")
                continue
            stats = summarize(values)
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_std"] = stats["std"]
            row[f"{metric}_min"] = stats["min"]
            row[f"{metric}_max"] = stats["max"]
        out.append(row)
    return out


def process_seed_block1(
    *,
    seed: int,
    all_train_records: Sequence[LabelRecord],
    all_test_records_by_week: Dict[str, List[LabelRecord]],
    test_weeks: Sequence[str],
    feature_groups: Sequence[str],
    k_values: Sequence[int],
    quantile_pairs: Sequence[tuple[float, float]],
    mix_k: int,
    model_type: str,
    lgbm_n_jobs: int,
    rf_n_estimators: int,
    rf_n_jobs: int,
    balance_train: bool,
    balance_test: bool,
    max_train_per_class: int | None,
    max_test_per_class: int | None,
    save_density_rows: bool,
) -> Dict[tuple[int, float, float], Dict[str, List[Dict[str, object]]]]:
    combos: List[tuple[int, float, float]] = [
        (k_density, low_q, high_q) for k_density in k_values for (low_q, high_q) in quantile_pairs
    ]
    results_by_combo: Dict[tuple[int, float, float], Dict[str, List[Dict[str, object]]]] = {
        combo: {
            "density_rows": [],
            "composition_rows": [],
            "metric_rows": [],
            "ranking_rows": [],
        }
        for combo in combos
    }

    train_label_records: Sequence[LabelRecord] = all_train_records
    if balance_train:
        train_label_records = balance_label_records(all_train_records, seed=seed, max_per_class=max_train_per_class)
    train_records = vectorize_selected(train_label_records)

    pooled_test_label_records: List[LabelRecord] = []
    for week_idx, week in enumerate(test_weeks):
        label_records = all_test_records_by_week[week]
        if balance_test:
            label_records = balance_label_records(
                label_records,
                seed=seed + week_idx + 1,
                max_per_class=max_test_per_class,
            )
        pooled_test_label_records.extend(label_records)
    pooled_test_records = vectorize_selected(pooled_test_label_records)

    train_samples = as_samples(train_records)
    test_samples = as_samples(pooled_test_records)
    y_train = labels_array(train_samples)
    y_test = labels_array(test_samples)
    train_features = {
        group: select_feature_group(train_samples, group)
        for group in feature_groups
    }
    test_features = {
        group: select_feature_group(test_samples, group)
        for group in feature_groups
    }

    X_train_all_raw = train_features["all"]
    X_test_all_raw = test_features["all"]
    all_scaler = fit_scaler(X_train_all_raw)
    X_train_all_scaled = transform_with_scaler(all_scaler, X_train_all_raw)
    X_test_all_scaled = transform_with_scaler(all_scaler, X_test_all_raw)

    density_score_cache: Dict[int, np.ndarray] = {
        k_density: compute_density_scores(X_train_all_scaled, X_test_all_scaled, k_density=k_density)
        for k_density in k_values
    }
    density_bin_cache: Dict[tuple[int, float, float], tuple[np.ndarray, Dict[str, float]]] = {
        (k_density, low_q, high_q): assign_density_bins(
            density_score_cache[k_density],
            low_quantile=low_q,
            high_quantile=high_q,
        )
        for k_density in k_values
        for (low_q, high_q) in quantile_pairs
    }

    group_artifacts: Dict[str, Dict[str, object]] = {}
    for group in feature_groups:
        X_train_raw = train_features[group]
        X_test_raw = test_features[group]
        scaler = fit_scaler(X_train_raw)
        X_train_scaled = transform_with_scaler(scaler, X_train_raw)
        X_test_scaled = transform_with_scaler(scaler, X_test_raw)
        clf = fit_classifier(
            X_train_scaled,
            y_train,
            seed=seed,
            model_type=model_type,
            lgbm_n_jobs=lgbm_n_jobs,
            rf_n_estimators=rf_n_estimators,
            rf_n_jobs=rf_n_jobs,
        )
        sample_mix = compute_sample_mix_at_k(X_test_scaled, y_test, k=mix_k)
        group_artifacts[group] = {
            "X_test_raw": X_test_raw,
            "X_test_scaled": X_test_scaled,
            "clf": clf,
            "sample_mix": sample_mix,
        }

    for combo in combos:
        k_density, low_q, high_q = combo
        density_scores = density_score_cache[k_density]
        density_bins, thresholds = density_bin_cache[combo]
        density_rows = results_by_combo[combo]["density_rows"]
        composition_rows = results_by_combo[combo]["composition_rows"]
        metric_rows = results_by_combo[combo]["metric_rows"]
        ranking_rows = results_by_combo[combo]["ranking_rows"]

        if save_density_rows:
            for idx, record in enumerate(pooled_test_records):
                density_rows.append(
                    {
                        "seed": seed,
                        "sample_index": idx,
                        "sha256": record.sample.sha256,
                        "week": record.week,
                        "label": record.sample.label,
                        "packed": int(record.packed),
                        "density_score": float(density_scores[idx]),
                        "density_bin": str(density_bins[idx]),
                        "density_q25": thresholds["q25"],
                        "density_q75": thresholds["q75"],
                        "density_low_quantile": thresholds["low_quantile"],
                        "density_high_quantile": thresholds["high_quantile"],
                        "density_low_threshold": thresholds["low_threshold"],
                        "density_high_threshold": thresholds["high_threshold"],
                    }
                )

        for density_bin in ["high_density", "mid_density", "low_density"]:
            selected = [record for record, bin_name in zip(pooled_test_records, density_bins) if bin_name == density_bin]
            composition_rows.append(composition_row(seed, density_bin, selected))

        per_bin_group_metrics: Dict[str, Dict[str, Dict[str, float]]] = {
            density_bin: {} for density_bin in ["high_density", "mid_density", "low_density"]
        }
        for group in feature_groups:
            artifact = group_artifacts[group]
            X_test_raw = artifact["X_test_raw"]  # type: ignore[assignment]
            X_test_scaled = artifact["X_test_scaled"]  # type: ignore[assignment]
            clf = artifact["clf"]
            sample_mix = artifact["sample_mix"]  # type: ignore[assignment]
            for density_bin in ["high_density", "mid_density", "low_density"]:
                mask = density_bins == density_bin
                X_bin_raw = X_test_raw[mask]
                X_bin_scaled = X_test_scaled[mask]
                y_bin = y_test[mask]
                row = {
                    "seed": seed,
                    "density_bin": density_bin,
                    "feature_group": group,
                    "n_samples": int(mask.sum()),
                    "auc": compute_auc(clf, X_bin_scaled, y_bin),
                    "mix_at_10": float(sample_mix[mask].mean()) if mask.any() else float("nan"),
                    "js_divergence": js_divergence_feature_mass(X_bin_raw, y_bin),
                }
                metric_rows.append(row)
                per_bin_group_metrics[density_bin][group] = {
                    "auc": float(row["auc"]),
                    "mix_at_10": float(row["mix_at_10"]),
                    "js_divergence": float(row["js_divergence"]),
                }

        for density_bin in ["high_density", "mid_density", "low_density"]:
            metrics_for_bin = per_bin_group_metrics[density_bin]
            auc_rank = feature_rankings({group: metrics["auc"] for group, metrics in metrics_for_bin.items()})
            mix_rank = feature_rankings(
                {group: metrics["mix_at_10"] for group, metrics in metrics_for_bin.items()},
                higher_is_better=False,
            )
            js_rank = feature_rankings({group: metrics["js_divergence"] for group, metrics in metrics_for_bin.items()})
            auc_mix_conflicts = rank_conflict_count(auc_rank, mix_rank)
            auc_js_conflicts = rank_conflict_count(auc_rank, js_rank)

            for group in feature_groups:
                ranking_rows.append(
                    {
                        "seed": seed,
                        "density_bin": density_bin,
                        "feature_group": group,
                        "auc_rank": auc_rank[group],
                        "mix_at_10_rank": mix_rank[group],
                        "js_divergence_rank": js_rank[group],
                        "auc_vs_mix_conflicts": auc_mix_conflicts,
                        "auc_vs_js_conflicts": auc_js_conflicts,
                    }
                )

    return results_by_combo


def run_from_config(config: dict) -> dict:
    data_root = Path(config["data_root"])
    platform = str(config["platform"])
    if platform != "Win32":
        raise ValueError("This first RQ3 pipeline is currently scoped to Win32.")

    packer_filter = str(config.get("packer_filter", "all"))
    train_weeks = list(config["train_weeks"])
    test_weeks = list(config["test_weeks"])
    seeds = [int(seed) for seed in config["seeds"]]
    feature_groups = list(config.get("feature_groups", FEATURE_GROUPS))
    mix_k = int(config.get("mix_k", 10))
    k_density_values = config.get("k_density_values")
    if k_density_values is None:
        k_values = [int(config.get("k_density", 10))]
    else:
        k_values = [int(value) for value in k_density_values]
    model_type = str(config.get("model_type", "logistic_regression"))
    lgbm_n_jobs = int(config.get("lgbm_n_jobs", 2 if model_type == "lightgbm" else 1))
    rf_n_estimators = int(config.get("rf_n_estimators", 500))
    rf_n_jobs = int(config.get("rf_n_jobs", -1 if model_type == "random_forest" else 1))
    num_workers = int(config.get("num_workers", 1))
    save_density_rows = bool(config.get("save_density_rows", False))
    balance_train = bool(config.get("balance_train", True))
    balance_test = bool(config.get("balance_test", True))
    max_train_per_class = config.get("max_train_per_class")
    max_test_per_class = config.get("max_test_per_class")
    show_progress = bool(config.get("show_progress", True))
    low_quantile = float(config.get("low_quantile", 0.25))
    high_quantile = float(config.get("high_quantile", 0.75))
    quantile_pairs_raw = config.get("quantile_pairs")
    if quantile_pairs_raw is None:
        quantile_pairs: List[tuple[float, float]] = [(low_quantile, high_quantile)]
    else:
        quantile_pairs = [(float(pair[0]), float(pair[1])) for pair in quantile_pairs_raw]

    train_paths = week_paths(data_root, platform, train_weeks, "train")
    all_train_records = load_label_records_from_paths(train_paths, packer_filter=packer_filter)
    all_test_records_by_week = {
        week: load_label_records_from_jsonl(
            week_paths(data_root, platform, [week], "test")[0],
            packer_filter=packer_filter,
        )
        for week in test_weeks
    }

    if num_workers < 1:
        raise ValueError("num_workers must be >= 1.")

    combos: List[tuple[int, float, float]] = [
        (k_density, low_q, high_q) for k_density in k_values for (low_q, high_q) in quantile_pairs
    ]
    results_by_combo: Dict[tuple[int, float, float], Dict[str, List[Dict[str, object]]]] = {
        combo: {
            "density_rows": [],
            "composition_rows": [],
            "metric_rows": [],
            "ranking_rows": [],
        }
        for combo in combos
    }

    if num_workers == 1:
        seed_results_iter: Iterable[Dict[tuple[int, float, float], Dict[str, List[Dict[str, object]]]]] = (
            process_seed_block1(
                seed=seed,
                all_train_records=all_train_records,
                all_test_records_by_week=all_test_records_by_week,
                test_weeks=test_weeks,
                feature_groups=feature_groups,
                k_values=k_values,
                quantile_pairs=quantile_pairs,
                mix_k=mix_k,
                model_type=model_type,
                lgbm_n_jobs=lgbm_n_jobs,
                rf_n_estimators=rf_n_estimators,
                rf_n_jobs=rf_n_jobs,
                balance_train=balance_train,
                balance_test=balance_test,
                max_train_per_class=max_train_per_class,
                max_test_per_class=max_test_per_class,
                save_density_rows=save_density_rows,
            )
            for seed in (tqdm(seeds, desc="RQ3 Block1 seeds") if show_progress else seeds)
        )
        for seed_results in seed_results_iter:
            for combo in combos:
                for key in ["density_rows", "composition_rows", "metric_rows", "ranking_rows"]:
                    results_by_combo[combo][key].extend(seed_results[combo][key])
    else:
        progress = tqdm(total=len(seeds), desc="RQ3 Block1 seeds") if show_progress else None
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    process_seed_block1,
                    seed=seed,
                    all_train_records=all_train_records,
                    all_test_records_by_week=all_test_records_by_week,
                    test_weeks=test_weeks,
                    feature_groups=feature_groups,
                    k_values=k_values,
                    quantile_pairs=quantile_pairs,
                    mix_k=mix_k,
                    model_type=model_type,
                    lgbm_n_jobs=lgbm_n_jobs,
                    rf_n_estimators=rf_n_estimators,
                    rf_n_jobs=rf_n_jobs,
                    balance_train=balance_train,
                    balance_test=balance_test,
                    max_train_per_class=max_train_per_class,
                    max_test_per_class=max_test_per_class,
                    save_density_rows=save_density_rows,
                )
                for seed in seeds
            ]
            for future in as_completed(futures):
                seed_results = future.result()
                for combo in combos:
                    for key in ["density_rows", "composition_rows", "metric_rows", "ranking_rows"]:
                        results_by_combo[combo][key].extend(seed_results[combo][key])
                if progress is not None:
                    progress.update(1)
        if progress is not None:
            progress.close()

    final_results_by_combo: Dict[tuple[int, float, float], dict] = {}
    for combo in combos:
        k_density, low_q, high_q = combo
        density_rows = results_by_combo[combo]["density_rows"]
        composition_rows = results_by_combo[combo]["composition_rows"]
        metric_rows = results_by_combo[combo]["metric_rows"]
        ranking_rows = results_by_combo[combo]["ranking_rows"]
        final_results_by_combo[combo] = {
            "config": {**config, "k_density": k_density, "low_quantile": low_q, "high_quantile": high_q},
            "metric_notes": {
                "auc": "higher is better",
                "mix_at_10": "lower is better; computed query-by-query on full evaluation pool then aggregated by density bin",
                "js_divergence": "higher means larger class-conditional distribution difference within the density bin",
                "density_score": f"negative mean cosine distance to k={k_density} nearest training neighbors; higher means denser local support",
                "density_quantiles": f"low={low_q:.2f}, high={high_q:.2f}",
                "auc_vs_mix_conflicts": "pairwise ordering conflicts between AUC ranking and Mix@10 ranking",
                "auc_vs_js_conflicts": "pairwise ordering conflicts between AUC ranking and JS divergence ranking",
            },
            "density_rows": density_rows,
            "composition_rows": composition_rows,
            "metric_rows": metric_rows,
            "ranking_rows": ranking_rows,
            "aggregate_metric_rows": aggregate_rows(
                metric_rows,
                group_keys=["density_bin", "feature_group"],
                metric_keys=["auc", "mix_at_10", "js_divergence"],
            ),
            "aggregate_composition_rows": aggregate_rows(
                composition_rows,
                group_keys=["density_bin"],
                metric_keys=["n_samples", "n_benign", "n_malware", "benign_to_malware_ratio", "n_packed_malware", "packed_ratio"],
            ),
            "aggregate_ranking_rows": aggregate_rows(
                ranking_rows,
                group_keys=["density_bin", "feature_group"],
                metric_keys=["auc_rank", "mix_at_10_rank", "js_divergence_rank", "auc_vs_mix_conflicts", "auc_vs_js_conflicts"],
            ),
        }

    if len(combos) == 1:
        return final_results_by_combo[combos[0]]
    return {
        "results_by_combo": final_results_by_combo,
        "config": config,
        "multi_k": len(k_values) > 1,
        "multi_quantile": len(quantile_pairs) > 1,
    }


def save_results(results: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if results["density_rows"]:
        write_csv(output_dir / "density_rows.csv", list(results["density_rows"][0].keys()), results["density_rows"])
    if results["composition_rows"]:
        write_csv(output_dir / "composition_rows.csv", list(results["composition_rows"][0].keys()), results["composition_rows"])
    if results["metric_rows"]:
        write_csv(output_dir / "metric_rows.csv", list(results["metric_rows"][0].keys()), results["metric_rows"])
    if results["ranking_rows"]:
        write_csv(output_dir / "ranking_rows.csv", list(results["ranking_rows"][0].keys()), results["ranking_rows"])
    if results["aggregate_metric_rows"]:
        write_csv(output_dir / "aggregate_metric_rows.csv", list(results["aggregate_metric_rows"][0].keys()), results["aggregate_metric_rows"])
    if results["aggregate_composition_rows"]:
        write_csv(output_dir / "aggregate_composition_rows.csv", list(results["aggregate_composition_rows"][0].keys()), results["aggregate_composition_rows"])
    if results["aggregate_ranking_rows"]:
        write_csv(output_dir / "aggregate_ranking_rows.csv", list(results["aggregate_ranking_rows"][0].keys()), results["aggregate_ranking_rows"])
    summary = {
        "config": results["config"],
        "metric_notes": results["metric_notes"],
        "n_density_rows": len(results["density_rows"]),
        "n_composition_rows": len(results["composition_rows"]),
        "n_metric_rows": len(results["metric_rows"]),
        "n_ranking_rows": len(results["ranking_rows"]),
    }
    (output_dir / "results_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RQ3 Block 1 density-stratified reliability analysis.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    data_root = Path(config["data_root"])
    if not data_root.is_absolute():
        candidates = [
            (args.config.parent / data_root).resolve(),
            (Path.cwd() / data_root).resolve(),
            (Path(__file__).resolve().parents[4] / data_root).resolve(),
        ]
        for candidate in candidates:
            if candidate.exists():
                data_root = candidate
                break
    config["data_root"] = str(data_root)

    results = run_from_config(config)
    if "results_by_combo" in results:
        multi_k = bool(results["multi_k"])
        multi_quantile = bool(results["multi_quantile"])
        for (k_density, low_q, high_q), combo_results in results["results_by_combo"].items():
            out_dir = args.output_dir
            if multi_k:
                out_dir = out_dir / f"k_{k_density}"
            if multi_quantile:
                out_dir = out_dir / f"q{int(low_q * 100):02d}_{int(high_q * 100):02d}"
            save_results(combo_results, out_dir)
    else:
        save_results(results, args.output_dir)


if __name__ == "__main__":
    main()
