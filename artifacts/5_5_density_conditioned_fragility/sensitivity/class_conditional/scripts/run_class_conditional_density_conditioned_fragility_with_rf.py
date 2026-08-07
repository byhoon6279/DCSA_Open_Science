#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
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
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    def tqdm(iterable=None, *args, **kwargs):  # type: ignore[override]
        return iterable if iterable is not None else []

# NOTE (Open Science relocation): this script is the exact patched runner used to
# produce the bundled D4 class-conditional sensitivity results in ../results/. The
# original runner resolved the shared `common` module via an RQ1-named path
# search; that has been replaced below with a direct path into this package's
# artifacts/shared/, matching the convention used by ../../../scripts/common/.
sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "artifacts" / "shared"))

from common import (  # type: ignore
    build_sample,
    labels_array,
    matches_packer_filter,
    select_feature_group,
    stabilize_features,
    summarize,
    week_paths,
)

"""
RQ3 Block 2: Density-Conditioned Perturbation Fragility

Main protocol:
- weekly files define only the temporal train/test boundary
- all train weeks are pooled into one train pool
- one balanced train subset is sampled per seed
- one model is trained per seed and feature subset
- all test weeks are pooled into one evaluation pool
- density is computed separately in each subset space
- perturbations are evaluated by density bin without retraining
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
    module=r"sklearn\..*",
)

DENSITY_BINS = ["high_density", "mid_density", "low_density"]


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
    if model_type == "logistic_regression":
        clf: Any = LogisticRegression(
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


def zero_scaled_template(scaler: StandardScaler, n_features: int) -> np.ndarray:
    zero_row = np.zeros((1, n_features), dtype=np.float32)
    return transform_with_scaler(scaler, zero_row)[0]


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
        return np.abs(np.asarray(clf.coef_, dtype=np.float32).ravel())

    if model_type == "random_forest":
        if importance_method != "permutation":
            raise ValueError(
                "Random Forest in RQ3 Block 2 currently supports importance_method='permutation' only."
            )
        if X_val_scaled is None or y_val is None:
            raise ValueError("Permutation importance requires a validation split.")
        result = permutation_importance(
            clf,
            np.asarray(X_val_scaled, dtype=np.float32),
            np.asarray(y_val),
            scoring="roc_auc",
            n_repeats=permutation_repeats,
            random_state=seed,
            n_jobs=1,
        )
        return np.asarray(result.importances_mean, dtype=np.float32)

    if model_type != "lightgbm":
        raise ValueError(f"Unsupported model_type: {model_type}")

    if importance_method == "permutation":
        if X_val_scaled is None or y_val is None:
            raise ValueError("Permutation importance requires a validation split.")
        result = permutation_importance(
            clf,
            np.asarray(X_val_scaled, dtype=np.float32),
            np.asarray(y_val),
            scoring="roc_auc",
            n_repeats=permutation_repeats,
            random_state=seed,
            n_jobs=1,
        )
        return np.asarray(result.importances_mean, dtype=np.float32)

    if importance_method == "gain":
        booster = getattr(clf, "booster_", None)
        if booster is not None:
            return np.asarray(booster.feature_importance(importance_type="gain"), dtype=np.float32)
        return np.asarray(getattr(clf, "feature_importances_"), dtype=np.float32)

    if importance_method == "shap":
        raise NotImplementedError(
            "importance_method='shap' is intentionally not implemented here yet. "
            "The current RQ3 Block 2 LightGBM extension supports 'permutation' "
            "(default) and 'gain'."
        )

    raise ValueError(
        "Unsupported importance_method for LightGBM: "
        f"{importance_method}. Use 'permutation' or 'gain'."
    )


def apply_zero_mask_scaled(
    X_scaled: np.ndarray,
    indices: np.ndarray,
    zero_scaled_values: np.ndarray,
) -> np.ndarray:
    masked = np.array(X_scaled, copy=True)
    masked[:, indices] = zero_scaled_values[indices]
    return masked


def compute_auc_from_proba(y_true: np.ndarray, proba: np.ndarray) -> float:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, proba))


def evaluate_flip_metrics(
    baseline_proba: np.ndarray,
    perturbed_proba: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    if len(y_true) == 0:
        return {
            "flip_rate": float("nan"),
            "benign_to_malware_rate_overall": float("nan"),
            "malware_to_benign_rate_overall": float("nan"),
            "benign_to_malware_rate_cond": float("nan"),
            "malware_to_benign_rate_cond": float("nan"),
            "mean_abs_probability_shift": float("nan"),
            "mean_signed_probability_shift": float("nan"),
            "mean_margin_shift": float("nan"),
        }

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
        "benign_to_malware_rate_overall": float(benign_to_malware.mean()),
        "malware_to_benign_rate_overall": float(malware_to_benign.mean()),
        "benign_to_malware_rate_cond": (
            float(benign_to_malware.sum() / benign_mask.sum()) if benign_mask.sum() else float("nan")
        ),
        "malware_to_benign_rate_cond": (
            float(malware_to_benign.sum() / malware_mask.sum()) if malware_mask.sum() else float("nan")
        ),
        "mean_abs_probability_shift": float(np.mean(np.abs(perturbed_proba - baseline_proba))),
        "mean_signed_probability_shift": float(np.mean(perturbed_proba - baseline_proba)),
        "mean_margin_shift": float(np.mean(perturbed_margin - baseline_margin)),
    }


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


def assign_class_conditional_density_bins(
    scores: np.ndarray,
    labels: np.ndarray,
    low_quantile: float = 0.25,
    high_quantile: float = 0.75,
) -> tuple[np.ndarray, Dict[str, float]]:
    if not (0.0 < low_quantile < high_quantile < 1.0):
        raise ValueError("Density quantiles must satisfy 0 < low_quantile < high_quantile < 1.")
    bins = np.full(len(scores), "mid_density", dtype=object)
    thresholds: Dict[str, float] = {
        "low_quantile": float(low_quantile),
        "high_quantile": float(high_quantile),
    }
    for label_value in sorted(np.unique(labels)):
        class_mask = labels == label_value
        class_scores = scores[class_mask]
        q_low = float(np.quantile(class_scores, low_quantile))
        q_high = float(np.quantile(class_scores, high_quantile))
        thresholds[f"label_{int(label_value)}_q_low"] = q_low
        thresholds[f"label_{int(label_value)}_q_high"] = q_high
        bins[class_mask] = "mid_density"
        bins[class_mask & (scores <= q_low)] = "low_density"
        bins[class_mask & (scores >= q_high)] = "high_density"
    return bins, thresholds


def as_samples(records: Sequence[SampleRecord]) -> List[object]:
    return [record.sample for record in records]


def composition_row(seed: int, feature_group: str, density_bin: str, records: Sequence[SampleRecord]) -> Dict[str, object]:
    benign = sum(record.sample.label == 0 for record in records)
    malware = sum(record.sample.label == 1 for record in records)
    packed_malware = sum(record.sample.label == 1 and record.packed for record in records)
    packed_ratio = float(packed_malware / malware) if malware else float("nan")
    week_counts: Dict[str, int] = {}
    for record in records:
        week_counts[record.week] = week_counts.get(record.week, 0) + 1
    return {
        "seed": seed,
        "feature_group": feature_group,
        "density_bin": density_bin,
        "n_samples": len(records),
        "n_benign": benign,
        "n_malware": malware,
        "benign_to_malware_ratio": float(benign / malware) if malware else float("nan"),
        "n_packed_malware": packed_malware,
        "packed_ratio": packed_ratio,
        "week_distribution_json": json.dumps(week_counts, sort_keys=True),
    }


def aggregate_rows(
    rows: List[Dict[str, object]],
    group_keys: Sequence[str],
    metric_keys: Sequence[str],
    passthrough_keys: Sequence[str] | None = None,
) -> List[Dict[str, object]]:
    grouped: Dict[tuple, List[Dict[str, object]]] = {}
    for row in rows:
        key = tuple(row[key_name] for key_name in group_keys)
        grouped.setdefault(key, []).append(row)

    out: List[Dict[str, object]] = []
    passthrough_keys = list(passthrough_keys or [])
    for key, bucket in sorted(grouped.items()):
        row: Dict[str, object] = {name: value for name, value in zip(group_keys, key)}
        row["n_rows"] = len(bucket)
        for key_name in passthrough_keys:
            row[key_name] = bucket[0][key_name]
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


def build_amplification_rows(seed_summary_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    lookup = {
        (
            int(row["seed"]),
            str(row["feature_group"]),
            str(row["perturbation_type"]),
            float(row["strength"]),
            str(row["density_bin"]),
        ): row
        for row in seed_summary_rows
    }

    keys = sorted(
        {
            (
                int(row["seed"]),
                str(row["feature_group"]),
                str(row["perturbation_type"]),
                float(row["strength"]),
            )
            for row in seed_summary_rows
        }
    )

    rows: List[Dict[str, object]] = []
    for seed, feature_group, perturbation_type, strength in keys:
        high = lookup.get((seed, feature_group, perturbation_type, strength, "high_density"))
        low = lookup.get((seed, feature_group, perturbation_type, strength, "low_density"))
        mid = lookup.get((seed, feature_group, perturbation_type, strength, "mid_density"))
        if high is None or low is None or mid is None:
            continue

        high_flip = float(high["flip_rate_mean"])
        mid_flip = float(mid["flip_rate_mean"])
        low_flip = float(low["flip_rate_mean"])
        high_delta_auc = float(high["delta_auc_mean"])
        mid_delta_auc = float(mid["delta_auc_mean"])
        low_delta_auc = float(low["delta_auc_mean"])
        high_auc_drop = float(high["auc_drop_mean"])
        mid_auc_drop = float(mid["auc_drop_mean"])
        low_auc_drop = float(low["auc_drop_mean"])

        rows.append(
            {
                "seed": seed,
                "feature_group": feature_group,
                "perturbation_type": perturbation_type,
                "strength": strength,
                "high_flip_rate": high_flip,
                "mid_flip_rate": mid_flip,
                "low_flip_rate": low_flip,
                "high_delta_auc": high_delta_auc,
                "mid_delta_auc": mid_delta_auc,
                "low_delta_auc": low_delta_auc,
                "high_auc_drop": high_auc_drop,
                "mid_auc_drop": mid_auc_drop,
                "low_auc_drop": low_auc_drop,
                "delta_flip_density": low_flip - high_flip,
                "flip_density_ratio": float(low_flip / high_flip) if high_flip > 0 else float("nan"),
                "delta_auc_density": low_delta_auc - high_delta_auc,
                "auc_drop_density": low_auc_drop - high_auc_drop,
            }
        )
    return rows


def process_seed_block2(
    *,
    seed: int,
    all_train_records: Sequence[LabelRecord],
    all_test_records_by_week: Dict[str, List[LabelRecord]],
    test_weeks: Sequence[str],
    feature_groups: Sequence[str],
    strengths: Sequence[float],
    random_trials: int,
    threshold: float,
    k_values: Sequence[int],
    model_type: str,
    importance_method: str,
    validation_fraction: float,
    permutation_repeats: int,
    lgbm_n_jobs: int,
    rf_n_estimators: int,
    rf_n_jobs: int,
    balance_train: bool,
    balance_test: bool,
    max_train_per_class: int | None,
    max_test_per_class: int | None,
    quantile_pairs: Sequence[tuple[float, float]],
    save_class_conditional_rows: bool,
    class_conditional_low_quantile: float,
    class_conditional_high_quantile: float,
) -> Dict[tuple[int, float, float], Dict[str, List[Dict[str, object]]]]:
    combos: List[tuple[int, float, float]] = [
        (k_density, low_q, high_q) for k_density in k_values for (low_q, high_q) in quantile_pairs
    ]
    results_by_combo: Dict[tuple[int, float, float], Dict[str, List[Dict[str, object]]]] = {
        combo: {
            "density_rows": [],
            "composition_rows": [],
            "trial_rows": [],
            "class_conditional_composition_rows": [],
            "class_conditional_trial_rows": [],
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

    for group in feature_groups:
        X_train_raw = select_feature_group(train_samples, group)
        X_test_raw = select_feature_group(test_samples, group)
        needs_validation_split = (
            model_type in {"lightgbm", "random_forest"} and importance_method == "permutation"
        )
        if needs_validation_split:
            train_idx, val_idx = split_train_validation_indices(
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

        scaler = fit_scaler(X_train_fit_raw)
        X_train_scaled = transform_with_scaler(scaler, X_train_fit_raw)
        X_test_scaled = transform_with_scaler(scaler, X_test_raw)
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
        zero_scaled_values = zero_scaled_template(scaler, X_train_raw.shape[1])
        baseline_proba = clf.predict_proba(X_test_scaled)[:, 1]

        density_score_cache: Dict[int, np.ndarray] = {
            k_density: compute_density_scores(X_train_scaled, X_test_scaled, k_density=k_density)
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

        perturbation_cache: List[Dict[str, object]] = []
        for strength in strengths:
            important_indices = select_top_indices(importance_scores, strength)
            n_mask = len(important_indices)
            important_scaled = apply_zero_mask_scaled(
                X_test_scaled,
                important_indices,
                zero_scaled_values,
            )
            perturbation_cache.append(
                {
                    "perturbation_type": "important",
                    "strength": strength,
                    "trial_id": 0,
                    "n_masked_features": n_mask,
                    "proba": clf.predict_proba(important_scaled)[:, 1],
                }
            )

            for trial_id in range(1, random_trials + 1):
                seed_material = f"{seed}-{group}-{strength:.5f}-{trial_id}".encode("utf-8")
                rng_seed = int(hashlib.sha256(seed_material).hexdigest()[:16], 16) % (2**32)
                rng = np.random.default_rng(rng_seed)
                random_indices = rng.choice(len(importance_scores), size=n_mask, replace=False)
                random_scaled = apply_zero_mask_scaled(
                    X_test_scaled,
                    random_indices,
                    zero_scaled_values,
                )
                perturbation_cache.append(
                    {
                        "perturbation_type": "random",
                        "strength": strength,
                        "trial_id": trial_id,
                        "n_masked_features": n_mask,
                        "proba": clf.predict_proba(random_scaled)[:, 1],
                    }
                )

        for combo in combos:
            k_density, low_q, high_q = combo
            density_scores = density_score_cache[k_density]
            density_bins, thresholds = density_bin_cache[combo]
            density_rows = results_by_combo[combo]["density_rows"]
            composition_rows = results_by_combo[combo]["composition_rows"]
            trial_rows = results_by_combo[combo]["trial_rows"]
            class_conditional_composition_rows = results_by_combo[combo]["class_conditional_composition_rows"]
            class_conditional_trial_rows = results_by_combo[combo]["class_conditional_trial_rows"]

            for idx, record in enumerate(pooled_test_records):
                density_rows.append(
                    {
                        "seed": seed,
                        "feature_group": group,
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

            for density_bin in DENSITY_BINS:
                selected = [record for record, bin_name in zip(pooled_test_records, density_bins) if bin_name == density_bin]
                composition_rows.append(composition_row(seed, group, density_bin, selected))

            class_conditional_bins = None
            class_conditional_thresholds: Dict[str, float] = {}
            if save_class_conditional_rows:
                class_conditional_bins, class_conditional_thresholds = assign_class_conditional_density_bins(
                    density_scores,
                    y_test,
                    low_quantile=class_conditional_low_quantile,
                    high_quantile=class_conditional_high_quantile,
                )
                for density_bin in DENSITY_BINS:
                    selected = [
                        record
                        for record, bin_name in zip(pooled_test_records, class_conditional_bins)
                        if bin_name == density_bin
                    ]
                    row = composition_row(seed, group, density_bin, selected)
                    row["density_definition"] = "class_conditional"
                    row["thresholds_json"] = json.dumps(class_conditional_thresholds, sort_keys=True)
                    class_conditional_composition_rows.append(row)

            for cached in perturbation_cache:
                perturbation_type = str(cached["perturbation_type"])
                strength = float(cached["strength"])
                trial_id = int(cached["trial_id"])
                n_mask = int(cached["n_masked_features"])
                perturbed_proba = cached["proba"]  # type: ignore[assignment]
                for density_bin in DENSITY_BINS:
                    mask = density_bins == density_bin
                    y_bin = y_test[mask]
                    baseline_bin = baseline_proba[mask]
                    perturbed_bin = perturbed_proba[mask]
                    metrics = evaluate_flip_metrics(baseline_bin, perturbed_bin, y_bin, threshold)
                    baseline_auc = compute_auc_from_proba(y_bin, baseline_bin)
                    perturbed_auc = compute_auc_from_proba(y_bin, perturbed_bin)
                    trial_rows.append(
                        {
                            "seed": seed,
                            "feature_group": group,
                            "density_bin": density_bin,
                            "perturbation_type": perturbation_type,
                            "strength": strength,
                            "trial_id": trial_id,
                            "n_samples": int(mask.sum()),
                            "n_masked_features": n_mask,
                            "baseline_auc": baseline_auc,
                            "perturbed_auc": perturbed_auc,
                            "delta_auc": baseline_auc - perturbed_auc,
                            "auc_drop": baseline_auc - perturbed_auc,
                            **metrics,
                        }
                    )
                    if save_class_conditional_rows and class_conditional_bins is not None:
                        cc_mask = class_conditional_bins == density_bin
                        cc_y_bin = y_test[cc_mask]
                        cc_baseline_bin = baseline_proba[cc_mask]
                        cc_perturbed_bin = perturbed_proba[cc_mask]
                        cc_metrics = evaluate_flip_metrics(cc_baseline_bin, cc_perturbed_bin, cc_y_bin, threshold)
                        cc_baseline_auc = compute_auc_from_proba(cc_y_bin, cc_baseline_bin)
                        cc_perturbed_auc = compute_auc_from_proba(cc_y_bin, cc_perturbed_bin)
                        class_conditional_trial_rows.append(
                            {
                                "seed": seed,
                                "feature_group": group,
                                "density_bin": density_bin,
                                "density_definition": "class_conditional",
                                "thresholds_json": json.dumps(class_conditional_thresholds, sort_keys=True),
                                "perturbation_type": perturbation_type,
                                "strength": strength,
                                "trial_id": trial_id,
                                "n_samples": int(cc_mask.sum()),
                                "n_masked_features": n_mask,
                                "baseline_auc": cc_baseline_auc,
                                "perturbed_auc": cc_perturbed_auc,
                                "delta_auc": cc_baseline_auc - cc_perturbed_auc,
                                "auc_drop": cc_baseline_auc - cc_perturbed_auc,
                                **cc_metrics,
                            }
                        )

    return results_by_combo


def run_from_config(config: dict) -> dict:
    data_root = Path(config["data_root"])
    platform = str(config["platform"])
    if platform != "Win32":
        raise ValueError("RQ3 Block 2 is currently scoped to Win32.")

    packer_filter = str(config.get("packer_filter", "all"))
    train_weeks = list(config["train_weeks"])
    test_weeks = list(config["test_weeks"])
    seeds = [int(seed) for seed in config["seeds"]]
    feature_groups = list(config["feature_groups"])
    strengths = [float(value) for value in config["perturbation_strengths"]]
    random_trials = int(config["random_trials"])
    threshold = float(config.get("decision_threshold", 0.5))
    k_density_values = config.get("k_density_values")
    if k_density_values is None:
        k_values = [int(config.get("k_density", 10))]
    else:
        k_values = [int(value) for value in k_density_values]
    balance_train = bool(config.get("balance_train", True))
    balance_test = bool(config.get("balance_test", True))
    max_train_per_class = config.get("max_train_per_class")
    max_test_per_class = config.get("max_test_per_class")
    show_progress = bool(config.get("show_progress", True))
    num_workers = int(config.get("num_workers", 1))
    model_type = str(config.get("model_type", "logistic_regression"))
    importance_method = str(
        config.get(
            "importance_method",
            "coefficient" if model_type == "logistic_regression" else "permutation",
        )
    )
    validation_fraction = float(config.get("validation_fraction", 0.2))
    permutation_repeats = int(config.get("permutation_repeats", 3))
    lgbm_n_jobs = int(config.get("lgbm_n_jobs", 2 if model_type == "lightgbm" else 1))
    rf_n_estimators = int(config.get("rf_n_estimators", 500))
    rf_n_jobs = int(config.get("rf_n_jobs", -1 if model_type == "random_forest" else 1))
    low_quantile = float(config.get("low_quantile", 0.25))
    high_quantile = float(config.get("high_quantile", 0.75))
    quantile_pairs_raw = config.get("quantile_pairs")
    if quantile_pairs_raw is None:
        quantile_pairs: List[tuple[float, float]] = [(low_quantile, high_quantile)]
    else:
        quantile_pairs = [(float(pair[0]), float(pair[1])) for pair in quantile_pairs_raw]
    save_class_conditional_rows = bool(config.get("save_class_conditional_rows", False))
    class_conditional_low_quantile = float(config.get("class_conditional_low_quantile", 0.25))
    class_conditional_high_quantile = float(config.get("class_conditional_high_quantile", 0.75))
    needs_validation_split = (
        model_type in {"lightgbm", "random_forest"} and importance_method == "permutation"
    )
    if model_type == "logistic_regression" and importance_method != "coefficient":
        raise ValueError(
            "Logistic Regression in RQ3 Block 2 currently requires "
            "importance_method='coefficient'."
        )
    if model_type == "lightgbm" and importance_method not in {"permutation", "gain", "shap"}:
        raise ValueError(
            "LightGBM in RQ3 Block 2 supports importance_method in "
            "{'permutation', 'gain', 'shap'}."
        )
    if model_type == "random_forest" and importance_method != "permutation":
        raise ValueError(
            "Random Forest in RQ3 Block 2 currently requires importance_method='permutation'."
        )
    if model_type == "lightgbm" and not (0.0 < validation_fraction < 1.0):
        raise ValueError("validation_fraction must be between 0 and 1.")
    if num_workers < 1:
        raise ValueError("num_workers must be >= 1.")

    train_paths = week_paths(data_root, platform, train_weeks, "train")
    all_train_records = load_label_records_from_paths(train_paths, packer_filter=packer_filter)
    all_test_records_by_week = {
        week: load_label_records_from_jsonl(
            week_paths(data_root, platform, [week], "test")[0],
            packer_filter=packer_filter,
        )
        for week in test_weeks
    }

    combos: List[tuple[int, float, float]] = [
        (k_density, low_q, high_q) for k_density in k_values for (low_q, high_q) in quantile_pairs
    ]
    results_by_combo: Dict[tuple[int, float, float], Dict[str, List[Dict[str, object]]]] = {
        combo: {
            "density_rows": [],
            "composition_rows": [],
            "trial_rows": [],
            "class_conditional_composition_rows": [],
            "class_conditional_trial_rows": [],
        }
        for combo in combos
    }

    if num_workers == 1:
        seed_results_iter: Iterable[Dict[tuple[int, float, float], Dict[str, List[Dict[str, object]]]]] = (
            process_seed_block2(
                seed=seed,
                all_train_records=all_train_records,
                all_test_records_by_week=all_test_records_by_week,
                test_weeks=test_weeks,
                feature_groups=feature_groups,
                strengths=strengths,
                random_trials=random_trials,
                threshold=threshold,
                k_values=k_values,
                model_type=model_type,
                importance_method=importance_method,
                validation_fraction=validation_fraction,
                permutation_repeats=permutation_repeats,
                lgbm_n_jobs=lgbm_n_jobs,
                rf_n_estimators=rf_n_estimators,
                rf_n_jobs=rf_n_jobs,
                balance_train=balance_train,
                balance_test=balance_test,
                max_train_per_class=max_train_per_class,
                max_test_per_class=max_test_per_class,
                quantile_pairs=quantile_pairs,
                save_class_conditional_rows=save_class_conditional_rows,
                class_conditional_low_quantile=class_conditional_low_quantile,
                class_conditional_high_quantile=class_conditional_high_quantile,
            )
            for seed in (tqdm(seeds, desc="RQ3 Block2 seeds") if show_progress else seeds)
        )
        for seed_results in seed_results_iter:
            for combo in combos:
                for key in [
                    "density_rows",
                    "composition_rows",
                    "trial_rows",
                    "class_conditional_composition_rows",
                    "class_conditional_trial_rows",
                ]:
                    results_by_combo[combo][key].extend(seed_results[combo][key])
    else:
        progress = tqdm(total=len(seeds), desc="RQ3 Block2 seeds") if show_progress else None
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    process_seed_block2,
                    seed=seed,
                    all_train_records=all_train_records,
                    all_test_records_by_week=all_test_records_by_week,
                    test_weeks=test_weeks,
                    feature_groups=feature_groups,
                    strengths=strengths,
                    random_trials=random_trials,
                    threshold=threshold,
                    k_values=k_values,
                    model_type=model_type,
                    importance_method=importance_method,
                    validation_fraction=validation_fraction,
                    permutation_repeats=permutation_repeats,
                    lgbm_n_jobs=lgbm_n_jobs,
                    rf_n_estimators=rf_n_estimators,
                    rf_n_jobs=rf_n_jobs,
                    balance_train=balance_train,
                    balance_test=balance_test,
                    max_train_per_class=max_train_per_class,
                    max_test_per_class=max_test_per_class,
                    quantile_pairs=quantile_pairs,
                    save_class_conditional_rows=save_class_conditional_rows,
                    class_conditional_low_quantile=class_conditional_low_quantile,
                    class_conditional_high_quantile=class_conditional_high_quantile,
                )
                for seed in seeds
            ]
            for future in as_completed(futures):
                seed_results = future.result()
                for combo in combos:
                    for key in [
                        "density_rows",
                        "composition_rows",
                        "trial_rows",
                        "class_conditional_composition_rows",
                        "class_conditional_trial_rows",
                    ]:
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
        trial_rows = results_by_combo[combo]["trial_rows"]
        class_conditional_composition_rows = results_by_combo[combo]["class_conditional_composition_rows"]
        class_conditional_trial_rows = results_by_combo[combo]["class_conditional_trial_rows"]
        seed_summary_rows = aggregate_rows(
            trial_rows,
            group_keys=["seed", "feature_group", "density_bin", "perturbation_type", "strength"],
            metric_keys=[
                "flip_rate",
                "benign_to_malware_rate_overall",
                "malware_to_benign_rate_overall",
                "benign_to_malware_rate_cond",
                "malware_to_benign_rate_cond",
                "mean_abs_probability_shift",
                "mean_signed_probability_shift",
                "mean_margin_shift",
                "baseline_auc",
                "perturbed_auc",
                "delta_auc",
                "auc_drop",
            ],
            passthrough_keys=["n_samples", "n_masked_features"],
        )
        amplification_rows = build_amplification_rows(seed_summary_rows)
        class_conditional_seed_summary_rows = aggregate_rows(
            class_conditional_trial_rows,
            group_keys=["seed", "feature_group", "density_bin", "density_definition", "perturbation_type", "strength"],
            metric_keys=[
                "flip_rate",
                "benign_to_malware_rate_overall",
                "malware_to_benign_rate_overall",
                "benign_to_malware_rate_cond",
                "malware_to_benign_rate_cond",
                "mean_abs_probability_shift",
                "mean_signed_probability_shift",
                "mean_margin_shift",
                "baseline_auc",
                "perturbed_auc",
                "delta_auc",
                "auc_drop",
            ],
            passthrough_keys=["n_samples", "n_masked_features", "thresholds_json"],
        ) if class_conditional_trial_rows else []
        class_conditional_amplification_rows = build_amplification_rows(class_conditional_seed_summary_rows)
        for row in class_conditional_amplification_rows:
            row["density_definition"] = "class_conditional"
        final_results_by_combo[combo] = {
            "config": {**config, "k_density": k_density, "low_quantile": low_q, "high_quantile": high_q},
            "metric_notes": {
                "density_score": f"negative mean cosine distance to k={k_density} nearest training neighbors; higher means denser local support",
                "flip_rate": "fraction of predictions that change after masking within the density bin",
                "delta_auc": "baseline_auc - perturbed_auc; larger positive values mean stronger degradation",
                "auc_drop": "alias of delta_auc for figure/table labeling: baseline_auc - perturbed_auc",
                "density_quantiles": f"low={low_q:.2f}, high={high_q:.2f}",
                "benign_to_malware_rate_overall": "fraction of all bin samples that flip from benign to malware",
                "malware_to_benign_rate_overall": "fraction of all bin samples that flip from malware to benign",
                "benign_to_malware_rate_cond": "fraction of benign samples in the bin that flip to malware",
                "malware_to_benign_rate_cond": "fraction of malware samples in the bin that flip to benign",
                "delta_flip_density": "low_density flip rate - high_density flip rate; larger positive values mean stronger low-density amplification",
                "flip_density_ratio": "low_density flip rate / high_density flip rate; supplementary amplification metric",
                "density_protocol": "subset-specific density bins computed in the same feature space where masking is applied",
                "model_type": model_type,
                "importance_method": importance_method,
            },
            "density_rows": density_rows,
            "composition_rows": composition_rows,
            "trial_rows": trial_rows,
            "class_conditional_composition_rows": class_conditional_composition_rows,
            "class_conditional_trial_rows": class_conditional_trial_rows,
            "seed_summary_rows": seed_summary_rows,
            "amplification_rows": amplification_rows,
            "class_conditional_seed_summary_rows": class_conditional_seed_summary_rows,
            "class_conditional_amplification_rows": class_conditional_amplification_rows,
            "aggregate_composition_rows": aggregate_rows(
                composition_rows,
                group_keys=["feature_group", "density_bin"],
                metric_keys=[
                    "n_samples",
                    "n_benign",
                    "n_malware",
                    "benign_to_malware_ratio",
                    "n_packed_malware",
                    "packed_ratio",
                ],
            ),
            "aggregate_seed_summary_rows": aggregate_rows(
                seed_summary_rows,
                group_keys=["feature_group", "density_bin", "perturbation_type", "strength"],
                metric_keys=[
                    "n_samples",
                    "n_masked_features",
                    "flip_rate_mean",
                    "benign_to_malware_rate_overall_mean",
                    "malware_to_benign_rate_overall_mean",
                    "benign_to_malware_rate_cond_mean",
                    "malware_to_benign_rate_cond_mean",
                    "mean_abs_probability_shift_mean",
                    "mean_signed_probability_shift_mean",
                    "mean_margin_shift_mean",
                    "baseline_auc_mean",
                    "perturbed_auc_mean",
                    "delta_auc_mean",
                    "auc_drop_mean",
                ],
            ),
            "aggregate_amplification_rows": aggregate_rows(
                amplification_rows,
                group_keys=["feature_group", "perturbation_type", "strength"],
                metric_keys=[
                    "high_flip_rate",
                    "mid_flip_rate",
                    "low_flip_rate",
                    "high_delta_auc",
                    "mid_delta_auc",
                    "low_delta_auc",
                    "high_auc_drop",
                    "mid_auc_drop",
                    "low_auc_drop",
                    "delta_flip_density",
                    "flip_density_ratio",
                    "delta_auc_density",
                    "auc_drop_density",
                ],
            ),
            "aggregate_class_conditional_composition_rows": aggregate_rows(
                class_conditional_composition_rows,
                group_keys=["feature_group", "density_bin", "density_definition"],
                metric_keys=[
                    "n_samples",
                    "n_benign",
                    "n_malware",
                    "benign_to_malware_ratio",
                    "n_packed_malware",
                    "packed_ratio",
                ],
            ) if class_conditional_composition_rows else [],
            "aggregate_class_conditional_seed_summary_rows": aggregate_rows(
                class_conditional_seed_summary_rows,
                group_keys=["feature_group", "density_bin", "density_definition", "perturbation_type", "strength"],
                metric_keys=[
                    "n_samples",
                    "n_masked_features",
                    "flip_rate_mean",
                    "benign_to_malware_rate_overall_mean",
                    "malware_to_benign_rate_overall_mean",
                    "benign_to_malware_rate_cond_mean",
                    "malware_to_benign_rate_cond_mean",
                    "mean_abs_probability_shift_mean",
                    "mean_signed_probability_shift_mean",
                    "mean_margin_shift_mean",
                    "baseline_auc_mean",
                    "perturbed_auc_mean",
                    "delta_auc_mean",
                    "auc_drop_mean",
                ],
                passthrough_keys=["thresholds_json"],
            ) if class_conditional_seed_summary_rows else [],
            "aggregate_class_conditional_amplification_rows": aggregate_rows(
                class_conditional_amplification_rows,
                group_keys=["feature_group", "density_definition", "perturbation_type", "strength"],
                metric_keys=[
                    "high_flip_rate",
                    "mid_flip_rate",
                    "low_flip_rate",
                    "high_delta_auc",
                    "mid_delta_auc",
                    "low_delta_auc",
                    "high_auc_drop",
                    "mid_auc_drop",
                    "low_auc_drop",
                    "delta_flip_density",
                    "flip_density_ratio",
                    "delta_auc_density",
                    "auc_drop_density",
                ],
            ) if class_conditional_amplification_rows else [],
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
    for key in [
        "density_rows",
        "composition_rows",
        "trial_rows",
        "class_conditional_composition_rows",
        "class_conditional_trial_rows",
        "seed_summary_rows",
        "amplification_rows",
        "aggregate_composition_rows",
        "aggregate_seed_summary_rows",
        "aggregate_amplification_rows",
        "class_conditional_seed_summary_rows",
        "class_conditional_amplification_rows",
        "aggregate_class_conditional_composition_rows",
        "aggregate_class_conditional_seed_summary_rows",
        "aggregate_class_conditional_amplification_rows",
    ]:
        rows = results[key]
        if rows:
            write_csv(output_dir / f"{key}.csv", list(rows[0].keys()), rows)

    summary = {
        "config": results["config"],
        "metric_notes": results["metric_notes"],
        "n_density_rows": len(results["density_rows"]),
        "n_composition_rows": len(results["composition_rows"]),
        "n_trial_rows": len(results["trial_rows"]),
        "n_class_conditional_composition_rows": len(results["class_conditional_composition_rows"]),
        "n_class_conditional_trial_rows": len(results["class_conditional_trial_rows"]),
        "n_seed_summary_rows": len(results["seed_summary_rows"]),
        "n_amplification_rows": len(results["amplification_rows"]),
        "n_class_conditional_seed_summary_rows": len(results["class_conditional_seed_summary_rows"]),
        "n_class_conditional_amplification_rows": len(results["class_conditional_amplification_rows"]),
    }
    (output_dir / "results_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RQ3 Block 2 density-conditioned perturbation fragility analysis.")
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
