from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np


DEFAULT_CANDIDATE_DIMENSIONS: tuple[int, ...] = (16, 32, 64, 128)
DEFAULT_FAMILIES: tuple[str, ...] = ("header", "section", "imports", "strings")
DEFAULT_METRICS: tuple[str, ...] = ("auc", "mix_at_10", "js_divergence", "same_family_rate_at_10")


@dataclass(frozen=True)
class ExperimentConfig:
    model: str
    view: str
    seeds: tuple[int, ...]
    candidate_dimensions: tuple[int, ...] = DEFAULT_CANDIDATE_DIMENSIONS
    families: tuple[str, ...] = DEFAULT_FAMILIES
    subset_repeats: int = 10
    null_repeats: int = 50
    num_workers: int = 1


@dataclass(frozen=True)
class MetricRow:
    model: str
    view: str
    seed: int
    family: str
    dimension: int
    subset_repeat: int
    sampling_mode: str
    subset_indices_hash: str
    auc: float
    mix_at_10: float
    js_divergence: float
    same_family_rate_at_10: float


@dataclass(frozen=True)
class NullRow:
    model: str
    view: str
    seed: int
    dimension: int
    null_repeat: int
    control_type: str
    subset_indices_hash: str
    auc: float
    mix_at_10: float
    js_divergence: float
    same_family_rate_at_10: float


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_common_dimensions(
    family_to_size: Mapping[str, int],
    candidate_dimensions: Sequence[int],
    families: Sequence[str],
) -> list[int]:
    common_limit = min(int(family_to_size[family]) for family in families)
    resolved = sorted({int(d) for d in candidate_dimensions if int(d) <= common_limit})
    if not resolved:
        raise ValueError(
            "No candidate dimensions fit within the smallest family size. "
            f"Smallest available size is {common_limit}."
        )
    return resolved


def make_rng(seed: int, *parts: object) -> np.random.Generator:
    payload = "|".join([str(seed), *(str(part) for part in parts)])
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    mixed = int.from_bytes(digest, byteorder="little", signed=False)
    return np.random.default_rng(mixed)


def sample_without_replacement(
    candidates: Sequence[int],
    size: int,
    seed: int,
    *parts: object,
) -> np.ndarray:
    if size > len(candidates):
        raise ValueError(f"Requested sample size {size} exceeds candidate count {len(candidates)}.")
    rng = make_rng(seed, *parts)
    return np.sort(rng.choice(np.asarray(candidates, dtype=int), size=size, replace=False))


def subset_indices_hash(indices: Sequence[int]) -> str:
    arr = np.asarray(indices, dtype=np.int64)
    payload = ",".join(str(int(v)) for v in arr.tolist())
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()


def mean_and_std(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def mean_ci(
    values: Sequence[float],
    confidence: float = 0.95,
    *,
    bootstrap_repeats: int = 10000,
    rng_seed: int = 0,
) -> tuple[float, float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(arr.mean())
    if arr.size == 1:
        return mean, mean, mean
    rng = make_rng(rng_seed, "bootstrap_mean_ci", arr.size, bootstrap_repeats, confidence)
    sample_idx = rng.integers(0, arr.size, size=(bootstrap_repeats, arr.size))
    boot_means = arr[sample_idx].mean(axis=1)
    alpha = 1.0 - confidence
    lo = float(np.quantile(boot_means, alpha / 2.0))
    hi = float(np.quantile(boot_means, 1.0 - alpha / 2.0))
    return mean, lo, hi


def aggregate_metric_rows_by_seed(rows: Sequence[MetricRow]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int, str, int, str], list[MetricRow]] = {}
    for row in rows:
        key = (row.model, row.view, row.seed, row.family, row.dimension, row.sampling_mode)
        grouped.setdefault(key, []).append(row)

    out: list[dict[str, object]] = []
    for key, group_rows in sorted(grouped.items()):
        model, view, seed, family, dimension, sampling_mode = key
        out.append(
            {
                "model": model,
                "view": view,
                "seed": seed,
                "family": family,
                "dimension": dimension,
                "sampling_mode": sampling_mode,
                "subset_indices_hash": "seed_level_aggregate",
                "n_subset_repeats": len(group_rows),
                "auc_mean": np.mean([r.auc for r in group_rows]),
                "mix_at_10_mean": np.mean([r.mix_at_10 for r in group_rows]),
                "js_divergence_mean": np.mean([r.js_divergence for r in group_rows]),
                "same_family_rate_at_10_mean": np.mean([r.same_family_rate_at_10 for r in group_rows]),
            }
        )
    return out


def aggregate_null_rows_by_seed(rows: Sequence[NullRow]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int, int, str], list[NullRow]] = {}
    for row in rows:
        key = (row.model, row.view, row.seed, row.dimension, row.control_type)
        grouped.setdefault(key, []).append(row)

    out: list[dict[str, object]] = []
    for key, group_rows in sorted(grouped.items()):
        model, view, seed, dimension, control_type = key
        out.append(
            {
                "model": model,
                "view": view,
                "seed": seed,
                "dimension": dimension,
                "control_type": control_type,
                "subset_indices_hash": "seed_level_aggregate",
                "n_null_repeats": len(group_rows),
                "auc_mean": np.mean([r.auc for r in group_rows]),
                "mix_at_10_mean": np.mean([r.mix_at_10 for r in group_rows]),
                "js_divergence_mean": np.mean([r.js_divergence for r in group_rows]),
                "same_family_rate_at_10_mean": np.mean([r.same_family_rate_at_10 for r in group_rows]),
            }
        )
    return out


def compare_family_against_null(
    family_seed_rows: Sequence[Mapping[str, object]],
    null_seed_rows: Sequence[Mapping[str, object]],
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> list[dict[str, object]]:
    null_lookup = {
        (
            str(row["model"]),
            str(row["view"]),
            int(row["seed"]),
            int(row["dimension"]),
        ): row
        for row in null_seed_rows
    }
    comparisons: list[dict[str, object]] = []
    for fam in family_seed_rows:
        key = (
            str(fam["model"]),
            str(fam["view"]),
            int(fam["seed"]),
            int(fam["dimension"]),
        )
        null = null_lookup.get(key)
        if null is None:
            continue
        for metric in metrics:
            family_value = float(fam[f"{metric}_mean"])
            null_value = float(null[f"{metric}_mean"])
            comparisons.append(
                {
                    "model": fam["model"],
                    "view": fam["view"],
                    "seed": fam["seed"],
                    "family": fam["family"],
                    "dimension": fam["dimension"],
                    "metric": metric,
                    "family_seed_mean": family_value,
                    "null_seed_mean": null_value,
                    "paired_difference": family_value - null_value,
                }
            )
    return comparisons


def summarize_paired_differences(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, int, str], list[float]] = {}
    for row in rows:
        key = (
            str(row["model"]),
            str(row["view"]),
            str(row["family"]),
            int(row["dimension"]),
            str(row["metric"]),
        )
        grouped.setdefault(key, []).append(float(row["paired_difference"]))

    out: list[dict[str, object]] = []
    for key, diffs in sorted(grouped.items()):
        model, view, family, dimension, metric = key
        mean, ci_lo, ci_hi = mean_ci(
            diffs,
            rng_seed=int.from_bytes(
                hashlib.blake2b(
                    f"{model}|{view}|{family}|{dimension}|{metric}".encode("utf-8"),
                    digest_size=8,
                ).digest(),
                byteorder="little",
                signed=False,
            ),
        )
        std_mean, std_std = mean_and_std(diffs)
        out.append(
            {
                "model": model,
                "view": view,
                "family": family,
                "dimension": dimension,
                "metric": metric,
                "n_protocol_seeds": len(diffs),
                "paired_difference_mean": mean,
                "paired_difference_ci_lo": ci_lo,
                "paired_difference_ci_hi": ci_hi,
                "paired_difference_std": std_std,
            }
        )
    return out


def percentile_of_score(score: float, null_distribution: Sequence[float]) -> float:
    arr = np.asarray(list(null_distribution), dtype=float)
    if arr.size == 0:
        return float("nan")
    return float((arr <= score).mean())
