#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
RQ1_DIR = SCRIPT_DIR.parents[2]
_SHARED_LIB_DIR = SCRIPT_DIR.parents[3] / "artifacts" / "shared"
if str(_SHARED_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARED_LIB_DIR))

from common import (  # type: ignore  # noqa: E402
    FEATURE_GROUPS,
    build_sample,
    matches_packer_filter,
    select_feature_group,
    summarize,
    week_paths,
)


DEFAULT_OUTPUT_DIR = RQ1_DIR / "RQ1-3" / "results" / "experiment_results" / "family_aligned_baseline_journal"

warnings.filterwarnings(
    "ignore",
    message=r"X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
    module=r"sklearn\.utils\.validation",
)


@dataclass(frozen=True)
class FamilySample:
    sample: object
    family: str
    family_confidence: float | None


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def feature_rankings(score_map: Dict[str, float], higher_is_better: bool = True) -> Dict[str, int]:
    if higher_is_better:
        ordered = sorted(score_map.items(), key=lambda item: (-item[1], item[0]))
    else:
        ordered = sorted(score_map.items(), key=lambda item: (item[1], item[0]))
    return {group: idx + 1 for idx, (group, _) in enumerate(ordered)}


def load_family_samples_from_jsonl(
    path: Path,
    packer_filter: str = "all",
    family_confidence_min: float | None = None,
) -> List[FamilySample]:
    rows: List[FamilySample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("label") != 1:
                continue
            if not matches_packer_filter(row, packer_filter):
                continue
            family = row.get("family")
            if not family:
                continue
            family_conf = row.get("family_confidence")
            if family_confidence_min is not None:
                if family_conf is None or float(family_conf) < family_confidence_min:
                    continue
            base_sample = build_sample(row, path)
            if base_sample is None:
                continue
            rows.append(
                FamilySample(
                    sample=base_sample,
                    family=str(family),
                    family_confidence=None if family_conf is None else float(family_conf),
                )
            )
    return rows


def load_family_samples_from_paths(
    paths: Sequence[Path],
    packer_filter: str = "all",
    family_confidence_min: float | None = None,
) -> List[FamilySample]:
    out: List[FamilySample] = []
    for path in tqdm(paths, desc="Loading train weeks", unit="file"):
        out.extend(
            load_family_samples_from_jsonl(
                path,
                packer_filter=packer_filter,
                family_confidence_min=family_confidence_min,
            )
        )
    return out


def filter_families(
    train_samples: Sequence[FamilySample],
    test_samples_by_week: Dict[str, List[FamilySample]],
    min_train_family_count: int,
    top_n_families: int | None,
) -> set[str]:
    train_counts = Counter(sample.family for sample in train_samples)
    eligible = {family for family, count in train_counts.items() if count >= min_train_family_count}
    if top_n_families is not None:
        ranked = [family for family, _ in train_counts.most_common() if family in eligible]
        eligible = set(ranked[:top_n_families])
    # Keep only families that appear at least once in any test week after train filtering.
    families_in_test = {sample.family for rows in test_samples_by_week.values() for sample in rows}
    return eligible & families_in_test


def subsample_by_family(
    samples: Sequence[FamilySample],
    allowed_families: set[str],
    seed: int,
    max_per_family: int | None,
) -> List[FamilySample]:
    grouped: Dict[str, List[FamilySample]] = defaultdict(list)
    for sample in samples:
        if sample.family in allowed_families:
            grouped[sample.family].append(sample)

    rng = np.random.default_rng(seed)
    picked: List[FamilySample] = []
    for family in sorted(grouped):
        rows = grouped[family]
        if max_per_family is not None and len(rows) > max_per_family:
            idx = rng.choice(len(rows), size=max_per_family, replace=False)
            rows = [rows[i] for i in idx]
        picked.extend(rows)
    rng.shuffle(picked)
    return picked


def select_group(samples: Sequence[FamilySample], group: str) -> np.ndarray:
    base_samples = [sample.sample for sample in samples]
    return select_feature_group(base_samples, group)


def encode_families(samples: Sequence[FamilySample], family_to_idx: Dict[str, int]) -> np.ndarray:
    return np.asarray([family_to_idx[sample.family] for sample in samples], dtype=int)


def compute_same_family_rate_at_k(X: np.ndarray, y: np.ndarray, k: int) -> float:
    if len(X) <= 1:
        return 0.0
    effective_k = max(1, min(k, len(X) - 1))
    nn = NearestNeighbors(n_neighbors=effective_k + 1, metric="euclidean")
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    return float((y[indices[:, 1:]] == y[:, None]).mean())


def compute_same_family_rates_at_ks(
    X: np.ndarray,
    y: np.ndarray,
    k_values: Sequence[int],
) -> Dict[int, float]:
    if len(X) <= 1:
        return {int(k): 0.0 for k in k_values}

    unique_k_values = sorted({int(k) for k in k_values})
    max_k = max(unique_k_values)
    effective_max_k = max(1, min(max_k, len(X) - 1))
    nn = NearestNeighbors(n_neighbors=effective_max_k + 1, metric="euclidean")
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    neighbor_labels = y[indices[:, 1:]]

    rates: Dict[int, float] = {}
    for k in unique_k_values:
        effective_k = max(1, min(k, len(X) - 1))
        rates[k] = float((neighbor_labels[:, :effective_k] == y[:, None]).mean())
    return rates


def safe_family_silhouette(X: np.ndarray, y: np.ndarray, sample_size: int = 5000) -> float:
    unique = np.unique(y)
    if len(unique) < 2 or len(X) < 3:
        return float("nan")
    if len(X) > sample_size:
        return float(silhouette_score(X, y, sample_size=sample_size, random_state=0))
    return float(silhouette_score(X, y))


def fit_family_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    model_type: str,
    lightgbm_n_jobs: int,
    rf_n_estimators: int,
    rf_n_jobs: int,
) -> Any:
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train)
    if model_type == "logistic_regression":
        clf = LogisticRegression(
            solver="saga",
            class_weight="balanced",
            penalty="l2",
            C=0.3,
            max_iter=3000,
            tol=1e-3,
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
        n_classes = int(len(np.unique(y_train)))
        clf = LGBMClassifier(
            objective="multiclass",
            num_class=n_classes,
            class_weight="balanced",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=lightgbm_n_jobs,
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


def fit_group_pipeline(
    train_samples: Sequence[FamilySample],
    group: str,
    family_to_idx: Dict[str, int],
    seed: int,
    model_type: str,
    lightgbm_n_jobs: int,
    rf_n_estimators: int,
    rf_n_jobs: int,
) -> Tuple[StandardScaler, Any]:
    X_train_raw = select_group(train_samples, group)
    y_train = encode_families(train_samples, family_to_idx)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    clf = fit_family_classifier(
        X_train,
        y_train,
        seed,
        model_type=model_type,
        lightgbm_n_jobs=lightgbm_n_jobs,
        rf_n_estimators=rf_n_estimators,
        rf_n_jobs=rf_n_jobs,
    )
    return scaler, clf


def evaluate_group_all_k(
    test_samples: Sequence[FamilySample],
    group: str,
    family_to_idx: Dict[str, int],
    scaler: StandardScaler,
    clf: Any,
    k_values: Sequence[int],
) -> Dict[str, object]:
    X_test_raw = select_group(test_samples, group)
    y_test = encode_families(test_samples, family_to_idx)
    X_test = scaler.transform(X_test_raw)
    pred = clf.predict(np.asarray(X_test, dtype=np.float64))
    return {
        "same_family_rate_at_k_by_k": compute_same_family_rates_at_ks(X_test, y_test, k_values=k_values),
        "family_silhouette": safe_family_silhouette(X_test, y_test),
        "family_macro_f1": float(f1_score(y_test, pred, average="macro")),
    }


def summarize_group_metrics(
    run_results: List[Dict[str, object]],
    feature_groups: Sequence[str],
) -> Tuple[Dict[str, Dict[str, dict]], Dict[str, Dict[str, dict]]]:
    aggregate_metrics: Dict[str, Dict[str, dict]] = {}
    aggregate_ranks: Dict[str, Dict[str, dict]] = {
        "same_family_rate_at_k_rank": {},
        "family_silhouette_rank": {},
        "family_macro_f1_rank": {},
    }
    for group in feature_groups:
        aggregate_metrics[group] = {
            metric: summarize([run["per_group"][group][metric] for run in run_results])
            for metric in ["same_family_rate_at_k", "family_silhouette", "family_macro_f1"]
        }
        for rank_metric in aggregate_ranks:
            aggregate_ranks[rank_metric][group] = summarize([run[rank_metric][group] for run in run_results])
    return aggregate_metrics, aggregate_ranks


def summarize_group_metrics_by_key(
    run_results: List[Dict[str, object]],
    key: str,
    feature_groups: Sequence[str],
) -> Dict[str, Dict[str, Dict[str, dict]]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for run in run_results:
        grouped[str(run[key])] = grouped.setdefault(str(run[key]), []) + [run]

    out: Dict[str, Dict[str, Dict[str, dict]]] = {}
    for key_value, rows in grouped.items():
        metrics, ranks = summarize_group_metrics(rows, feature_groups)
        out[key_value] = {"aggregate_metrics": metrics, "aggregate_ranks": ranks}
    return out


def weekly_rows(run_results: List[Dict[str, object]], feature_groups: Sequence[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for run in run_results:
        row: Dict[str, object] = {
            "seed": run["seed"],
            "k": run["k"],
            "test_week": run["test_week"],
            "n_train_samples": run["n_train_samples"],
            "n_test_samples": run["n_test_samples"],
            "n_train_families": run["n_train_families"],
            "n_test_families": run["n_test_families"],
        }
        for group in feature_groups:
            metrics = run["per_group"][group]
            row[f"{group}_same_family_rate_at_k"] = metrics["same_family_rate_at_k"]
            row[f"{group}_family_silhouette"] = metrics["family_silhouette"]
            row[f"{group}_family_macro_f1"] = metrics["family_macro_f1"]
            row[f"{group}_same_family_rate_at_k_rank"] = run["same_family_rate_at_k_rank"][group]
            row[f"{group}_family_silhouette_rank"] = run["family_silhouette_rank"][group]
            row[f"{group}_family_macro_f1_rank"] = run["family_macro_f1_rank"][group]
        rows.append(row)
    return rows


def aggregate_rows(results: dict) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for group in results["config"]["enabled_groups"]:
        for metric, stats in results["aggregate_metrics"][group].items():
            rows.append({"scope": "metric", "feature_group": group, "metric": metric, **stats})
        for metric, payload in results["aggregate_ranks"].items():
            rows.append({"scope": "rank", "feature_group": group, "metric": metric, **payload[group]})
    return rows


def aggregate_rows_by_grouping(
    grouped_results: Dict[str, Dict[str, Dict[str, dict]]],
    grouping_name: str,
    feature_groups: Sequence[str],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
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
            for metric, group_stats in payload["aggregate_ranks"].items():
                rows.append(
                    {
                        grouping_name: grouping_value,
                        "scope": "rank",
                        "feature_group": group,
                        "metric": metric,
                        **group_stats[group],
                    }
                )
    return rows


def run_from_config(config: dict, config_path: Path | None = None) -> dict:
    data_root = Path(config["data_root"])
    if not data_root.is_absolute() and config_path is not None:
        data_root = (config_path.parent / data_root).resolve()
    platform = config["platform"]
    packer_filter = config.get("packer_filter", "all")
    train_weeks = config["train_weeks"]
    test_weeks = config["test_weeks"]
    seeds = [int(seed) for seed in config.get("seeds", [config.get("seed", 42)])]
    family_confidence_min = config.get("family_confidence_min")
    min_train_family_count = int(config.get("min_train_family_count", 20))
    top_n_families = config.get("top_n_families")
    max_train_per_family = config.get("max_train_per_family")
    max_test_per_family = config.get("max_test_per_family")
    k_values = list(config.get("k_values", [config.get("k", 10)]))
    k_values = [int(k) for k in k_values]
    model_type = str(config.get("model_type", "logistic_regression"))
    lightgbm_n_jobs = int(config.get("lightgbm_n_jobs", 3))
    rf_n_estimators = int(config.get("rf_n_estimators", 500))
    rf_n_jobs = int(config.get("rf_n_jobs", -1))
    requested_groups = config.get("feature_groups", FEATURE_GROUPS)
    enabled_groups = [group for group in FEATURE_GROUPS if group in requested_groups]
    if not enabled_groups:
        raise ValueError("feature_groups must contain at least one supported feature group.")

    train_paths = week_paths(data_root, platform, train_weeks, "train")
    test_paths = {week: week_paths(data_root, platform, [week], "test")[0] for week in test_weeks}

    all_train_samples = load_family_samples_from_paths(
        train_paths,
        packer_filter=packer_filter,
        family_confidence_min=family_confidence_min,
    )
    raw_test_samples_by_week = {
        week: load_family_samples_from_jsonl(
            path,
            packer_filter=packer_filter,
            family_confidence_min=family_confidence_min,
        )
        for week, path in tqdm(
            test_paths.items(),
            desc="Loading test weeks",
            unit="week",
            total=len(test_paths),
        )
    }

    allowed_families = filter_families(
        train_samples=all_train_samples,
        test_samples_by_week=raw_test_samples_by_week,
        min_train_family_count=min_train_family_count,
        top_n_families=top_n_families,
    )

    run_results: List[Dict[str, object]] = []
    total_evals = len(seeds) * len(enabled_groups) * len(test_weeks)
    progress = tqdm(total=total_evals, desc="Evaluating group-week evaluations", unit="eval")
    for seed in seeds:
        train_samples = subsample_by_family(
            all_train_samples,
            allowed_families=allowed_families,
            seed=seed,
            max_per_family=max_train_per_family,
        )
        train_family_set = {sample.family for sample in train_samples}
        if len(train_family_set) < 2:
            progress.update(len(enabled_groups) * len(test_weeks))
            continue
        family_to_idx = {family: idx for idx, family in enumerate(sorted(train_family_set))}

        prepared_test_samples_by_week: Dict[str, List[FamilySample]] = {}
        test_family_counts_by_week: Dict[str, int] = {}
        for week_idx, week in enumerate(test_weeks):
            test_samples = subsample_by_family(
                raw_test_samples_by_week[week],
                allowed_families=train_family_set,
                seed=seed + week_idx + 1,
                max_per_family=max_test_per_family,
            )
            test_samples = [sample for sample in test_samples if sample.family in train_family_set]
            test_family_set = {sample.family for sample in test_samples}
            prepared_test_samples_by_week[week] = test_samples
            test_family_counts_by_week[week] = len(test_family_set)

        fitted_group_pipelines: Dict[str, Tuple[StandardScaler, Any]] = {}
        for group in enabled_groups:
            fitted_group_pipelines[group] = fit_group_pipeline(
                train_samples,
                group,
                family_to_idx,
                seed=seed,
                model_type=model_type,
                lightgbm_n_jobs=lightgbm_n_jobs,
                rf_n_estimators=rf_n_estimators,
                rf_n_jobs=rf_n_jobs,
            )

        for week in test_weeks:
            test_samples = prepared_test_samples_by_week[week]
            test_family_count = test_family_counts_by_week[week]
            if test_family_count < 2:
                progress.update(len(enabled_groups))
                continue

            per_group_base: Dict[str, Dict[str, object]] = {}
            for group in enabled_groups:
                scaler, clf = fitted_group_pipelines[group]
                per_group_base[group] = evaluate_group_all_k(
                    test_samples,
                    group,
                    family_to_idx,
                    scaler,
                    clf,
                    k_values=k_values,
                )
                progress.update(1)

            for k in k_values:
                per_group: Dict[str, Dict[str, float]] = {}
                for group in enabled_groups:
                    base_metrics = per_group_base[group]
                    per_group[group] = {
                        "same_family_rate_at_k": float(base_metrics["same_family_rate_at_k_by_k"][k]),
                        "family_silhouette": float(base_metrics["family_silhouette"]),
                        "family_macro_f1": float(base_metrics["family_macro_f1"]),
                    }

                same_family_rate_ranks = feature_rankings(
                    {group: metrics["same_family_rate_at_k"] for group, metrics in per_group.items()}
                )
                silhouette_ranks = feature_rankings(
                    {group: metrics["family_silhouette"] for group, metrics in per_group.items()}
                )
                macro_f1_ranks = feature_rankings(
                    {group: metrics["family_macro_f1"] for group, metrics in per_group.items()}
                )

                for group in FEATURE_GROUPS:
                    if group not in per_group:
                        per_group[group] = {
                            "same_family_rate_at_k": float("nan"),
                            "family_silhouette": float("nan"),
                            "family_macro_f1": float("nan"),
                        }
                        same_family_rate_ranks[group] = 0
                        silhouette_ranks[group] = 0
                        macro_f1_ranks[group] = 0

                run_results.append(
                    {
                        "seed": seed,
                        "k": k,
                        "test_week": week,
                        "n_train_samples": len(train_samples),
                        "n_test_samples": len(test_samples),
                        "n_train_families": len(train_family_set),
                        "n_test_families": test_family_count,
                        "per_group": per_group,
                        "same_family_rate_at_k_rank": same_family_rate_ranks,
                        "family_silhouette_rank": silhouette_ranks,
                        "family_macro_f1_rank": macro_f1_ranks,
                    }
                )
    progress.close()

    filtered_results = []
    for run in run_results:
        if any(not np.isnan(run["per_group"][group]["family_macro_f1"]) for group in enabled_groups):
            filtered_results.append(run)
    run_results = filtered_results

    aggregate_metrics, aggregate_ranks = summarize_group_metrics(run_results, enabled_groups)
    aggregate_by_k = summarize_group_metrics_by_key(run_results, "k", enabled_groups)
    return {
        "config": {
            **config,
            "resolved_train_paths": [str(path) for path in train_paths],
            "resolved_test_paths": {week: str(path) for week, path in test_paths.items()},
            "allowed_family_count": len(allowed_families),
            "allowed_families_preview": sorted(allowed_families)[:20],
            "enabled_groups": enabled_groups,
            "packer_filter": packer_filter,
            "model_type": model_type,
            "lightgbm_n_jobs": lightgbm_n_jobs,
            "rf_n_estimators": rf_n_estimators,
            "rf_n_jobs": rf_n_jobs,
            "k_values": k_values,
            "question": (
                "Do feature subsets preserve malware family structure within Win32 malware samples, "
                "and does that taxonomy-aligned ranking agree with the existing decision-level and "
                "representation-level subset rankings?"
            ),
            "stage_note": (
                "This is a taxonomy-aligned baseline, not a direct semantic separability result."
            ),
        },
        "run_results": run_results,
        "aggregate_metrics": aggregate_metrics,
        "aggregate_ranks": aggregate_ranks,
        "aggregate_by_k": aggregate_by_k,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the family-aligned separability baseline.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", choices=["logistic_regression", "lightgbm", "random_forest"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.model is not None:
        config["model_type"] = args.model
    results = run_from_config(config, config_path=args.config.resolve())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    enabled_groups = results["config"]["enabled_groups"]
    weekly = weekly_rows(results["run_results"], enabled_groups)
    if weekly:
        write_csv(args.output_dir / "weekly_results.csv", list(weekly[0].keys()), weekly)

    aggregate = aggregate_rows(results)
    if aggregate:
        write_csv(args.output_dir / "aggregate_summary.csv", list(aggregate[0].keys()), aggregate)

    aggregate_by_k = aggregate_rows_by_grouping(results["aggregate_by_k"], "k", enabled_groups)
    if aggregate_by_k:
        write_csv(args.output_dir / "aggregate_by_k.csv", list(aggregate_by_k[0].keys()), aggregate_by_k)

    print(f"Wrote results to {args.output_dir}")
    print(f"Run count: {len(results['run_results'])}")
    print(f"Allowed family count: {results['config']['allowed_family_count']}")
    print(f"Model: {results['config'].get('model_type', 'logistic_regression')}")
    for group in results["config"]["enabled_groups"]:
        macro_f1 = results["aggregate_metrics"][group]["family_macro_f1"]["mean"]
        purity = results["aggregate_metrics"][group]["same_family_rate_at_k"]["mean"]
        print(
            f"{group}: family_macro_f1={macro_f1:.4f}, "
            f"same_family_rate_at_k={purity:.4f}"
        )


if __name__ == "__main__":
    main()
