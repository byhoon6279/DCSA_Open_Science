#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


FEATURE_GROUPS = ["header", "section", "imports", "strings", "all"]


@dataclass(frozen=True)
class FamilySample:
    sample: Any
    family: str
    family_confidence: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("canonical_rq1_common", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load common module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def load_family_file(
    path: Path,
    common: Any,
    packer_filter: str,
    family_confidence_min: float | None,
) -> tuple[list[FamilySample], dict[str, Any]]:
    digest = hashlib.sha256()
    samples: list[FamilySample] = []
    total_rows = 0
    malware_rows = 0
    matched_packer_rows = 0
    with path.open("rb") as handle:
        for raw_line in handle:
            digest.update(raw_line)
            line = raw_line.strip()
            if not line:
                continue
            total_rows += 1
            row = json.loads(line)
            if row.get("label") != 1:
                continue
            malware_rows += 1
            if not common.matches_packer_filter(row, packer_filter):
                continue
            matched_packer_rows += 1
            family = row.get("family")
            if not family:
                continue
            confidence = row.get("family_confidence")
            if family_confidence_min is not None and (
                confidence is None or float(confidence) < family_confidence_min
            ):
                continue
            sample = common.build_sample(row, path)
            if sample is None:
                continue
            samples.append(
                FamilySample(
                    sample=sample,
                    family=str(family),
                    family_confidence=None if confidence is None else float(confidence),
                )
            )
    return samples, {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "size_bytes": path.stat().st_size,
        "total_rows": total_rows,
        "malware_rows": malware_rows,
        "packer_filter": packer_filter,
        "matched_packer_rows": matched_packer_rows,
        "usable_family_rows": len(samples),
    }


def filter_families(
    train_samples: Sequence[FamilySample],
    test_samples_by_week: dict[str, list[FamilySample]],
    min_train_family_count: int,
    top_n_families: int | None,
) -> set[str]:
    counts = Counter(sample.family for sample in train_samples)
    eligible = {family for family, count in counts.items() if count >= min_train_family_count}
    if top_n_families is not None:
        ranked = [family for family, _ in counts.most_common() if family in eligible]
        eligible = set(ranked[:top_n_families])
    in_test = {sample.family for rows in test_samples_by_week.values() for sample in rows}
    return eligible & in_test


def subsample_by_family(
    samples: Sequence[FamilySample],
    allowed_families: set[str],
    seed: int,
    max_per_family: int | None,
) -> list[FamilySample]:
    grouped: dict[str, list[FamilySample]] = defaultdict(list)
    for sample in samples:
        if sample.family in allowed_families:
            grouped[sample.family].append(sample)
    rng = np.random.default_rng(seed)
    selected: list[FamilySample] = []
    for family in sorted(grouped):
        rows = grouped[family]
        if max_per_family is not None and len(rows) > max_per_family:
            indices = rng.choice(len(rows), size=max_per_family, replace=False)
            rows = [rows[int(index)] for index in indices]
        selected.extend(rows)
    rng.shuffle(selected)
    return selected


def selected_manifest_rows(
    view: str,
    split: str,
    seed: int,
    week: str,
    samples: Sequence[FamilySample],
) -> Iterable[dict[str, Any]]:
    for position, wrapped in enumerate(samples):
        sample = wrapped.sample
        yield {
            "view": view,
            "split": split,
            "seed": seed,
            "week": week,
            "position": position,
            "sha256": sample.sha256,
            "family": wrapped.family,
            "source_path": sample.source_path,
        }


def explicit_nonself_neighbors(X: np.ndarray, max_k: int) -> np.ndarray:
    n_samples = len(X)
    if n_samples <= 1:
        return np.empty((n_samples, 0), dtype=int)
    effective_k = min(max_k, n_samples - 1)
    # Request one extra candidate, then remove the query row by identity rather
    # than assuming that self is always returned in column zero.
    search_k = min(n_samples, effective_k + 1)
    nn = NearestNeighbors(n_neighbors=search_k, metric="euclidean")
    nn.fit(X)
    indices = nn.kneighbors(X, return_distance=False)
    output = np.empty((n_samples, effective_k), dtype=int)
    for row_index, candidates in enumerate(indices):
        nonself = candidates[candidates != row_index]
        if len(nonself) < effective_k:
            # This is only expected under unusual exact-tie behavior. Query all
            # rows to guarantee k explicitly non-self neighbors.
            all_candidates = nn.kneighbors(
                X[row_index : row_index + 1], n_neighbors=n_samples, return_distance=False
            )[0]
            nonself = all_candidates[all_candidates != row_index]
        output[row_index] = nonself[:effective_k]
    return output


def same_family_rates(X: np.ndarray, y: np.ndarray, k_values: Sequence[int]) -> dict[int, float]:
    neighbors = explicit_nonself_neighbors(X, max(k_values))
    rates: dict[int, float] = {}
    for k in k_values:
        effective_k = min(int(k), max(0, len(X) - 1))
        if effective_k == 0:
            rates[int(k)] = 0.0
        else:
            rates[int(k)] = float(
                (y[neighbors[:, :effective_k]] == y[:, None]).mean()
            )
    return rates


def fit_models(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    models: Sequence[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    fitted: dict[str, Any] = {}
    for model in models:
        if model == "lr":
            estimator = LogisticRegression(
                solver="saga",
                class_weight="balanced",
                penalty="l2",
                C=0.3,
                max_iter=3000,
                tol=1e-3,
                random_state=seed,
            )
        elif model == "lightgbm":
            from lightgbm import LGBMClassifier

            estimator = LGBMClassifier(
                objective="multiclass",
                num_class=int(len(np.unique(y))),
                class_weight="balanced",
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                n_jobs=int(config["lightgbm_n_jobs"]),
                verbosity=-1,
            )
        elif model == "rf":
            estimator = RandomForestClassifier(
                n_estimators=int(config["rf_n_estimators"]),
                random_state=seed,
                n_jobs=int(config["rf_n_jobs"]),
                class_weight=None,
            )
        else:
            raise ValueError(f"Unknown model: {model}")
        estimator.fit(X, y)
        fitted[model] = estimator
    return fitted


def run_view(
    view: str,
    packer_filter: str,
    protocol: dict[str, Any],
    config: dict[str, Any],
    common: Any,
    output_dir: Path,
) -> None:
    data_root = Path(config["data_root"])
    platform_name = protocol["platform"]
    train_paths = common.week_paths(data_root, platform_name, protocol["train_weeks"], "train")
    test_paths = {
        week: common.week_paths(data_root, platform_name, [week], "test")[0]
        for week in protocol["test_weeks"]
    }
    confidence_min = protocol.get("family_confidence_min")
    input_rows: list[dict[str, Any]] = []
    all_train: list[FamilySample] = []
    for path in tqdm(train_paths, desc=f"{view}: load train files", unit="file"):
        rows, metadata = load_family_file(path, common, packer_filter, confidence_min)
        all_train.extend(rows)
        input_rows.append({"view": view, "split": "train", "week": path.stem, **metadata})
    tests: dict[str, list[FamilySample]] = {}
    for week, path in tqdm(test_paths.items(), desc=f"{view}: load test files", unit="file"):
        rows, metadata = load_family_file(path, common, packer_filter, confidence_min)
        tests[week] = rows
        input_rows.append({"view": view, "split": "test", "week": week, **metadata})
    allowed = filter_families(
        all_train,
        tests,
        int(protocol["min_train_family_count"]),
        protocol.get("top_n_families"),
    )
    view_dir = output_dir / view
    view_dir.mkdir(parents=True, exist_ok=True)
    write_csv(view_dir / "input_file_manifest.csv", input_rows)

    selection_rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    k_values = [int(k) for k in config["k_values"]]
    groups = [group for group in FEATURE_GROUPS if group in protocol["feature_groups"]]
    scaler_dir = view_dir / "scalers"
    scaler_dir.mkdir(exist_ok=True)

    for seed in protocol["seeds"]:
        seed = int(seed)
        train = subsample_by_family(
            all_train, allowed, seed, protocol.get("max_train_per_family")
        )
        family_set = {sample.family for sample in train}
        family_to_idx = {family: index for index, family in enumerate(sorted(family_set))}
        selection_rows.extend(selected_manifest_rows(view, "train", seed, "all_train", train))
        prepared_tests: dict[str, list[FamilySample]] = {}
        for week_index, week in enumerate(protocol["test_weeks"]):
            selected = subsample_by_family(
                tests[week],
                family_set,
                seed + week_index + 1,
                protocol.get("max_test_per_family"),
            )
            selected = [sample for sample in selected if sample.family in family_set]
            prepared_tests[week] = selected
            selection_rows.extend(selected_manifest_rows(view, "test", seed, week, selected))

        for group in groups:
            X_train_raw = common.select_feature_group([row.sample for row in train], group)
            y_train = np.asarray([family_to_idx[row.family] for row in train], dtype=int)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train_raw)
            np.savez_compressed(
                scaler_dir / f"seed_{seed}_{group}.npz",
                mean=scaler.mean_,
                scale=scaler.scale_,
                var=scaler.var_,
                n_samples_seen=np.asarray(scaler.n_samples_seen_),
            )
            fitted = fit_models(X_train, y_train, seed, config["models"], config)
            train_hash = array_sha256(X_train)
            for week in protocol["test_weeks"]:
                test = prepared_tests[week]
                X_test_raw = common.select_feature_group([row.sample for row in test], group)
                y_test = np.asarray([family_to_idx[row.family] for row in test], dtype=int)
                X_test = scaler.transform(X_test_raw)
                geometry = same_family_rates(X_test, y_test, k_values)
                test_hash = array_sha256(X_test)
                matrix_rows.append(
                    {
                        "view": view,
                        "seed": seed,
                        "week": week,
                        "feature_group": group,
                        "n_train": len(train),
                        "n_test": len(test),
                        "n_families": len(family_set),
                        "train_matrix_sha256": train_hash,
                        "test_matrix_sha256": test_hash,
                        "scaler_mean_sha256": array_sha256(scaler.mean_),
                        "scaler_scale_sha256": array_sha256(scaler.scale_),
                    }
                )
                predictions = {
                    model: estimator.predict(np.asarray(X_test, dtype=np.float64))
                    for model, estimator in fitted.items()
                }
                for k in k_values:
                    for model in config["models"]:
                        weekly_rows.append(
                            {
                                "view": view,
                                "packer_filter": packer_filter,
                                "seed": seed,
                                "week": week,
                                "feature_group": group,
                                "k": k,
                                "model": model,
                                "n_train": len(train),
                                "n_test": len(test),
                                "n_families": len(family_set),
                                "same_family_rate_at_k": geometry[k],
                                "family_macro_f1": float(
                                    f1_score(y_test, predictions[model], average="macro")
                                ),
                            }
                        )

    write_csv(view_dir / "selected_records.csv", selection_rows)
    write_csv(view_dir / "matrix_audit.csv", matrix_rows)
    write_csv(view_dir / "weekly_metrics.csv", weekly_rows)
    aggregate: list[dict[str, Any]] = []
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in weekly_rows:
        key = (row["view"], row["feature_group"], row["k"], row["model"])
        grouped[key].append(row)
    for (view_name, group, k, model), rows in sorted(grouped.items()):
        geometry_values = np.asarray([row["same_family_rate_at_k"] for row in rows])
        f1_values = np.asarray([row["family_macro_f1"] for row in rows])
        aggregate.append(
            {
                "view": view_name,
                "feature_group": group,
                "k": k,
                "model": model,
                "n_seed_week": len(rows),
                "same_family_rate_mean": float(geometry_values.mean()),
                "same_family_rate_std": float(geometry_values.std(ddof=1)),
                "family_macro_f1_mean": float(f1_values.mean()),
                "family_macro_f1_std": float(f1_values.std(ddof=1)),
            }
        )
    write_csv(view_dir / "aggregate_metrics.csv", aggregate)
    metadata = {
        "analysis_id": config["analysis_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "view": view,
        "packer_filter": packer_filter,
        "allowed_families": sorted(allowed),
        "allowed_family_count": len(allowed),
        "raw_usable_train_records": len(all_train),
        "raw_usable_test_records_by_week": {week: len(rows) for week, rows in tests.items()},
    }
    (view_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    protocol = json.loads(Path(config["protocol_config"]).read_text(encoding="utf-8"))
    common = load_module(Path(config["common_module"]))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = {
        **config,
        "protocol": protocol,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    try:
        import lightgbm

        resolved["runtime"]["lightgbm"] = lightgbm.__version__
    except ModuleNotFoundError:
        resolved["runtime"]["lightgbm"] = None
    (output_dir / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2), encoding="utf-8"
    )
    for view, packer_filter in config["views"].items():
        run_view(view, packer_filter, protocol, config, common, output_dir)


if __name__ == "__main__":
    main()
