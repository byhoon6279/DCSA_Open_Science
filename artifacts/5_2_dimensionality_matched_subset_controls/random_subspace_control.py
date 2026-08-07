#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACTS_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from common_library_adapter import CommonLibraryAdapter
from shared import (
    DEFAULT_CANDIDATE_DIMENSIONS,
    DEFAULT_FAMILIES,
    ExperimentConfig,
    NullRow,
    aggregate_null_rows_by_seed,
    compare_family_against_null,
    percentile_of_score,
    resolve_common_dimensions,
    sample_without_replacement,
    summarize_paired_differences,
    subset_indices_hash,
    write_csv,
)


def load_config(path: Path) -> ExperimentConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ExperimentConfig(
        model=str(payload["model"]),
        view=str(payload["view"]),
        seeds=tuple(int(seed) for seed in payload["seeds"]),
        candidate_dimensions=tuple(int(d) for d in payload.get("candidate_dimensions", DEFAULT_CANDIDATE_DIMENSIONS)),
        families=tuple(str(name) for name in payload.get("families", DEFAULT_FAMILIES)),
        subset_repeats=int(payload.get("subset_repeats", 10)),
        null_repeats=int(payload.get("null_repeats", 50)),
        num_workers=int(payload.get("num_workers", 1)),
    )


def normalize_model_name(model: str) -> str:
    normalized = model.strip().lower()
    if normalized in {"lr", "logistic_regression", "logreg"}:
        return "logistic_regression"
    if normalized in {"lightgbm", "lgbm"}:
        return "lightgbm"
    raise NotImplementedError(
        f"Unsupported model '{model}' in random-subspace runner. "
        "Currently supported: lr/logistic_regression and lightgbm placeholder."
    )


def build_adapter(
    *,
    config_path: Path,
    model: str,
    mix_k: int,
    common_py: Path | None,
) -> CommonLibraryAdapter:
    model_name = normalize_model_name(model)
    base_rq1_config = (
        ARTIFACTS_ROOT
        / "5_1_representation_level_separability_disagreement"
        / "configs"
        / "LR"
        / "win32_all_train_all_test.json"
    )
    if model_name == "lightgbm":
        base_rq1_config = (
            ARTIFACTS_ROOT
            / "5_1_representation_level_separability_disagreement"
            / "configs"
            / "LightGBM"
            / "win32_all_train_all_test_lightgbm.json"
        )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if "base_rq1_config" in payload:
        base_rq1_config = Path(str(payload["base_rq1_config"])).expanduser()
    return CommonLibraryAdapter(
        base_rq1_config=base_rq1_config,
        common_py=common_py,
        model_type=model_name,
        mix_k=mix_k,
        show_progress=False,
    )


def run_null_metric_pipeline(
    *,
    adapter: CommonLibraryAdapter,
    model: str,
    view: str,
    seed: int,
    dimension: int,
    subset_indices: Sequence[int],
    null_repeat: int,
) -> NullRow:
    if view.strip().lower() != "wild_b":
        raise NotImplementedError(
            f"View '{view}' is not wired yet in the adapter-backed random-subspace runner. "
            "Use wild_b first."
        )
    metrics = adapter.run_global_subset_metrics(
        selected_indices=subset_indices,
        seed=seed,
    )
    return NullRow(
        model=normalize_model_name(model),
        view=view,
        seed=seed,
        dimension=dimension,
        null_repeat=null_repeat,
        control_type="global_unrestricted_random",
        subset_indices_hash=subset_indices_hash(subset_indices),
        auc=float(metrics["auc"]),
        mix_at_10=float(metrics["mix_at_10"]),
        js_divergence=float(metrics["js_divergence"]),
        same_family_rate_at_10=float(metrics["same_family_rate_at_10"]),
    )


def load_family_seed_summary(path: Path) -> list[dict[str, object]]:
    import csv

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def metric_column_name(metric: str) -> str:
    return {
        "auc": "auc",
        "mix_at_10": "mix_at_10",
        "js_divergence": "js_divergence",
        "same_family_rate_at_10": "same_family_rate_at_10",
    }[metric]


def run_seed_null_jobs(
    *,
    config_path: Path,
    common_py: Path | None,
    mix_k: int,
    model: str,
    view: str,
    seed: int,
    dimensions: Sequence[int],
    null_repeats: int,
    all_indices: Sequence[int],
) -> list[dict[str, object]]:
    adapter = build_adapter(
        config_path=config_path,
        model=model,
        mix_k=mix_k,
        common_py=common_py,
    )
    raw_rows: list[dict[str, object]] = []
    for dimension in dimensions:
        for null_repeat in range(null_repeats):
            subset_indices = sample_without_replacement(
                all_indices,
                dimension,
                seed,
                "global_unrestricted_random",
                dimension,
                null_repeat,
            )
            row = run_null_metric_pipeline(
                adapter=adapter,
                model=model,
                view=view,
                seed=seed,
                dimension=dimension,
                subset_indices=subset_indices,
                null_repeat=null_repeat,
            )
            if not row.subset_indices_hash:
                row = NullRow(
                    model=row.model,
                    view=row.view,
                    seed=row.seed,
                    dimension=row.dimension,
                    null_repeat=row.null_repeat,
                    control_type=row.control_type,
                    subset_indices_hash=subset_indices_hash(subset_indices),
                    auc=row.auc,
                    mix_at_10=row.mix_at_10,
                    js_divergence=row.js_divergence,
                    same_family_rate_at_10=row.same_family_rate_at_10,
                )
            raw_rows.append(row.__dict__)
    return raw_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RA-Q4 unrestricted random-subspace control.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--family-seed-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--common-py", type=Path, help="Optional explicit path to the shared common.py library")
    parser.add_argument("--mix-k", type=int, default=10)
    args = parser.parse_args()

    config = load_config(args.config)
    adapter = build_adapter(
        config_path=args.config,
        model=config.model,
        mix_k=args.mix_k,
        common_py=args.common_py,
    )
    family_to_indices = adapter.resolve_family_indices()
    family_to_size = {family: len(family_to_indices[family]) for family in config.families}
    dimensions = resolve_common_dimensions(family_to_size, config.candidate_dimensions, config.families)
    all_indices = adapter.resolve_all_feature_indices()

    raw_rows: list[dict[str, object]] = []
    if config.num_workers <= 1:
        for seed in config.seeds:
            raw_rows.extend(
                run_seed_null_jobs(
                    config_path=args.config,
                    common_py=args.common_py,
                    mix_k=args.mix_k,
                    model=config.model,
                    view=config.view,
                    seed=seed,
                    dimensions=dimensions,
                    null_repeats=config.null_repeats,
                    all_indices=all_indices,
                )
            )
    else:
        mp_ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=config.num_workers, mp_context=mp_ctx) as executor:
            futures = [
                executor.submit(
                    run_seed_null_jobs,
                    config_path=args.config,
                    common_py=args.common_py,
                    mix_k=args.mix_k,
                    model=config.model,
                    view=config.view,
                    seed=seed,
                    dimensions=dimensions,
                    null_repeats=config.null_repeats,
                    all_indices=all_indices,
                )
                for seed in config.seeds
            ]
            for future in futures:
                raw_rows.extend(future.result())

    null_rows = [NullRow(**row) for row in raw_rows]

    null_seed_summary = aggregate_null_rows_by_seed(null_rows)
    family_seed_summary = load_family_seed_summary(args.family_seed_summary)
    paired_rows = compare_family_against_null(family_seed_summary, null_seed_summary)
    paired_summary = summarize_paired_differences(paired_rows)

    percentile_rows: list[dict[str, object]] = []
    for fam in family_seed_summary:
        for metric in ("auc", "mix_at_10", "js_divergence", "same_family_rate_at_10"):
            null_dist = [
                float(getattr(row, metric_column_name(metric)))
                for row in null_rows
                if row.model == fam["model"]
                and row.view == fam["view"]
                and int(row.seed) == int(fam["seed"])
                and int(row.dimension) == int(fam["dimension"])
            ]
            family_seed_mean = float(fam[f"{metric}_mean"])
            percentile_rows.append(
                {
                    "model": fam["model"],
                    "view": fam["view"],
                    "seed": fam["seed"],
                    "family": fam["family"],
                    "dimension": fam["dimension"],
                    "metric": metric,
                    "family_seed_mean": family_seed_mean,
                    "family_percentile_in_null": percentile_of_score(family_seed_mean, null_dist),
                    "n_null_repeats": len(null_dist),
                }
            )

    write_csv(args.output_dir / "random_subspace_rows.csv", raw_rows)
    write_csv(args.output_dir / "random_subspace_seed_summary.csv", null_seed_summary)
    write_csv(args.output_dir / "family_vs_null_paired_rows.csv", paired_rows)
    write_csv(args.output_dir / "family_vs_null_paired_summary.csv", paired_summary)
    write_csv(args.output_dir / "family_vs_null_percentiles.csv", percentile_rows)


if __name__ == "__main__":
    main()
