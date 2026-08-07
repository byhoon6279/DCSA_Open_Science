#!/usr/bin/env python3
from __future__ import annotations

import argparse
import multiprocessing as mp
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACTS_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from common_library_adapter import CommonLibraryAdapter
from shared import (
    DEFAULT_CANDIDATE_DIMENSIONS,
    DEFAULT_FAMILIES,
    ExperimentConfig,
    MetricRow,
    aggregate_metric_rows_by_seed,
    resolve_common_dimensions,
    sample_without_replacement,
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
        f"Unsupported model '{model}' in dimension-matched runner. "
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


def run_metric_pipeline(
    *,
    adapter: CommonLibraryAdapter,
    model: str,
    view: str,
    seed: int,
    family: str,
    dimension: int,
    subset_indices: Sequence[int],
    sampling_mode: str,
    subset_repeat: int,
) -> MetricRow:
    if view.strip().lower() != "wild_b":
        raise NotImplementedError(
            f"View '{view}' is not wired yet in the adapter-backed dimension-matched runner. "
            "Use wild_b first."
        )
    metrics = adapter.run_subset_metrics(
        family=family,
        selected_indices=subset_indices,
        seed=seed,
    )
    return MetricRow(
        model=normalize_model_name(model),
        view=view,
        seed=seed,
        family=family,
        dimension=dimension,
        subset_repeat=subset_repeat,
        sampling_mode=sampling_mode,
        subset_indices_hash=subset_indices_hash(subset_indices),
        auc=float(metrics["auc"]),
        mix_at_10=float(metrics["mix_at_10"]),
        js_divergence=float(metrics["js_divergence"]),
        same_family_rate_at_10=float(metrics["same_family_rate_at_10"]),
    )


def run_seed_dimension_jobs(
    *,
    config_path: Path,
    common_py: Path | None,
    mix_k: int,
    model: str,
    view: str,
    seed: int,
    families: Sequence[str],
    dimensions: Sequence[int],
    subset_repeats: int,
) -> list[dict[str, object]]:
    adapter = build_adapter(
        config_path=config_path,
        model=model,
        mix_k=mix_k,
        common_py=common_py,
    )
    family_to_indices = adapter.resolve_family_indices()
    raw_rows: list[dict[str, object]] = []
    for family in families:
        candidates = family_to_indices[family]
        for dimension in dimensions:
            for subset_repeat in range(subset_repeats):
                subset_indices = sample_without_replacement(
                    candidates,
                    dimension,
                    seed,
                    "family_random",
                    family,
                    dimension,
                    subset_repeat,
                )
                row = run_metric_pipeline(
                    adapter=adapter,
                    model=model,
                    view=view,
                    seed=seed,
                    family=family,
                    dimension=dimension,
                    subset_indices=subset_indices,
                    sampling_mode="family_random",
                    subset_repeat=subset_repeat,
                )
                raw_rows.append(row.__dict__)
    return raw_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RA-Q4 dimension-matched family subset audit.")
    parser.add_argument("--config", type=Path, required=True)
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

    raw_rows: list[dict[str, object]] = []
    if config.num_workers <= 1:
        for seed in config.seeds:
            raw_rows.extend(
                run_seed_dimension_jobs(
                    config_path=args.config,
                    common_py=args.common_py,
                    mix_k=args.mix_k,
                    model=config.model,
                    view=config.view,
                    seed=seed,
                    families=config.families,
                    dimensions=dimensions,
                    subset_repeats=config.subset_repeats,
                )
            )
    else:
        mp_ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=config.num_workers, mp_context=mp_ctx) as executor:
            futures = [
                executor.submit(
                    run_seed_dimension_jobs,
                    config_path=args.config,
                    common_py=args.common_py,
                    mix_k=args.mix_k,
                    model=config.model,
                    view=config.view,
                    seed=seed,
                    families=config.families,
                    dimensions=dimensions,
                    subset_repeats=config.subset_repeats,
                )
                for seed in config.seeds
            ]
            for future in futures:
                raw_rows.extend(future.result())

    metric_rows = [MetricRow(**row) for row in raw_rows]

    seed_summary = aggregate_metric_rows_by_seed(metric_rows)
    write_csv(args.output_dir / "dimension_matched_rows.csv", raw_rows)
    write_csv(args.output_dir / "dimension_matched_seed_summary.csv", seed_summary)


if __name__ == "__main__":
    main()
