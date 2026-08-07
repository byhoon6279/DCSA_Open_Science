#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACTS_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from common_library_adapter import CommonLibraryAdapter
from shared import ExperimentConfig, resolve_common_dimensions, sample_without_replacement, subset_indices_hash, write_csv
DEFAULT_BASE_CONFIG = (
    ARTIFACTS_ROOT
    / "5_1_representation_level_separability_disagreement"
    / "configs"
    / "LR"
    / "win32_all_train_all_test.json"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results" / "lr_wild_b_smoke"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RA-Q4 LR Wild(B) smoke test.")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dimension", type=int, default=16)
    parser.add_argument("--subset-repeats", type=int, default=1)
    args = parser.parse_args()

    adapter = CommonLibraryAdapter(
        base_rq1_config=args.base_config,
        model_type="logistic_regression",
        mix_k=10,
        show_progress=False,
    )
    families = ("header", "section", "imports", "strings")
    dimensions = resolve_common_dimensions(adapter.family_to_size, [args.dimension], families)
    if len(dimensions) != 1:
        raise ValueError(f"Requested dimension {args.dimension} is not supported across all families.")
    d = dimensions[0]

    rows: list[dict[str, object]] = []
    family_to_indices = adapter.resolve_family_indices()
    for family in families:
        candidates = family_to_indices[family]
        for subset_repeat in range(args.subset_repeats):
            indices = sample_without_replacement(
                candidates,
                d,
                args.seed,
                "family_random",
                family,
                d,
                subset_repeat,
            )
            metrics = adapter.run_subset_metrics(
                family=family,
                selected_indices=indices,
                seed=args.seed,
            )
            rows.append(
                {
                    "model": "logistic_regression",
                    "view": "wild_b",
                    "seed": args.seed,
                    "family": family,
                    "dimension": d,
                    "subset_repeat": subset_repeat,
                    "sampling_mode": "family_random",
                    "subset_indices_hash": subset_indices_hash(indices),
                    **metrics,
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "lr_wild_b_smoke_rows.csv", rows)
    print(f"Saved smoke rows to {args.output_dir / 'lr_wild_b_smoke_rows.csv'}")


if __name__ == "__main__":
    main()
