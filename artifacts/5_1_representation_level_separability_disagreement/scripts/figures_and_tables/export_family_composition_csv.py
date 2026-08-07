"""
Build family composition tables for taxonomy-aligned baselines.

The table reconstructs the effective family composition used by the
family-aligned experiments under the same filtering and capping policy.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from family_aligned_baseline import (  # type: ignore  # noqa: E402
    FamilySample,
    filter_families,
    load_family_samples_from_jsonl,
)
from common import week_paths  # type: ignore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-config", type=Path, required=True)
    parser.add_argument("--unpacked-config", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def resolve_data_root(config: dict, config_path: Path) -> Path:
    data_root = Path(config["data_root"])
    if not data_root.is_absolute():
        data_root = (config_path.parent / data_root).resolve()
    return data_root


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def count_by_family(samples: Sequence[FamilySample]) -> Counter:
    return Counter(sample.family for sample in samples)


def capped_train_counts(
    train_samples: Sequence[FamilySample],
    allowed_families: set[str],
    max_train_per_family: int | None,
) -> Dict[str, int]:
    counts = count_by_family(train_samples)
    out: Dict[str, int] = {}
    for family in allowed_families:
        count = counts.get(family, 0)
        out[family] = min(count, max_train_per_family) if max_train_per_family is not None else count
    return out


def capped_test_counts_across_weeks(
    test_samples_by_week: Dict[str, list[FamilySample]],
    allowed_families: set[str],
    max_test_per_family: int | None,
) -> Dict[str, int]:
    totals: Dict[str, int] = defaultdict(int)
    for samples in test_samples_by_week.values():
        counts = count_by_family([sample for sample in samples if sample.family in allowed_families])
        for family in allowed_families:
            count = counts.get(family, 0)
            totals[family] += min(count, max_test_per_family) if max_test_per_family is not None else count
    return dict(totals)


def build_setting_counts(config_path: Path) -> dict:
    config = load_config(config_path)
    data_root = resolve_data_root(config, config_path)
    platform = config["platform"]
    packer_filter = config.get("packer_filter", "all")
    family_confidence_min = config.get("family_confidence_min")
    train_weeks = config["train_weeks"]
    test_weeks = config["test_weeks"]
    min_train_family_count = int(config.get("min_train_family_count", 20))
    top_n_families = config.get("top_n_families")
    max_train_per_family = config.get("max_train_per_family")
    max_test_per_family = config.get("max_test_per_family")

    train_paths = week_paths(data_root, platform, train_weeks, "train")
    test_paths = {week: week_paths(data_root, platform, [week], "test")[0] for week in test_weeks}

    train_samples: list[FamilySample] = []
    for path in train_paths:
        train_samples.extend(
            load_family_samples_from_jsonl(
                path,
                packer_filter=packer_filter,
                family_confidence_min=family_confidence_min,
            )
        )

    test_samples_by_week = {
        week: load_family_samples_from_jsonl(
            path,
            packer_filter=packer_filter,
            family_confidence_min=family_confidence_min,
        )
        for week, path in test_paths.items()
    }

    allowed_families = filter_families(
        train_samples=train_samples,
        test_samples_by_week=test_samples_by_week,
        min_train_family_count=min_train_family_count,
        top_n_families=top_n_families,
    )

    train_raw_counts = count_by_family(train_samples)
    test_raw_counts = count_by_family(
        [sample for rows in test_samples_by_week.values() for sample in rows]
    )
    train_effective_counts = capped_train_counts(train_samples, allowed_families, max_train_per_family)
    test_effective_counts = capped_test_counts_across_weeks(
        test_samples_by_week,
        allowed_families,
        max_test_per_family,
    )

    return {
        "allowed_families": allowed_families,
        "train_raw_counts": train_raw_counts,
        "test_raw_counts": test_raw_counts,
        "train_effective_counts": train_effective_counts,
        "test_effective_counts": test_effective_counts,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    original = build_setting_counts(args.original_config.resolve())
    unpacked = build_setting_counts(args.unpacked_config.resolve())

    families = sorted(original["allowed_families"] | unpacked["allowed_families"])
    rows: list[dict[str, object]] = []

    for family in families:
        rows.append(
            {
                "family": family,
                "original_included": family in original["allowed_families"],
                "original_train_raw": original["train_raw_counts"].get(family, 0),
                "original_train_effective": original["train_effective_counts"].get(family, 0),
                "original_test_raw": original["test_raw_counts"].get(family, 0),
                "original_test_effective": original["test_effective_counts"].get(family, 0),
                "unpacked_included": family in unpacked["allowed_families"],
                "unpacked_train_raw": unpacked["train_raw_counts"].get(family, 0),
                "unpacked_train_effective": unpacked["train_effective_counts"].get(family, 0),
                "unpacked_test_raw": unpacked["test_raw_counts"].get(family, 0),
                "unpacked_test_effective": unpacked["test_effective_counts"].get(family, 0),
            }
        )

    write_csv(args.output_csv, rows)
    print(f"Wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
